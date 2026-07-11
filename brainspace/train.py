import warnings
import math
import os
import sys
import time
from typing import Optional, Callable
import random

import h5py
import numpy as np
from scipy.stats import qmc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange

from .models import StaticNN, DynamicNN
from .datasets import Dataset
from .internal.registry import format_duration
from .config import RunConfig, DatasetConfig, ModelConfig, TrainConfig


# ============================================================================
# Global Seed Manager
# ============================================================================

class SeedManager:
    """
    Unified interface to control all randomness across numpy, torch, and python.

    Usage:
        SeedManager.set_seed(42)  # Set all seeds
        # or with device specification
        SeedManager.set_seed(42, device='cuda')
    """

    @staticmethod
    def set_seed(seed: int, device: str = 'cpu') -> None:
        """
        Set seeds for numpy, torch, and python random for reproducibility.

        Args:
            seed: Seed value to use
            device: 'cpu', 'cuda', or None for both
        """
        # Python built-in random
        random.seed(seed)

        # NumPy
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)
        if device == 'cuda' or device is None:
            torch.cuda.manual_seed_all(seed)

        # For CUDA reproducibility
        if device == 'cuda' or device is None:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class Processor:
    def __init__(self,
                 dataset,
                 model,
                 epochs: int = 3000,
                 criterion=None,
                 optimizer=None,
                 scheduler=None,
                 visualizer=None,
                 batch_size: Optional[int] = None,
                 epoch_hooks: Optional[list] = None,
                 batch_sampler=None,
                 step_granularity: str = 'epoch',
                 *,
                 seed: Optional[int] = None,
                 dtype=torch.float32,
                 device=None):
        """
        Args:
            dataset: a pre-built Dataset instance (see datasets.Dataset /
                DatasetConfig.build()) providing train/test tensors and metadata.
            model: the model to train.
            epochs: number of epochs.
            criterion: the loss function to use for training.
            optimizer: the optimizer to use for training.
            scheduler: the scheduler to use for training.
            visualizer: optional Visualizer instance for streaming logging during training.
            batch_size: if set, train (with gradient accumulation) and evaluate in
                micro-batches of this size to bound peak memory. None (default) runs
                the full batch in a single pass (original behavior). Gradient
                accumulation is numerically equivalent to the full-batch gradient.
            epoch_hooks: optional list of callables ``hook(processor, epoch)`` run at
                the start of every training epoch (e.g. domain parameter schedules).
            batch_sampler: optional iterable yielding per-step training batches as
                index arrays, or ``(indices, weights)`` tuples for importance-weighted
                losses (``criterion(out, y, weights)``). Re-iterated each epoch.
                None (default) uses contiguous ``batch_size`` chunks.
            step_granularity: 'epoch' (default) accumulates gradients over all
                batches and steps the optimizer/scheduler once per epoch, exactly
                equivalent to a full-batch step; 'batch' steps after every batch
                (per-batch SGD, scheduler stepped per batch).
            seed (Optional[int]): random seed for reproducibility. If None, a random seed will be generated.
            dtype: logs type for model parameters and computations (default: torch.float32).
        """
        self._set_environment(seed=seed, dtype=dtype, device=device)

        x_range = dataset.x_range
        data_dim = dataset.data_dim
        N = dataset.N
        use_padding = dataset.use_padding
        ground_truth = dataset.ground_truth
        cot = dataset.cot_keep_fraction

        self.x_range = x_range
        self.data_dim = data_dim
        self.use_padding = use_padding
        self.epoch_hooks = list(epoch_hooks) if epoch_hooks else []
        self.batch_sampler = batch_sampler
        if step_granularity not in ('epoch', 'batch'):
            raise ValueError(f"step_granularity must be 'epoch' or 'batch', got {step_granularity!r}")
        self.step_granularity = step_granularity
        assert isinstance(model, (StaticNN, DynamicNN)), \
            "Processor supports StaticNN and DynamicNN architectures"
        if isinstance(model, StaticNN) and use_padding:
            raise ValueError("StaticNN models do not support padded (variable-length) data")
        assert isinstance(data_dim, (tuple, list)) and len(data_dim) == 2, \
            "data_dim must be (seq_len, feat_dim)"
        self.visualizer = visualizer

        if criterion is None:
            criterion = nn.MSELoss()

        self.x_train = dataset.x_train
        self.y_train = dataset.y_train
        self.x_test = dataset.x_test
        self.y_test = dataset.y_test
        self.train_mask = dataset.train_mask
        self.test_mask = dataset.test_mask
        self.y_train_intermediate = dataset.y_train_intermediate
        self.y_test_intermediate = dataset.y_test_intermediate
        self.train_cot_mask = dataset.train_cot_mask.float()
        self.test_cot_mask = dataset.test_cot_mask.float()
        self.train_sequence_lengths = dataset.train_sequence_lengths
        self.d = dataset.d

        # Test-set partition layout (random samples vs. axis-probe samples),
        # computed once by generate_dataset alongside x_test itself.
        self._x_test_random_n = dataset.x_test_random_n
        self._x_test_axis_start = dataset.x_test_axis_start
        self._x_test_axis_d = dataset.x_test_axis_d
        self._x_test_axis_end = dataset.x_test_axis_end

        self.N = N

        # Auto-detect intermediate logging capability
        self.intermediate_logging = (
            isinstance(model, DynamicNN) and
            getattr(ground_truth, 'has_intermediate', False) and
            not cot
        )

        # Chain-of-Thought supervision setup
        # cot: float in [0, 1] representing fraction of intermediate steps to keep
        # 0: no intermediate supervision, 1: full supervision
        self.cot_keep_fraction = (
            float(cot) if isinstance(cot, (float, int)) and not isinstance(cot, bool)
            else (1.0 if cot else 0.0)
        )
        self.cot = bool(self.cot_keep_fraction)  # True if any supervision
        if cot:
            if not hasattr(model, 'seq2seq'):
                raise ValueError("Model must have 'seq2seq' attribute for cot=True")
            model.pool = False
            model.seq2seq = True

        # Training
        self.model = model.to(self.device)
        self.criterion = criterion

        if not optimizer:
            optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        # Denominator for chunked intermediate-supervision training normalization
        # (fixed for the whole run; train_cot_mask is fixed at dataset construction).
        self._train_valid_total = self.train_cot_mask.sum().clamp(min=1)

        # Logging setup
        if use_padding:
            x_train_np = self.x_train.cpu().numpy()
            x_test_np = self.x_test.cpu().numpy()
        else:
            x_train_np = self.x_train.cpu().numpy().reshape(-1, self.d)
            x_test_np = self.x_test.cpu().numpy().reshape(-1, self.d)

        self.logs = {
            "x_train": x_train_np,
            "y_train": self.y_train.cpu().numpy().squeeze(),
            "x_test": x_test_np,
            "y_test": self.y_test.cpu().numpy().squeeze(),
            "train_loss": [],
            "test_loss": [],
            "f_test": [],
            "hidden_states": [],
        }
        if getattr(model, 'seq2seq', False):
            self.logs["step_train_losses"] = []
            self.logs["step_test_losses"] = []
        elif self.intermediate_logging:
            self.logs["step_test_losses"] = []
        self.metadata = {
            "dataset": dataset.metadata,
            "model": getattr(model, 'metadata', {"arch": model.__class__.__name__}),
            "train": {
                "optimizer": self.optimizer.__class__.__name__,
                "criterion": criterion.__class__.__name__,
                "scheduler": self.scheduler.__class__.__name__ if self.scheduler else None,
                "epochs": epochs,
            },
        }

        # Configure visualizer if provided
        if visualizer is not None:
            visualizer.attach_processor(self)


    @staticmethod
    def resolve_device(device):
        """Resolve a TrainConfig.device (None/str/torch.device) to a torch.device.

        None auto-detects CUDA, mirroring _set_environment, so that data and
        model always land on the same device.
        """
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            return torch.device(device)
        return device

    @classmethod
    def from_run_config(cls, config: RunConfig, visualizer=None, dataset=None) -> 'Processor':
        """
        Initialize Processor from a RunConfig instance.

        Handles all model/optimizer/dataset construction automatically.

        Args:
            config: the RunConfig to build from.
            visualizer: optional Visualizer to stream updates to.
            dataset: optional pre-built Dataset to reuse instead of rebuilding
                from ``config.data``. Dataset generation seeds only a *local*
                ``np.random.default_rng`` and never touches the global torch /
                numpy / python RNG, so the seed set below (before the model is
                built) leaves the model-initialization RNG state identical
                whether or not the dataset is rebuilt here. Injecting a cached
                dataset therefore yields bit-identical model init + training to
                the build-it path — as long as the caller passes a dataset that
                matches ``config.data`` and ``config.train.seed``/``cot``.
        """
        dc, mc, tc = config.data, config.model, config.train

        # Resolve device upfront so data and model land on the same device.
        resolved_device = cls.resolve_device(tc.device)

        # Seed before construction so both data generation and model weight
        # initialization are reproducible for a given tc.seed. (Processor.__init__
        # re-seeds afterward to make the training loop itself reproducible.)
        if tc.seed is not None:
            SeedManager.set_seed(tc.seed, device=str(resolved_device))

        # Build dataset (once), seeded from the run seed — unless the caller
        # supplied a matching pre-built dataset to reuse (see docstring).
        if dataset is None:
            dataset = dc.build(seed=tc.seed, device=resolved_device, cot=tc.cot)

        # Build model using inferred input_dim. DynamicNN archs consume
        # per-timestep features; StaticNN archs consume the flattened sample.
        input_dim = dataset.feature_dim
        if isinstance(mc.arch, type) and issubclass(mc.arch, StaticNN):
            input_dim = math.prod(dataset.data_dim)
        model = mc.build(input_dim=input_dim)

        # Build optimizer + scheduler from TrainConfig
        optimizer, scheduler = tc.build_optimizer_and_scheduler(model)

        return cls(
            dataset=dataset,
            model=model,
            epochs=tc.epochs,
            criterion=tc.criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            visualizer=visualizer,
            batch_size=tc.batch_size,
            epoch_hooks=tc.build_epoch_hooks(),
            seed=tc.seed,
            dtype=tc.dtype,
            device=resolved_device,
        )

    def _batch_bounds(self, n: int):
        """Yield (start, stop) slice bounds over n samples, chunked by self.batch_size.

        With batch_size=None (or >= n) this yields a single full-batch slice, so
        all chunked code paths reduce exactly to the original single-pass behavior.
        """
        bs = self.batch_size if (self.batch_size and self.batch_size < n) else n
        for start in range(0, n, bs):
            yield start, min(start + bs, n)

    def _iter_batches(self, n: int):
        """Yield (index, weights) training batches for one epoch.

        Default (no batch_sampler): contiguous slice objects over n samples via
        _batch_bounds, weights=None — byte-identical to the historical
        gradient-accumulation path. With a batch_sampler: whatever index batches
        it yields (optionally (indices, weights) tuples), as index tensors.
        """
        if self.batch_sampler is None:
            for start, stop in self._batch_bounds(n):
                yield slice(start, stop), None
            return
        for batch in self.batch_sampler:
            if isinstance(batch, tuple) and len(batch) == 2:
                idx, weights = batch
            else:
                idx, weights = batch, None
            idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
            if weights is not None and not torch.is_tensor(weights):
                weights = torch.as_tensor(weights, dtype=self.dtype, device=self.device)
            yield idx, weights

    def _model_forward(self, x, mask):
        """Dispatch a forward pass by model family: DynamicNN models receive
        (B, S, F) input plus the validity mask; StaticNN models receive the
        flattened (B, S*F) sample and no mask."""
        if isinstance(self.model, StaticNN):
            return self.model(x.reshape(x.shape[0], -1))
        return self.model(x, mask=mask)

    def _forward_in_batches(self, x, mask):
        """Run the model forward over x in micro-batches, concatenating outputs.

        Numerically exact vs. a single full-batch forward (every model is
        per-sample along the batch dim); only the peak attention/activation
        memory is bounded. Intended for use under torch.no_grad() (evaluation).
        """
        outs, hiddens = [], []
        for start, stop in self._batch_bounds(x.shape[0]):
            mb = mask[start:stop] if mask is not None else None
            ob, hb = self._model_forward(x[start:stop], mb)
            outs.append(ob)
            hiddens.append(hb)
        if len(outs) == 1:
            return outs[0], hiddens[0]
        return torch.cat(outs, dim=0), torch.cat(hiddens, dim=0)

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        epoch = len(self.logs['train_loss'])
        if self.visualizer is not None:
            self.visualizer.begin_epoch(epoch)

        x_clean = torch.nan_to_num(self.x_train, nan=0.0)
        N = x_clean.shape[0]

        if getattr(self.model, 'seq2seq', False):
            if self.batch_sampler is not None or self.step_granularity != 'epoch':
                raise NotImplementedError(
                    "batch_sampler / step_granularity='batch' are not supported "
                    "with seq2seq (per-timestep-supervised) training")
            # Model always emits a per-step (b, T) output when seq2seq=True,
            # regardless of cot_keep_fraction (0 => train_cot_mask keeps only
            # each row's final valid step, matching scalar-target supervision).
            # train_cot_mask is fixed at dataset-construction time (including any
            # sparse-supervision Bernoulli draw), so gradients accumulated over
            # micro-batches equal the full-batch normalized loss gradient exactly.
            denom = self._train_valid_total

            train_loss_val = 0.0
            step_sum = None       # accumulated (T,) sum of step losses over samples
            step_valid_n = None   # accumulated (T,) valid counts
            for start, stop in self._batch_bounds(N):
                out, _ = self._model_forward(x_clean[start:stop], self.train_mask[start:stop])
                out = out.squeeze(-1)   # (b, T)
                # Supervise only where target is finite (sequence valid AND enough elements for GT),
                # further restricted by the fixed sparse-supervision mask.
                valid = self.train_cot_mask[start:stop]
                targets = torch.where(valid.bool(), self.y_train_intermediate[start:stop],
                                      torch.zeros_like(self.y_train_intermediate[start:stop]))
                step_losses = F.mse_loss(out, targets, reduction='none')  # (b, T)
                loss = (step_losses * valid).sum() / denom
                loss.backward()
                train_loss_val += loss.item()
                sv = (step_losses.detach() * valid).sum(0)   # (T,)
                vn = valid.sum(0)                            # (T,)
                step_sum = sv if step_sum is None else step_sum + sv
                step_valid_n = vn if step_valid_n is None else step_valid_n + vn

            step_loss_vec = (step_sum / step_valid_n.clamp(min=1)).cpu().numpy()        # (T,)
            step_loss_vec[step_valid_n.cpu().numpy() == 0] = float('nan')               # unsupervised → nan
            self.logs['step_train_losses'].append(step_loss_vec)
        else:
            # step_granularity='epoch': criterion is a mean-reduction loss
            # (e.g. MSELoss): scaling each micro-batch mean by (chunk_n / N)
            # makes the accumulated gradient identical to the full-batch
            # mean-loss gradient, with one optimizer step per epoch.
            # step_granularity='batch': unscaled per-batch loss, optimizer and
            # scheduler stepped after every batch.
            train_loss_val = 0.0
            for idx, weights in self._iter_batches(N):
                out, _ = self._model_forward(x_clean[idx], self.train_mask[idx])
                out = out.squeeze(-1) if out.dim() > 1 else out
                y = self.y_train[idx]
                if weights is not None:
                    loss = self.criterion(out, y, weights)
                else:
                    loss = self.criterion(out, y)
                if self.step_granularity == 'epoch':
                    loss = loss * (y.shape[0] / N)
                loss.backward()
                loss_val = loss.item()
                train_loss_val += loss_val
                if self.visualizer is not None:
                    self.visualizer.record_batch(epoch, {
                        'indices': idx, 'weights': weights, 'loss': loss_val})
                if self.step_granularity == 'batch':
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()

        self.logs['train_loss'].append(train_loss_val)
        if self.step_granularity == 'epoch':
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
        if self.visualizer is not None:
            self.visualizer.end_epoch(epoch)

    def test_epoch(self, epoch: int = None):
        self.model.eval()
        with torch.no_grad():
            x_clean = torch.nan_to_num(self.x_test, nan=0.0)
            out, hidden = self._forward_in_batches(x_clean, self.test_mask)

            if getattr(self.model, 'seq2seq', False):
                out_seq = out.squeeze(-1)   # (B, T)
                n = getattr(self, '_x_test_random_n', out_seq.shape[0])
                valid = self.test_cot_mask[:n]
                targets = torch.where(valid.bool(), self.y_test_intermediate[:n],
                                      torch.zeros_like(self.y_test_intermediate[:n]))
                step_losses = F.mse_loss(out_seq[:n], targets, reduction='none')  # (n, T)
                test_loss = (step_losses * valid).sum() / valid.sum().clamp(min=1)
                step_valid_n = valid.sum(0)
                step_loss_vec = ((step_losses.detach() * valid).sum(0) /
                                 step_valid_n.clamp(min=1)).cpu().numpy()
                step_loss_vec[step_valid_n.cpu().numpy() == 0] = float('nan')
                self.logs['step_test_losses'].append(step_loss_vec)
                # Log final-valid-step output for existing visualizations
                last_t = (self.test_mask.long().sum(dim=1) - 1).clamp(min=0)
                B = out_seq.shape[0]
                f_test_np = out_seq[torch.arange(B, device=self.device), last_t].cpu().numpy()
                hidden_np = (hidden[torch.arange(B, device=self.device), last_t]
                             if hidden.dim() == 3
                             else hidden).cpu().numpy()
            else:
                out = out.squeeze()
                n = getattr(self, '_x_test_random_n', out.shape[0])
                test_loss = self.criterion(out[:n], self.y_test[:n])
                f_test_np = out.cpu().numpy()
                hidden_np = hidden.cpu().numpy()

                # Intermediate logging (diagnostic, no training change)
                # Supports both padded and non-padded: group test samples by sequence length
                if self.intermediate_logging:
                    n = getattr(self, '_x_test_random_n', out.shape[0])
                    T_max = self.y_test_intermediate.shape[1]
                    step_loss_vec = np.full(T_max, float('nan'))

                    # Get actual sequence length for each sample
                    seq_lens = self.test_mask[:n].long().sum(dim=1)  # (n,)

                    # For each possible sequence length, compute loss for samples with that length
                    for t in range(1, T_max + 1):
                        # Find all samples whose actual sequence length is t
                        idx_t = (seq_lens == t).nonzero(as_tuple=True)[0]
                        if len(idx_t) == 0:
                            continue
                        # Get corresponding ground truth values at time step t-1
                        gt_t = self.y_test_intermediate[:n, t - 1][idx_t]
                        # Filter for finite ground truth
                        valid_mask = gt_t.isfinite()
                        valid = idx_t[valid_mask]
                        if len(valid) == 0:
                            continue
                        # For seq2seq (padded with seq2seq=True): out has shape (n, T)
                        # For non-seq2seq: out has shape (n,) — use final output for all time steps
                        if out.dim() > 1:
                            out_t = out[:n, t - 1][valid]
                        else:
                            out_t = out[:n][valid]
                        step_loss_vec[t - 1] = F.mse_loss(out_t, self.y_test[:n][valid]).item()

                    self.logs['step_test_losses'].append(step_loss_vec)

            self.logs['test_loss'].append(test_loss.item())
            self.logs['f_test'].append(f_test_np)
            self.logs['hidden_states'].append(hidden_np)

            if self.visualizer is not None:
                current_epoch = epoch if epoch is not None else len(self.logs['test_loss']) - 1
                self.visualizer.update(current_epoch)

    def run(self):
        # At most 5 checkpoints, at least 100 epochs apart, always include the last epoch.
        step = max(100, math.ceil((self.epochs - 1) / 4)) if self.epochs > 1 else 1
        log_epochs = set(range(0, self.epochs, step)) | {self.epochs - 1}
        t_start = time.time()

        for epoch in trange(self.epochs, disable=not sys.stdout.isatty(), desc='Training'):
            for hook in self.epoch_hooks:
                hook(self, epoch)
            self.train_epoch()
            self.test_epoch(epoch)
            if not sys.stdout.isatty() and epoch in log_epochs:
                elapsed = time.time() - t_start
                if epoch == self.epochs - 1:
                    time_str = format_duration(elapsed)
                else:
                    total_estd = elapsed / (epoch + 1) * self.epochs
                    time_str = f"{format_duration(elapsed)}/{format_duration(total_estd)}"
                print(
                    f"  epoch {epoch}/{self.epochs - 1}"
                    f"  train={self.logs['train_loss'][-1]:.4f}"
                    f"  test={self.logs['test_loss'][-1]:.4f}"
                    f"  [{time_str}]",
                    flush=True,
                )

        # convert list entries to np arrays
        for key, val in self.logs.items():
            if isinstance(val, list):
                self.logs[key] = np.array(val)

        print(f"\nTraining complete!")
        print(
            f"Final losses: Train Loss: {self.logs['train_loss'][-1]:.6f}, Test Loss: {self.logs['test_loss'][-1]:.6f}")

        # Note: Visualizer finalization is handled by the caller (Experiment or main script)
        # Processor only updates the visualizer during training via visualizer.update()
        # This allows Visualizer to aggregate across multiple trials before finalization

    def save(self, filename='train_out/training_data.h5', output_dir='train_out'):
        to_save = {'metadata': self.metadata, 'logs': self.logs}

        def dict_to_h5(dic, h5_file, path='/'):
            """Recursively saves a dictionary to an HDF5 file."""
            for key, value in dic.items():
                dataset_path = f"{path}{key}"

                if isinstance(value, dict):
                    # Create a Group for nested dictionaries
                    h5_file.create_group(dataset_path)
                    dict_to_h5(value, h5_file, dataset_path + '/')
                elif isinstance(value, str):
                    # Handle strings: encode as bytes
                    h5_file.create_dataset(dataset_path, data=value.encode('utf-8'))
                elif isinstance(value, (int, float, bool)):
                    # Handle scalars
                    h5_file.create_dataset(dataset_path, data=value)
                else:
                    # Handle arrays and other types
                    try:
                        h5_file.create_dataset(dataset_path, data=np.array(value))
                    except (TypeError, ValueError):
                        # If conversion fails, try as bytes string
                        warnings.warn(f"Could not save {key} as array. Saving as string instead.")
                        h5_file.create_dataset(dataset_path, data=str(value).encode('utf-8'))

        os.makedirs(output_dir, exist_ok=True)
        with h5py.File(filename, 'w') as hf:
            dict_to_h5(to_save, hf)

        # Save model weights
        model_path = os.path.join(output_dir, 'model.pt')
        torch.save(self.model.state_dict(), model_path)

    def print_summary(self):
        """Print training summary from logs."""
        preds_shape = np.array(self.logs['f_test']).shape
        hidden_shape = np.array(self.logs['hidden_states']).shape

        print("\n" + "=" * 75)
        print("TRAINING SUMMARY (from logs)")
        print("=" * 75)
        print(f"\n[Configuration]")
        print(f"  Epochs: {self.epochs}")
        print(f"  Device: {self.device}")
        print(f"  Model: {self.model.__class__.__name__}")
        print(f"\n[Data]")
        print(f"  Train samples: {len(self.x_train)}")
        print(f"  Test samples: {len(self.x_test)}")
        print(f"  Input shape: {self.x_train.shape}")
        print(f"\n[Loss Statistics]")
        print(f"  Final train loss: {self.logs['train_loss'][-1]:.6f}")
        print(f"  Final test loss:  {self.logs['test_loss'][-1]:.6f}")
        print(f"  Best train loss:  {np.min(self.logs['train_loss']):.6f} (epoch {np.argmin(self.logs['train_loss'])})")
        print(f"  Best test loss:   {np.min(self.logs['test_loss']):.6f} (epoch {np.argmin(self.logs['test_loss'])})")
        print(f"\n[HDF5 File Structure]")
        print(f"  ├── logs/")
        print(f"  │   ├── x_train: {self.logs['x_train'].shape}")
        print(f"  │   ├── y_train: {self.logs['y_train'].shape}")
        print(f"  │   ├── x_test: {self.logs['x_test'].shape}")
        print(f"  │   ├── y_test: {self.logs['y_test'].shape}")
        print(f"  │   ├── train_loss: {len(self.logs['train_loss'])} entries")
        print(f"  │   ├── test_loss: {len(self.logs['test_loss'])} entries")
        print(f"  │   ├── f_test: {preds_shape} entries")
        print(f"  │   └── hidden_states: {hidden_shape} entries")
        print(f"  └── metadata/")
        print(f"      ├── dataset: {self.metadata['dataset']}")
        print(f"      ├── model: {self.metadata['model']}")
        print(f"      └── train: {self.metadata['train']}")
        print("=" * 75 + "\n")

    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    def _set_environment(self, *, dtype=torch.float32, seed: Optional[int] = None, device=None):
        """
        Set random seeds and device for reproducibility and performance.
        Args:
            dtype: Data type to use for model parameters and computations.
            seed: Optional random seed for reproducibility. If None, a random seed will be generated.
            device: Device to use. Can be None, str, or torch.device.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.seed = seed if seed is not None else 42
        # Use global SeedManager for consistent seed setting
        SeedManager.set_seed(self.seed, device=str(self.device))
        self.rng = np.random.default_rng(self.seed)
        self.dtype = dtype

    def reset(self):
        self._set_environment(seed=self.seed, dtype=self.dtype)
        self.logs = {
            "train_loss": [],
            "test_loss": [],
            "f_test": [],
            "hidden_states": [],
        }
        if getattr(self.model, 'seq2seq', False):
            self.logs["step_train_losses"] = []
            self.logs["step_test_losses"] = []
        elif self.intermediate_logging:
            self.logs["step_test_losses"] = []
