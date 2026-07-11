"""Training-pipeline invariants: seeding, reproducibility, chunking, logging."""
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from brainspace.config import RunConfig, DatasetConfig, ModelConfig, TrainConfig
from brainspace.datasets import cumsum, Dataset
from brainspace.models import LSTM
from brainspace.train import Processor, SeedManager
from tests.conftest import make_tiny_run_config, make_direct_processor, requires_cuda


# ----------------------------------------------------------------------------
# SeedManager
# ----------------------------------------------------------------------------

def test_seed_manager_reproducible_draws():
    SeedManager.set_seed(123)
    a_np, a_t = np.random.rand(5), torch.rand(5)
    SeedManager.set_seed(123)
    b_np, b_t = np.random.rand(5), torch.rand(5)
    assert np.array_equal(a_np, b_np)
    assert torch.equal(a_t, b_t)


def test_seed_manager_different_seeds_differ():
    SeedManager.set_seed(1)
    a = torch.rand(5)
    SeedManager.set_seed(2)
    b = torch.rand(5)
    assert not torch.equal(a, b)


# ----------------------------------------------------------------------------
# Reproducibility (the canonical regression invariant)
# ----------------------------------------------------------------------------

@pytest.mark.slow
def test_processor_reproducibility():
    # Direct constructor: same seed must reproduce identical loss curves.
    p1 = make_direct_processor(epochs=3, seed=42)
    p1.run()
    p2 = make_direct_processor(epochs=3, seed=42)
    p2.run()
    np.testing.assert_allclose(p1.logs["train_loss"], p2.logs["train_loss"])
    np.testing.assert_allclose(p1.logs["test_loss"], p2.logs["test_loss"])


@pytest.mark.slow
def test_from_run_config_reproducibility():
    # from_run_config must also be reproducible: seeded dataset + seeded model init.
    cfg = make_tiny_run_config(epochs=3)
    p1 = Processor.from_run_config(cfg)
    p1.run()
    p2 = Processor.from_run_config(cfg)
    p2.run()
    assert np.array_equal(p1.x_train.cpu().numpy(), p2.x_train.cpu().numpy())
    np.testing.assert_allclose(p1.logs["train_loss"], p2.logs["train_loss"])
    np.testing.assert_allclose(p1.logs["test_loss"], p2.logs["test_loss"])


# ----------------------------------------------------------------------------
# Gradient-accumulation equivalence (documented "numerically equivalent")
# ----------------------------------------------------------------------------

@pytest.mark.slow
def test_batch_chunking_matches_full_batch():
    # Isolate the chunking math from data/weight randomness: one processor,
    # identical initial weights (saved/restored), compare the train-epoch loss
    # computed full-batch vs in micro-batches.
    import copy
    proc = make_direct_processor(epochs=1, seed=42)
    init_state = copy.deepcopy(proc.model.state_dict())

    proc.batch_size = None
    proc.train_epoch()
    full = proc.logs["train_loss"][-1]

    proc.model.load_state_dict(init_state)
    proc.logs["train_loss"] = []
    proc.batch_size = 8  # N=32 -> 4 micro-batches
    proc.train_epoch()
    chunked = proc.logs["train_loss"][-1]

    assert full == pytest.approx(chunked, rel=1e-5, abs=1e-5)


def test_batch_bounds_full_slice_when_none(tiny_processor):
    tiny_processor.batch_size = None
    assert list(tiny_processor._batch_bounds(32)) == [(0, 32)]


def test_batch_bounds_covers_all_samples(tiny_processor):
    tiny_processor.batch_size = 10
    bounds = list(tiny_processor._batch_bounds(32))
    assert bounds[0][0] == 0 and bounds[-1][1] == 32
    # Contiguous, non-overlapping cover.
    for (_, stop), (start, _) in zip(bounds, bounds[1:]):
        assert stop == start


# ----------------------------------------------------------------------------
# run() logging
# ----------------------------------------------------------------------------

@pytest.mark.slow
def test_run_populates_logs(tiny_run_config):
    proc = Processor.from_run_config(tiny_run_config)
    proc.run()
    for key in ("train_loss", "test_loss", "f_test", "hidden_states"):
        assert isinstance(proc.logs[key], np.ndarray)
        assert len(proc.logs[key]) == tiny_run_config.train.epochs


@pytest.mark.slow
def test_from_run_config_smoke_run():
    proc = Processor.from_run_config(make_tiny_run_config(epochs=1))
    proc.run()
    assert len(proc.logs["train_loss"]) == 1


# ----------------------------------------------------------------------------
# Intermediate supervision path (padded + seq2seq)
# ----------------------------------------------------------------------------

@pytest.mark.slow
def test_intermediate_supervision_logs():
    cfg = RunConfig(
        data=DatasetConfig(ground_truth=cumsum(), data_dim=(6, 1), N=32,
                           use_padding=True, min_seq_len=3),
        model=ModelConfig(arch=LSTM, hidden_dim=8, num_layers=1,
                          seq2seq=True, pool=False),
        train=TrainConfig(epochs=2, criterion=nn.MSELoss(), seed=42,
                          cot=1, optimizer_cls=optim.AdamW, lr=1e-2),
    )
    proc = Processor.from_run_config(cfg)
    proc.run()
    assert len(proc.logs["step_train_losses"]) == 2
    assert len(proc.logs["step_test_losses"]) == 2


@pytest.mark.slow
def test_sparse_intermediate_supervision_mask_fixed_across_epochs():
    # cot as a float in (0, 1) enables sparse (Bernoulli) supervision.
    # Per design, the supervision set must be drawn once and stay fixed for the
    # whole run rather than resampled every epoch.
    cfg = RunConfig(
        data=DatasetConfig(ground_truth=cumsum(), data_dim=(6, 1), N=32,
                           use_padding=True, min_seq_len=3),
        model=ModelConfig(arch=LSTM, hidden_dim=8, num_layers=1,
                          seq2seq=True, pool=False),
        train=TrainConfig(epochs=1, criterion=nn.MSELoss(), seed=42,
                          cot=0.5, optimizer_cls=optim.AdamW, lr=1e-2),
    )
    proc = Processor.from_run_config(cfg)
    assert proc.cot_keep_fraction == 0.5

    proc.train_epoch()
    mask_after_epoch1 = proc.train_cot_mask.clone()

    proc.train_epoch()
    mask_after_epoch2 = proc.train_cot_mask

    # Same mask object/values reused, not resampled.
    assert torch.equal(mask_after_epoch1, mask_after_epoch2)

    # Each sample's own final valid timestep must always be supervised.
    last_idx = (proc.train_mask.long().sum(dim=1) - 1).clamp(min=0)
    kept_at_last = mask_after_epoch2[torch.arange(proc.train_mask.shape[0]), last_idx]
    finite_at_last = proc.y_train_intermediate[torch.arange(proc.train_mask.shape[0]), last_idx].isfinite()
    assert torch.all(kept_at_last[finite_at_last] > 0)


# ----------------------------------------------------------------------------
# CUDA
# ----------------------------------------------------------------------------

@requires_cuda
@pytest.mark.slow
def test_processor_run_on_cuda():
    proc = Processor.from_run_config(make_tiny_run_config(epochs=2, device="cuda"))
    proc.run()
    assert len(proc.logs["train_loss"]) == 2


# ----------------------------------------------------------------------------
# Epoch hooks
# ----------------------------------------------------------------------------

def test_epoch_hooks_called_once_per_epoch_in_order():
    """epoch_hooks run at the start of every training epoch with (proc, epoch)."""
    calls = []
    SeedManager.set_seed(42)
    dataset = Dataset(cumsum(), data_dim=(6, 1), N=32, seed=42, device='cpu')
    model = LSTM(input_dim=dataset.feature_dim, hidden_dim=8, output_dim=1, num_layers=1)
    proc = Processor(
        dataset=dataset, model=model,
        epochs=4, criterion=nn.MSELoss(),
        optimizer=optim.AdamW(model.parameters(), lr=1e-3),
        epoch_hooks=[lambda p, e: calls.append((p is proc, e))],
        seed=42, device='cpu',
    )
    proc.run()
    assert calls == [(True, 0), (True, 1), (True, 2), (True, 3)]


def test_train_config_build_epoch_hooks_default_empty():
    assert TrainConfig().build_epoch_hooks() == []
