import numpy as np
import torch
import os
import json
import math
from collections import namedtuple
from itertools import combinations
from scipy.stats import qmc
from typing import Callable, Optional, Union


DatasetResult = namedtuple('DatasetResult', [
    'x_train', 'y_train', 'x_test', 'y_test',
    'train_mask', 'test_mask',
    'y_train_intermediate', 'y_test_intermediate',
    'train_cot_mask', 'test_cot_mask',
    'train_sequence_lengths', 'd',
    'x_test_random_n', 'x_test_axis_start', 'x_test_axis_d', 'x_test_axis_end',
])


class Dataset:
    """
    Dataset container for functional-regression experiments.

    Generates train/test splits given a ground truth callable:
    - StaticNN:  f(x: Tensor[N, d]) -> Tensor[N]
    - DynamicNN: f(x: Tensor[N, T, d]) -> Tensor[N, T]  (prefix outputs)

    All attributes (x_train, y_train, train_mask, feature_dim, etc.) are
    set during initialization.

    Example:
        ds = Dataset(cumsum(), x_range=(-8, 8), data_dim=(10, 1), N=2048, use_padding=True)
        print(ds.x_train.shape, ds.feature_dim)  # (2048, 10, 1), 1
    """
    def __init__(
        self,
        ground_truth: Callable,
        x_range: tuple = (-8, 8),
        data_dim: tuple = (10, 1),
        N: int = 2048,
        use_padding: bool = False,
        min_seq_len: int = 5,
        cot: Union[bool, float] = False,
        seed: Optional[int] = None,
        device=None,
    ):
        """
        Args:
            ground_truth: callable mapping inputs to targets
            x_range: (min, max) for data values
            data_dim: (seq_len, feature_dim) for padded, or any shape for flat
                - Padded: always (seq_len, feature_dim); model input_dim = feature_dim
                - Non-padded: any shape; model input_dim = data_dim[-1] (last dimension)
            N: number of samples
            use_padding: if True, generate variable-length sequences
            min_seq_len: minimum sequence length (for padded data)
            cot: float in [0, 1] for Chain-of-Thought supervision density.
                0: no intermediate supervision (final step only).
                1: full CoT (all intermediate steps supervised).
                0.5: 50% of intermediate steps supervised (sampled randomly,
                     except final step is always kept).
                The subset of steps is fixed once (drawn here), not resampled.
            seed: random seed for reproducible data generation
            device: torch device
        """
        rng = np.random.default_rng(seed)
        result = generate_dataset(
            N=N, data_dim=data_dim, x_range=x_range, ground_truth=ground_truth,
            use_padding=use_padding, min_seq_len=min_seq_len, cot=cot,
            rng=rng, device=device
        )

        for field in result._fields:
            setattr(self, field, getattr(result, field))

        # feature_dim (model input_dim) is always the last dimension
        # For padded data: data_dim[-1] (feature_dim per timestep)
        # For non-padded shaped data: data_dim[-1] (feature_dim per element)
        self.feature_dim = data_dim[-1]
        self.ground_truth = ground_truth
        self.x_range = x_range
        self.data_dim = data_dim
        self.N = N
        self.use_padding = use_padding
        self.min_seq_len = min_seq_len
        self.cot_keep_fraction = (
            float(cot) if isinstance(cot, (float, int)) and not isinstance(cot, bool)
            else (1.0 if cot else 0.0)
        )
        self.cot = bool(self.cot_keep_fraction)

        self.metadata = {
            "x_range": x_range,
            "data_dim": data_dim,
            "N": N,
            "use_padding": use_padding,
            "ground_truth": ground_truth.__class__.__name__,
            "x_test_random_n": self.x_test_random_n,
            "x_test_axis_start": self.x_test_axis_start,
            "x_test_axis_end": self.x_test_axis_end,
            "x_test_axis_d": self.x_test_axis_d,
        }


def _prefix_mask(x: torch.Tensor) -> torch.Tensor:
    """Expand (N, T, feat) into prefix-masked (N, T, T*feat).

    Row ``t`` contains the flattened elements seen up to and including time step
    ``t``; future positions are filled with ``-inf`` (the tropical identity), so
    a max-plus aggregate over the last dimension yields a prefix output.
    """
    N, T, feat = x.shape
    x_flat   = x.reshape(N, T * feat)                                 # (N, T*feat)
    step_end = torch.arange(1, T + 1, device=x.device) * feat         # (T,)
    col_idx  = torch.arange(T * feat, device=x.device)               # (T*feat,)
    invalid  = col_idx.unsqueeze(0) >= step_end.unsqueeze(1)         # (T, T*feat)
    return x_flat.unsqueeze(1).expand(N, T, T * feat).masked_fill(invalid, float('-inf'))


class GroundTruth:
    """Callable ground-truth target function.

    Subclasses implement ``__call__``: a callable mapping inputs to targets.
    2D input (N, feat) maps to (N,) final targets; 3D input (N, T, feat) maps
    to (N, T) prefix targets (set ``has_intermediate = True`` to enable
    per-timestep diagnostics/supervision). Domain packages may add richer
    protocols (e.g. exact symbolic representations) in subclasses.
    """

    has_intermediate = False

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CumulativeSum(GroundTruth):
    """Cumulative sum — the running total of all elements seen so far.

    - 2D input (N, feat)    → (N,)   sum of all elements per sample
    - 3D input (N, T, feat) → (N, T) prefix sums at each time step
      (NaN padding contributes 0)
    """

    has_intermediate = True

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0)
        if x.dim() == 2:
            return x.sum(dim=-1)
        return x.sum(dim=-1).cumsum(dim=-1)  # (N, T)


# Back-compat factory alias, mirroring the lowercase ground-truth names.
cumsum = CumulativeSum


def x2(dim):
    return lambda x: torch.sum(x**2, dim=dim)

def sin():
    return torch.sin


def generate_padded_sequences(N: int, min_len: int, max_len: int, feature_dim: int,
                              x_range: tuple, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate variable-length sequences with padding.

    Returns:
        (padded_data, mask) where:
        - padded_data: (N, max_len, feature_dim) with NaN padding on invalid positions
        - mask: (N, max_len) boolean mask (True=valid, False=padded)
    """
    seq_lens = rng.integers(min_len, max_len + 1, size=N)
    mask = torch.arange(max_len) < torch.from_numpy(seq_lens).unsqueeze(1)
    values = rng.uniform(x_range[0], x_range[1], size=(N, max_len, feature_dim))
    padded_data = torch.from_numpy(values).float()
    padded_data[~mask] = float('nan')
    return padded_data, mask


def pad_to_length(x: torch.Tensor, mask: torch.Tensor, target_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad or truncate data to a specific target length."""
    N, current_length, D = x.shape
    if current_length == target_length:
        return x, mask
    elif current_length < target_length:
        padded = torch.full((N, target_length, D), float('nan'), dtype=x.dtype, device=x.device)
        padded[:, :current_length] = x
        new_mask = torch.zeros((N, target_length), dtype=torch.bool, device=x.device)
        new_mask[:, :current_length] = mask
        return padded, new_mask
    else:
        return x[:, :target_length], mask[:, :target_length]


def create_padded_data(x: torch.Tensor, pad_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad existing data to a fixed sequence length using NaN."""
    N = x.shape[0]
    original_seq_len = x.shape[1] if x.dim() > 1 else 1
    D = x.shape[2] if x.dim() > 2 else (x.shape[1] if x.dim() == 2 else 1)
    padded_x = torch.full((N, pad_length, D), float('nan'), dtype=x.dtype, device=x.device)
    copy_len = min(original_seq_len, pad_length)
    padded_x[:, :copy_len] = x[:, :copy_len]
    mask = torch.zeros((N, pad_length), dtype=torch.bool, device=x.device)
    mask[:, :copy_len] = True
    return padded_x, mask


def generate_axis_data(N: int, dims: tuple, x_range: tuple, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate axis data: N samples along axes in high-dimensional space.

    Returns:
        (x_axis, axis_mask) where:
        - x_axis: (N+1, max_len, feature_dim) with origin appended
        - axis_mask: (N+1, max_len) all True (fully valid)
    """
    lin = torch.linspace(x_range[0], x_range[1], steps=N // math.prod(dims))
    eye = torch.eye(math.prod(dims)).unsqueeze(0)
    x_axis = eye * lin.view(-1, 1, 1)
    x_axis = x_axis.view(-1, *dims)
    origin = torch.zeros(1, *dims)
    axis_mask = torch.ones(x_axis.size(0) + 1, dims[0], dtype=torch.bool)
    result = torch.cat((x_axis, origin), dim=0)
    if device is not None:
        result = result.to(device)
        axis_mask = axis_mask.to(device)
    return result, axis_mask


def _cot_mask(mask: torch.Tensor, y_intermediate: torch.Tensor, keep_fraction: float,
              rng: np.random.Generator, device) -> torch.Tensor:
    """Boolean mask of which (sample, timestep) intermediate targets to supervise.

    Starts from all steps that are both sequence-valid and have a finite
    ground-truth target. If ``keep_fraction < 1``, randomly drops a subset of
    those (fixed once here, not resampled), while always keeping each row's
    own last valid timestep.
    """
    valid = mask & y_intermediate.isfinite()
    if keep_fraction < 1.0:
        keep = torch.from_numpy(rng.random(valid.shape)).to(device) < keep_fraction
        last_idx = (mask.long().sum(dim=1) - 1).clamp(min=0)
        keep[torch.arange(valid.shape[0], device=device), last_idx] = True
        valid = valid & keep
    return valid


def generate_dataset(N: int, data_dim: tuple, x_range: tuple, ground_truth,
                     use_padding: bool = False, min_seq_len: int = 5,
                     cot: Union[bool, float] = False,
                     rng: np.random.Generator = None, device=None) -> DatasetResult:
    """
    Generate complete dataset (train + test) with intermediate supervision targets.

    Args:
        N: number of training samples
        data_dim: (seq_len, feature_dim) or (feature_dim,)
        x_range: (min, max) for data values
        ground_truth: callable that maps data to targets
        use_padding: if True, generate variable-length sequences with masking
        min_seq_len: minimum sequence length (for padded data)
        cot: float in [0, 1] for Chain-of-Thought supervision density (see
            ``Dataset.__init__`` for semantics)
        rng: numpy random generator
        device: torch device for tensors

    Returns:
        DatasetResult with all tensors on the specified device
    """
    if rng is None:
        rng = np.random.default_rng()

    cot_keep_fraction = (
        float(cot) if isinstance(cot, (float, int)) and not isinstance(cot, bool)
        else (1.0 if cot else 0.0)
    )
    if not 0.0 <= cot_keep_fraction <= 1.0:
        raise ValueError(f"cot float value must be in [0, 1], got {cot_keep_fraction}")

    if device is None:
        device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    OOD_FACTOR = 5

    if use_padding:
        max_seq_len = data_dim[0]
        feature_dim = data_dim[1]
        d = feature_dim

        x_train, train_mask = generate_padded_sequences(N, min_seq_len, max_seq_len, feature_dim, x_range, rng)
        x_train = x_train.to(device)
        train_mask = train_mask.to(device)

        max_seq_len_test = min_seq_len + int((max_seq_len - min_seq_len) * OOD_FACTOR)
        x_test_rand, test_mask_rand = generate_padded_sequences(2 * N + 1, min_seq_len, max_seq_len_test, feature_dim, x_range, rng)
        x_axis, axis_mask = generate_axis_data(N, (max_seq_len_test, feature_dim), x_range, device=device)
        x_test = torch.cat((x_test_rand.to(device), x_axis), dim=0) * OOD_FACTOR
        test_mask = torch.cat((test_mask_rand.to(device), axis_mask), dim=0)

        y_all_train = ground_truth(x_train).to(device)
        y_all_test = ground_truth(x_test).to(device)

        train_sequence_lengths = set(int(m.sum()) for m in train_mask)

        N_train = x_train.shape[0]
        N_test = x_test.shape[0]
        last_t_train = (train_mask.long().sum(dim=1) - 1).clamp(min=0)
        last_t_test = (test_mask.long().sum(dim=1) - 1).clamp(min=0)
        y_train = y_all_train[torch.arange(N_train, device=device), last_t_train]
        y_test = y_all_test[torch.arange(N_test, device=device), last_t_test]

        train_cot_mask = _cot_mask(train_mask, y_all_train, cot_keep_fraction, rng, device)
        test_cot_mask = _cot_mask(test_mask, y_all_test, cot_keep_fraction, rng, device)

        # Layout of x_test: 2*N+1 random (OOD) sequences followed by an axis-probe block.
        x_test_random_n = 2 * N + 1
        x_test_axis_d = max_seq_len_test * feature_dim
        x_test_axis_start = 2 * N + 1
        n_per_axis = N // x_test_axis_d
        x_test_axis_end = 2 * N + 1 + n_per_axis * x_test_axis_d

        return DatasetResult(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            train_mask=train_mask, test_mask=test_mask,
            y_train_intermediate=y_all_train, y_test_intermediate=y_all_test,
            train_cot_mask=train_cot_mask, test_cot_mask=test_cot_mask,
            train_sequence_lengths=train_sequence_lengths, d=d,
            x_test_random_n=x_test_random_n, x_test_axis_start=x_test_axis_start,
            x_test_axis_d=x_test_axis_d, x_test_axis_end=x_test_axis_end,
        )
    else:
        d = math.prod(data_dim)  # Flattened dimension for LHS

        x_train = qmc.LatinHypercube(d=d, rng=rng).random(N).reshape(N, *data_dim)
        x_train = x_train * (x_range[1] - x_range[0]) + x_range[0]
        x_train = torch.from_numpy(x_train).float().to(device)

        lin = np.linspace(x_range[0], x_range[1], N // d)
        x_axis = (np.eye(d)[None, ...] * lin[:, None, None]).reshape(-1, d).reshape(-1, *data_dim)
        x_test = np.vstack((x_train.cpu().numpy(), x_axis, np.zeros((1, *data_dim)))) * OOD_FACTOR
        x_test = torch.from_numpy(x_test).float().to(device)

        y_all_train = ground_truth(x_train).to(device)
        y_all_test = ground_truth(x_test).to(device)

        T = x_train.shape[1]
        N_train = x_train.shape[0]
        N_test = x_test.shape[0]
        train_mask = torch.ones(N_train, T, dtype=torch.bool, device=device)
        test_mask = torch.ones(N_test, T, dtype=torch.bool, device=device)
        train_sequence_lengths = {T}

        y_train = y_all_train[:, -1]
        y_test = y_all_test[:, -1]

        train_cot_mask = _cot_mask(train_mask, y_all_train, cot_keep_fraction, rng, device)
        test_cot_mask = _cot_mask(test_mask, y_all_test, cot_keep_fraction, rng, device)

        # Layout of x_test: N training-derived rows, then an axis-probe block
        # spanning the full d * (N // d) rows.
        x_test_random_n = N
        x_test_axis_start = N
        x_test_axis_d = d
        x_test_axis_end = N + d * (N // d)

        return DatasetResult(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            train_mask=train_mask, test_mask=test_mask,
            y_train_intermediate=y_all_train, y_test_intermediate=y_all_test,
            train_cot_mask=train_cot_mask, test_cot_mask=test_cot_mask,
            train_sequence_lengths=train_sequence_lengths, d=d,
            x_test_random_n=x_test_random_n, x_test_axis_start=x_test_axis_start,
            x_test_axis_d=x_test_axis_d, x_test_axis_end=x_test_axis_end,
        )
