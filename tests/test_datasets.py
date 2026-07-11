"""Invariants for dataset generation utilities."""
import math

import numpy as np
import pytest
import torch

from brainspace.datasets import (
    cumsum, x2, sin, generate_dataset, Dataset,
    pad_to_length, create_padded_data, generate_axis_data,
    CumulativeSum, GroundTruth,
)
from tests.conftest import requires_cuda


# ----------------------------------------------------------------------------
# cumsum ground truth
# ----------------------------------------------------------------------------

def test_cumsum_has_intermediate_flag():
    assert cumsum().has_intermediate is True


def test_cumsum_is_class_with_alias():
    assert cumsum is CumulativeSum
    assert isinstance(cumsum(), CumulativeSum)


def test_cumsum_2d_matches_manual_sum():
    fn = cumsum()
    x = torch.randn(10, 7)
    assert torch.allclose(fn(x), x.sum(dim=-1))


def test_cumsum_3d_prefix_sums_and_nan_padding():
    fn = cumsum()
    x = torch.arange(1, 5, dtype=torch.float32).reshape(1, 4, 1)  # values 1..4
    out = fn(x)  # (1, 4)
    assert out.shape == (1, 4)
    assert torch.allclose(out[0], torch.tensor([1.0, 3.0, 6.0, 10.0]))
    # NaN padding contributes zero, so the running total is unchanged after it
    x_pad = torch.tensor([[[1.0], [2.0], [float('nan')], [3.0]]])
    out_pad = fn(x_pad)
    assert torch.allclose(out_pad[0], torch.tensor([1.0, 3.0, 3.0, 6.0]))


def test_ground_truth_base_requires_call():
    with pytest.raises(NotImplementedError):
        GroundTruth()(torch.randn(3, 2))


# ----------------------------------------------------------------------------
# generate_dataset (non-padded)
# ----------------------------------------------------------------------------

def test_generate_dataset_nonpadded_shapes_and_masks():
    rng = np.random.default_rng(0)
    res = generate_dataset(N=64, data_dim=(5, 1), x_range=(-8, 8),
                           ground_truth=cumsum(), rng=rng)
    assert res.d == math.prod((5, 1))
    assert res.x_train.shape == (64, 5, 1)
    assert res.y_train.shape[0] == res.x_train.shape[0]
    assert res.train_mask.shape == (64, 5)
    assert res.train_mask.all() and res.test_mask.all()
    # y_train is the final-timestep target.
    assert torch.allclose(res.y_train, res.y_train_intermediate[:, -1])
    assert res.train_sequence_lengths == {5}


def test_generate_dataset_nonpadded_reproducible():
    def build():
        return generate_dataset(N=32, data_dim=(4, 1), x_range=(-8, 8),
                                ground_truth=cumsum(),
                                rng=np.random.default_rng(123))
    a, b = build(), build()
    assert torch.equal(a.x_train, b.x_train)
    assert torch.equal(a.y_train, b.y_train)


# ----------------------------------------------------------------------------
# generate_dataset (padded)
# ----------------------------------------------------------------------------

def test_generate_dataset_padded_masks_and_nan():
    rng = np.random.default_rng(0)
    res = generate_dataset(N=32, data_dim=(6, 1), x_range=(-8, 8),
                           ground_truth=cumsum(), use_padding=True,
                           min_seq_len=3, rng=rng)
    # Padded (masked-out) positions must be NaN; valid positions finite.
    invalid = ~res.train_mask
    assert torch.isnan(res.x_train[invalid]).all()
    assert torch.isfinite(res.x_train[res.train_mask]).all()
    # Terminal sequence lengths fall within [min_seq_len, max_seq_len].
    assert all(3 <= L <= 6 for L in res.train_sequence_lengths)


def test_generate_dataset_padded_y_from_last_valid_step():
    rng = np.random.default_rng(1)
    res = generate_dataset(N=16, data_dim=(6, 1), x_range=(-8, 8),
                           ground_truth=cumsum(), use_padding=True,
                           min_seq_len=3, rng=rng)
    last_t = (res.train_mask.long().sum(dim=1) - 1).clamp(min=0)
    idx = torch.arange(res.x_train.shape[0])
    assert torch.allclose(res.y_train, res.y_train_intermediate[idx, last_t],
                          equal_nan=True)


def test_generate_dataset_cot_mask_keeps_last_valid_step_and_respects_finiteness():
    rng = np.random.default_rng(0)
    res = generate_dataset(N=32, data_dim=(6, 1), x_range=(-8, 8),
                           ground_truth=cumsum(), use_padding=True,
                           min_seq_len=3, cot=0.5, rng=rng)

    # Sparse mask never supervises a padded/non-finite step.
    assert not (res.train_cot_mask & ~res.train_mask).any()
    assert not (res.train_cot_mask & ~res.y_train_intermediate.isfinite()).any()

    # Each row's own last valid timestep is always kept when its target is finite.
    last_idx = (res.train_mask.long().sum(dim=1) - 1).clamp(min=0)
    idx = torch.arange(res.train_mask.shape[0])
    finite_at_last = res.y_train_intermediate[idx, last_idx].isfinite()
    assert torch.all(res.train_cot_mask[idx, last_idx][finite_at_last])


# ----------------------------------------------------------------------------
# Dataset wrapper
# ----------------------------------------------------------------------------

def test_dataset_wrapper_attributes():
    ds = Dataset(cumsum(), x_range=(-8, 8), data_dim=(5, 2), N=20, seed=7)
    assert ds.feature_dim == 2          # last dim of data_dim
    assert ds.x_train.shape == (20, 5, 2)
    assert ds.N == 20 and ds.use_padding is False


def test_dataset_wrapper_seed_reproducible():
    a = Dataset(cumsum(), data_dim=(4, 1), N=16, seed=99)
    b = Dataset(cumsum(), data_dim=(4, 1), N=16, seed=99)
    assert torch.equal(a.x_train, b.x_train)


# ----------------------------------------------------------------------------
# pad_to_length / create_padded_data
# ----------------------------------------------------------------------------

def test_pad_to_length_identity():
    x = torch.randn(3, 5, 1)
    mask = torch.ones(3, 5, dtype=torch.bool)
    px, pm = pad_to_length(x, mask, 5)
    assert px.shape == (3, 5, 1) and torch.equal(px, x)


def test_pad_to_length_pads_with_nan():
    x = torch.randn(3, 4, 1)
    mask = torch.ones(3, 4, dtype=torch.bool)
    px, pm = pad_to_length(x, mask, 7)
    assert px.shape == (3, 7, 1)
    assert torch.isnan(px[:, 4:]).all()
    assert pm[:, :4].all() and not pm[:, 4:].any()


def test_pad_to_length_truncates():
    x = torch.randn(3, 8, 1)
    mask = torch.ones(3, 8, dtype=torch.bool)
    px, pm = pad_to_length(x, mask, 5)
    assert px.shape == (3, 5, 1) and pm.shape == (3, 5)


def test_create_padded_data_shapes_and_mask():
    x = torch.randn(4, 3, 2)
    px, mask = create_padded_data(x, pad_length=6)
    assert px.shape == (4, 6, 2) and mask.shape == (4, 6)
    assert mask[:, :3].all() and not mask[:, 3:].any()
    assert torch.isnan(px[:, 3:]).all()


# ----------------------------------------------------------------------------
# generate_axis_data
# ----------------------------------------------------------------------------

def test_generate_axis_data_origin_and_mask():
    x_axis, mask = generate_axis_data(N=30, dims=(3, 1), x_range=(-8, 8))
    # Last row is the appended origin (all zeros).
    assert torch.allclose(x_axis[-1], torch.zeros_like(x_axis[-1]))
    assert mask.all()
    assert mask.shape[0] == x_axis.shape[0]


# ----------------------------------------------------------------------------
# helper ground-truth functions
# ----------------------------------------------------------------------------

def test_x2_and_sin_helpers():
    x = torch.randn(5, 3)
    assert torch.allclose(x2(dim=-1)(x), torch.sum(x ** 2, dim=-1))
    assert torch.allclose(sin()(x), torch.sin(x))


# ----------------------------------------------------------------------------
# CUDA device movement
# ----------------------------------------------------------------------------

@requires_cuda
def test_generate_dataset_on_cuda():
    res = generate_dataset(N=16, data_dim=(4, 1), x_range=(-8, 8),
                           ground_truth=cumsum(),
                           rng=np.random.default_rng(0), device="cuda")
    assert res.x_train.is_cuda and res.y_train.is_cuda
