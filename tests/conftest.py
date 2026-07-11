"""Shared fixtures for the BrainSpace core test suite.

All fixtures run on CPU with tiny configurations so the suite stays fast and
deterministic. Heavy end-to-end work (full training, FFmpeg rendering) lives in
tests marked ``slow``; CUDA-specific paths are guarded by ``requires_cuda``.
"""
import gc
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from brainspace.config import RunConfig, DatasetConfig, ModelConfig, TrainConfig
from brainspace.datasets import cumsum, Dataset
from brainspace.models import LSTM, GRU
from brainspace.train import Processor, SeedManager


# Skip-marker for CUDA tests: collected everywhere, skipped when no GPU present.
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device not available"
)


@pytest.fixture(autouse=True)
def _reset_seed_and_cleanup(request):
    """Reset all RNGs before every test and cleanup CUDA memory after.

    This is critical for HPC environments where CUDA memory accumulation
    across tests can cause segfaults, even when tests use CPU models.

    Note: CUDA cleanup disabled for CUDA tests as it can cause segfaults
    on HPC systems when called after GPU tests. Non-CUDA tests still get cleanup.
    """
    SeedManager.set_seed(0)
    yield

    # Skip cleanup for CUDA tests to avoid segfaults
    if "cuda" in request.keywords:
        return

    # Aggressive cleanup after each test (prevents accumulation on HPC)
    try:
        # Force Python garbage collection first
        gc.collect()

        if torch.cuda.is_available():
            # Clear all GPU memory
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Also reset peak memory stats to avoid tracking memory across tests
            torch.cuda.reset_peak_memory_stats()
            # Reset accumulated stats
            torch.cuda.reset_accumulated_memory_stats()
    except Exception as e:
        # Cleanup errors shouldn't fail the test suite
        pass  # Silently ignore cleanup errors


@pytest.fixture
def cpu_device():
    return torch.device("cpu")


@pytest.fixture
def tiny_dataset_config():
    """Small non-padded dataset config (N=32, seq_len=6, feat=1)."""
    return DatasetConfig(
        ground_truth=cumsum(),
        x_range=(-8, 8),
        data_dim=(6, 1),
        N=32,
        use_padding=False,
    )


@pytest.fixture
def tiny_padded_config():
    """Small padded (variable-length) dataset config."""
    return DatasetConfig(
        ground_truth=cumsum(),
        x_range=(-8, 8),
        data_dim=(6, 1),
        N=32,
        use_padding=True,
        min_seq_len=3,
    )


@pytest.fixture
def small_lstm():
    """Factory for a tiny LSTM; call with keyword overrides. Always uses CPU for tests."""
    def _make(**kwargs):
        defaults = dict(input_dim=1, hidden_dim=8, output_dim=1,
                        num_layers=1, device="cpu")
        defaults.update(kwargs)
        model = LSTM(**defaults)
        # Explicitly ensure CPU to avoid GPU memory accumulation in HPC environments
        if hasattr(model, 'to'):
            model = model.to("cpu")
        return model
    return _make


@pytest.fixture
def small_gru():
    """Factory for a tiny GRU; call with keyword overrides. Always uses CPU for tests."""
    def _make(**kwargs):
        defaults = dict(input_dim=1, hidden_dim=8, output_dim=1,
                        num_layers=1, device="cpu")
        defaults.update(kwargs)
        model = GRU(**defaults)
        # Explicitly ensure CPU to avoid GPU memory accumulation in HPC environments
        if hasattr(model, 'to'):
            model = model.to("cpu")
        return model
    return _make


def make_tiny_run_config(epochs=3, device="cpu", **model_kwargs):
    """Build a tiny RunConfig usable for fast end-to-end Processor runs."""
    model_defaults = dict(arch=LSTM, hidden_dim=8, num_layers=1, pool=True, seq2seq=False)
    model_defaults.update(model_kwargs)
    return RunConfig(
        data=DatasetConfig(ground_truth=cumsum(), data_dim=(6, 1), N=32),
        model=ModelConfig(**model_defaults),
        train=TrainConfig(epochs=epochs, criterion=nn.MSELoss(),
                          seed=42, device=device, optimizer_cls=optim.AdamW, lr=1e-2),
    )


def make_direct_processor(epochs=3, seed=42, device="cpu", batch_size=None, **model_kwargs):
    """Build a Processor via the direct constructor with deterministic init.

    Seeds the global RNG before building the model so weight initialization is
    reproducible, then constructs the Processor (which generates the dataset from
    its own seeded RNG). Used by tests that need identical data and weights across
    runs while isolating a single behavior (e.g. batch-chunking equivalence).
    """
    SeedManager.set_seed(seed)
    dataset = Dataset(cumsum(), data_dim=(6, 1), N=32, seed=seed, device=device)
    mk = dict(input_dim=dataset.feature_dim, hidden_dim=8, output_dim=1, num_layers=1, device=device)
    mk.update(model_kwargs)
    model = LSTM(**mk)
    return Processor(
        dataset=dataset, model=model,
        optimizer=optim.AdamW(model.parameters(), lr=1e-2),
        epochs=epochs, criterion=nn.MSELoss(), batch_size=batch_size,
        seed=seed, device=device,
    )


@pytest.fixture
def tiny_run_config():
    return make_tiny_run_config(epochs=3)


@pytest.fixture
def tiny_processor(tiny_run_config):
    """Processor built (but not yet run) from the tiny config."""
    return Processor.from_run_config(tiny_run_config)


@pytest.fixture
def trained_tiny_processor(tiny_run_config):
    """Processor that has completed a short training run (for viz tests)."""
    proc = Processor.from_run_config(tiny_run_config)
    proc.run()
    return proc


@pytest.fixture
def out_dir(tmp_path):
    """Per-test output directory (string path) for visualization artifacts."""
    d = tmp_path / "viz_out"
    d.mkdir()
    return str(d)
