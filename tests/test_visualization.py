"""Visualization pipeline: registry logic + full FFmpeg/matplotlib rendering.

The end-to-end rendering tests are marked ``slow`` (they train a tiny model and
invoke FFmpeg). They assert that artifact files are produced and non-empty.
"""
import os

import numpy as np
import pytest

from brainspace.config import RunConfig, DatasetConfig, ModelConfig, TrainConfig
from brainspace.datasets import cumsum
from brainspace.models import LSTM
from brainspace.train import Processor
from brainspace.visualization import (
    PlotBackend, MatplotlibRenderer, Visualizer, LossHistoryPlot,
    IntermediateLossHistory, SequenceLengthLossPlot, PCA3D,
)
from tests.conftest import requires_cuda


def _nonempty(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _viz_run_config(epochs=5):
    return RunConfig(
        data=DatasetConfig(ground_truth=cumsum(), data_dim=(6, 1), N=32),
        model=ModelConfig(arch=LSTM, hidden_dim=8, num_layers=1, pool=True, seq2seq=False),
        train=TrainConfig(epochs=epochs, seed=42, device="cpu", lr=1e-2),
    )


# ----------------------------------------------------------------------------
# PlotBackend
# ----------------------------------------------------------------------------

def test_plotbackend_returns_matplotlib_renderer():
    PlotBackend.set_backend("matplotlib")
    r = PlotBackend.get_renderer()
    assert isinstance(r, MatplotlibRenderer)
    # Singleton identity is stable across calls.
    assert PlotBackend.get_renderer() is r


# ----------------------------------------------------------------------------
# Visualizer registry
# ----------------------------------------------------------------------------

def test_register_methods_populate_registry():
    viz = Visualizer(name="t")
    viz.register_loss_history()
    viz.register_function_space_convergence()
    assert "loss_history" in viz.visualizations
    assert "function_space" in viz.visualizations


def test_attach_processor_autoregisters_for_dynamicnn(tiny_processor):
    # register_defaults() unconditionally registers DynamicNN visualizations;
    # attach_processor() only sets the processor reference.
    viz = Visualizer(name="t")
    viz.register_defaults()
    viz.attach_processor(tiny_processor)
    assert "intermediate_loss_history" in viz.visualizations
    assert "multi_axis_convergence_1d" in viz.visualizations


def test_sampling_controls_update_frequency(tiny_processor):
    viz = Visualizer(name="t", sampling=3)
    viz.attach_processor(tiny_processor)
    lh = LossHistoryPlot()
    viz.register(lh)
    for epoch in range(9):
        viz.update(epoch)
    # Only epochs 0, 3, 6 are forwarded to the visualization (every 3rd).
    assert lh.epoch_numbers == [0, 3, 6]


class _FakeProcessor:
    """Minimal stand-in exposing just the logs step-loss visualizations read."""

    def __init__(self):
        self.logs = {'step_train_losses': [], 'step_test_losses': []}
        self.train_sequence_lengths = [1, 2, 3]

    def step(self, n_steps=3):
        self.logs['step_train_losses'].append(np.ones(n_steps))
        self.logs['step_test_losses'].append(np.ones(n_steps) * 2)


def test_sequence_length_and_intermediate_history_use_real_epoch_numbers():
    # Regression test for epoch<->sampling mixups: with sampling=5, epoch
    # checkpoints/axes must be real epoch numbers (multiples of 5), never raw
    # frame indices into the collected per-trial arrays.
    sampling = 5
    viz = Visualizer(name="t", sampling=sampling)
    processor = _FakeProcessor()
    viz.attach_processor(processor)

    ilh = IntermediateLossHistory()
    sllp = SequenceLengthLossPlot()
    viz.register(ilh)
    viz.register(sllp)

    n_epochs = 21  # epochs 0, 5, 10, 15, 20 get forwarded
    for epoch in range(n_epochs):
        processor.step()
        viz.update(epoch)
    viz.next_trial()

    expected_epochs = list(range(0, n_epochs, sampling))
    assert ilh.all_epoch_numbers == [expected_epochs]
    assert sllp.all_epoch_numbers == [expected_epochs]

    # Frame count matches how many times update() was actually forwarded.
    assert ilh.epoch_axis(0, len(expected_epochs)).tolist() == expected_epochs
    assert sllp.epoch_axis(0, len(expected_epochs)).tolist() == expected_epochs


class _FakeHiddenStateProcessor:
    """Minimal stand-in exposing the f_test/hidden_states logs PCA3D reads."""

    def __init__(self):
        self.logs = {}

    def step(self):
        self.logs['f_test'] = [np.random.randn(6, 1)]
        self.logs['hidden_states'] = [np.random.randn(6, 4)]


def test_pca3d_procrustes_frame_indices_stay_aligned_with_sampling(tmp_path):
    # Regression test: PCA3D.sampling already gates which epochs reach update()
    # (frame_keys is pre-sampled), so procrustes-mode frame_indices must cover
    # every collected frame, not re-apply sampling a second time - otherwise
    # the PCA-projected frames desync from the (unresampled) epochs/f_test
    # arrays used alongside them during animation.
    sampling = 5
    n_real_epochs = 50
    viz = Visualizer(name="t", sampling=sampling, output_dir=str(tmp_path))
    processor = _FakeHiddenStateProcessor()
    viz.attach_processor(processor)

    pca3d = PCA3D(mode='procrustes')
    viz.register(pca3d)

    for epoch in range(n_real_epochs):
        processor.step()
        viz.update(epoch)
    viz.next_trial()

    try:
        frame_keys = pca3d._h5_trial_frames[0]
        epochs = pca3d.epoch_axis(0, len(frame_keys))
        frame_indices = np.arange(len(frame_keys))

        assert len(frame_keys) == len(range(0, n_real_epochs, sampling))
        assert len(epochs) == len(frame_keys)
        assert len(frame_indices) == len(frame_keys)
    finally:
        if pca3d._h5_file is not None:
            pca3d._h5_file.close()


# ----------------------------------------------------------------------------
# Direct renderer unit check
# ----------------------------------------------------------------------------

@pytest.mark.slow
def test_render_2d_plot_creates_png(out_dir):
    renderer = MatplotlibRenderer()
    path = os.path.join(out_dir, "plot.png")
    x = np.arange(10)
    renderer.render_2d_plot(x, x + 1.0, x + 2.0, "Title", "x", "y", path)
    assert _nonempty(path)


# ----------------------------------------------------------------------------
# End-to-end render (matplotlib frames + FFmpeg encode)
# ----------------------------------------------------------------------------

@pytest.mark.slow
def test_end_to_end_render_produces_artifacts(out_dir):
    viz = Visualizer(name="run", output_dir=out_dir, sampling=1)
    proc = Processor.from_run_config(_viz_run_config(), visualizer=viz)
    viz.register_defaults()
    proc.run()
    viz.finalize()

    # PNG artifacts (matplotlib).
    assert _nonempty(os.path.join(out_dir, "run_loss_history.png"))
    assert _nonempty(os.path.join(out_dir, "run_function_space.png"))
    # MP4 artifacts (FFmpeg-encoded animations).
    assert _nonempty(os.path.join(out_dir, "run_pca_3d_procrustes.mp4"))
    assert _nonempty(os.path.join(out_dir, "run_multi_axis_1d_convergence.mp4"))


@pytest.mark.slow
def test_background_finalize_produces_artifacts(out_dir):
    viz = Visualizer(name="bg", output_dir=out_dir, sampling=1)
    proc = Processor.from_run_config(_viz_run_config(), visualizer=viz)
    viz.register_loss_history()
    proc.run()
    viz.finalize(background=True)
    assert viz.wait_for_background(timeout=120) is True
    assert _nonempty(os.path.join(out_dir, "bg_loss_history.png"))


@requires_cuda
@pytest.mark.slow
def test_end_to_end_render_cuda(out_dir):
    viz = Visualizer(name="gpu", output_dir=out_dir, sampling=1)
    proc = Processor.from_run_config(_viz_run_config(), visualizer=viz)
    viz.register_loss_history()
    proc.run()
    viz.finalize()
    assert _nonempty(os.path.join(out_dir, "gpu_loss_history.png"))
