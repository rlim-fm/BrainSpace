"""
Visualization System for Training Monitoring and Analysis
Visualizer takes processor as parameter and dynamically extracts data needed.
Each visualization implements update() and finalize() methods.

Uses matplotlib for frame generation and FFmpeg for efficient video encoding.
"""
import traceback
import warnings
import math
import threading
from typing import Optional, Dict, Any, List, Tuple
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
import h5py
import os
import pickle
import shutil

# ============================================================================
# Helper Functions
# ============================================================================

# Process-wide lock serializing all matplotlib rendering. pyplot's global figure
# state and matplotlib's mathtext parser (pyparsing) are NOT thread-safe; without
# this, concurrent background finalizations of multiple Visualizers corrupt the
# shared parser state and raise spurious mathtext ParseExceptions.
_RENDER_LOCK = threading.Lock()



# ============================================================================
# Pickleable compute functions for ProcessPoolExecutor
# ============================================================================

def _compute_pca_batch(hidden_states, epoch_indices, n_components):
    """Fit PCA on multiple epochs (pickleable for ProcessPoolExecutor)."""
    results = {}
    for epoch_idx in epoch_indices:
        pca = PCA(n_components=n_components)
        pca.fit(hidden_states[epoch_idx])
        results[epoch_idx] = (pca.components_, pca.explained_variance_)
    return results


def _align_procrustes(hidden_2d_list):
    """Apply Procrustes alignment across frames (pickleable for ProcessPoolExecutor)."""
    if not hidden_2d_list:
        return np.array([])
    aligned = [hidden_2d_list[0]]
    for epoch_idx in range(1, len(hidden_2d_list)):
        A = hidden_2d_list[epoch_idx]
        B = hidden_2d_list[epoch_idx - 1]
        R, _ = orthogonal_procrustes(A, B)
        aligned.append(A @ R)
    return np.array(aligned)


def _compute_pca_projection(f_test, y_test):
    """Compute 3D PCA projection for a trial (pickleable for ProcessPoolExecutor)."""
    n_comps = min(3, f_test.shape[0] - 1, f_test.shape[1])
    if n_comps < 1:
        return None

    pca = PCA(n_components=n_comps)
    pca.fit(f_test)
    f_3d_temp = pca.transform(f_test)
    y_3d_temp = pca.transform(y_test[None, :])[0]

    # Pad to 3D
    f_3d = np.pad(f_3d_temp, ((0, 0), (0, max(0, 3 - f_3d_temp.shape[1]))), mode='constant')[:, :3]
    y_3d = np.pad(y_3d_temp, (0, max(0, 3 - len(y_3d_temp))), mode='constant')[:3]
    return (f_3d, y_3d)


# ============================================================================
# Parallel Frame Computation - Separate CPU work from rendering
# ============================================================================

class FrameComputationQueue:
    """
    Manages parallel frame computation using ThreadPoolExecutor.
    Decouples expensive PCA/frame work from rendering, allowing it to run
    in background during or before visualization finalization.

    Uses threads instead of processes to avoid spawn overhead on HPC systems.
    sklearn PCA and scipy orthogonal_procrustes release the GIL during compute,
    so threads are efficient and avoid the 500MB+/process memory cost of spawning subprocesses.
    """

    def __init__(self, max_workers: Optional[int] = None):
        if max_workers is None:
            cpu_count = multiprocessing.cpu_count()
            max_workers = max(1, cpu_count // 2)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = {}
        self.results_cache = {}

    def submit_pca_fit(self, task_id: str, hidden_states: np.ndarray, epoch_indices: np.ndarray,
                       n_components: int = 2) -> None:
        """Queue PCA fitting for multiple epochs in parallel."""
        future = self.executor.submit(_compute_pca_batch, hidden_states, epoch_indices, n_components)
        self.futures[task_id] = future

    def submit_procrustes_alignment(self, task_id: str, hidden_2d_list: List[np.ndarray]) -> None:
        """Queue Procrustes alignment across frames."""
        future = self.executor.submit(_align_procrustes, hidden_2d_list)
        self.futures[task_id] = future

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Retrieve completed result, blocking if necessary."""
        if task_id in self.results_cache:
            return self.results_cache[task_id]

        if task_id not in self.futures:
            raise KeyError(f"Task {task_id} not found")

        result = self.futures[task_id].result(timeout=timeout)
        self.results_cache[task_id] = result
        return result

    def wait_all(self, timeout: Optional[float] = None) -> bool:
        """Wait for all submitted tasks to complete. Returns True if all completed."""
        try:
            for future in as_completed(self.futures.values(), timeout=timeout):
                pass
            return True
        except TimeoutError:
            return False

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)

    def __del__(self):
        self.shutdown()


# ============================================================================
# Renderer Backend Pattern - Separate Rendering from Data Processing
# ============================================================================

class RendererBackend(ABC):
    """Abstract base class for rendering backends (matplotlib, fastplotlib, etc)."""

    @abstractmethod
    def render_2d_plot(self, x, y_train, y_test, title, xlabel, ylabel, output_file):
        """Render a 2D line plot (loss history, etc)."""
        pass

    @abstractmethod
    def render_1d_animation(self, x_1d, y_1d, f_1d, epochs, axis, x_range,
                           output_file, processor_metadata):
        """Render 1D functional convergence animation."""
        pass

    @abstractmethod
    def render_3d_animation(self, pc1_frames, pc2_frames, f_test, y_test, in_domain,
                           epochs, mode, output_file, processor_metadata):
        """Render 3D PCA animation with domain coloring."""
        pass

    @abstractmethod
    def render_3d_scatter(self, f_3d, y_3d, output_file):
        """Render 3D scatter plot (function space convergence)."""
        pass

    @abstractmethod
    def render_combined_loss_history(self, epoch_nums, all_train, all_test, title, xlabel, ylabel, output_file):
        """Render combined loss history with mean ± std across trials."""
        pass

    @abstractmethod
    def render_multi_1d_animation(self, trials_data, output_file):
        """Render 1D convergence animation grid (one panel per trial)."""
        pass

    @abstractmethod
    def render_multi_3d_animation(self, trials_data, output_file):
        """Render 3D PCA animation grid (one panel per trial)."""
        pass

    @abstractmethod
    def render_multi_3d_scatter(self, trials_data, output_file):
        """Render 3D scatter plot grid (one panel per trial)."""
        pass

    @abstractmethod
    def render_intermediate_loss_history(self, all_step_train, all_step_test,
                                         train_sequence_lengths, output_file,
                                         all_epochs=None, max_train_len=None):
        """Render per-time-step loss history with color gradient (per-trial panels).

        Args:
            all_step_train: List of (epochs, T) arrays for training (one per trial), or None if test-only
            all_step_test: List of (epochs, T) arrays for testing (one per trial)
            train_sequence_lengths: Set of terminal sequence lengths (for line styling)
            all_epochs: List of real epoch-number arrays (one per trial), used as the
                x-axis. Falls back to frame indices (0..N-1) if not provided.
            max_train_len: Maximum training sequence length (for length-OOD visual marker)
            output_file: Output file path
        """
        pass

    @abstractmethod
    def render_multi_axis_1d_animation(self, grid_data, output_file):
        """Render multi-axis 1D convergence animation grid.

        Args:
            grid_data: List of axis groups, each a list of trial tuples
                      grid_data[axis_idx][trial_idx] = (x_1d, y_1d, f_1d, epochs, axis, x_range, metadata)
            output_file: Output file path
        """
        pass

    @abstractmethod
    def render_loss_by_sequence_length(self, all_step_train, all_step_test,
                                        train_sequence_lengths, checkpoint_epochs,
                                        output_file, checkpoint_epoch_labels=None,
                                        max_train_len=None):
        """Render loss-vs-sequence-length curves, one line per epoch checkpoint (per-trial panels).

        Args:
            all_step_train: List of (epochs, T) arrays for training (one per trial), or None if test-only
            all_step_test: List of (epochs, T) arrays for testing (one per trial)
            train_sequence_lengths: Set of terminal training sequence lengths (for length-OOD marker)
            checkpoint_epochs: List of epoch indices (row indices into the (epochs, T) arrays) to plot
            checkpoint_epoch_labels: Real epoch numbers corresponding to checkpoint_epochs,
                used for display labels. Falls back to checkpoint_epochs if not provided.
            output_file: Output file path
            max_train_len: Maximum training sequence length (for length-OOD visual marker)
        """
        pass


class MatplotlibRenderer(RendererBackend):
    """CPU-based rendering using matplotlib."""

    def render_2d_plot(self, x, y_train, y_test, title, xlabel, ylabel, output_file):
        """Render a 2D line plot (loss history)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y_train, linewidth=2, label='Train Loss', alpha=0.8)
        ax.plot(x, y_test, linewidth=2, label='Test Loss', alpha=0.8)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=11)
        fig.tight_layout()

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=150)
        plt.close(fig)

    def render_1d_animation(self, x_1d, y_1d, f_1d, epochs, axis, x_range,
                           output_file, processor_metadata):
        """Render 1D functional convergence animation."""
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(x_1d, y_1d, 'k--', linewidth=3, label='Ground Truth', zorder=2)
        line_anim, = ax.plot([], [], 'b-', linewidth=2, label='Network Prediction', zorder=1)
        epoch_text = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        x_min, x_max = processor_metadata.get('dataset', {}).get('x_range', (x_1d.min(), x_1d.max()))
        ax.axvspan(xmin=x_min, xmax=x_max, color='blue', alpha=0.1, label='Training domain')

        x_margin = (x_1d.max() - x_1d.min()) * 0.05
        y_min, y_max = float(min(y_1d.min(), f_1d.min())), float(max(y_1d.max(), f_1d.max()))
        y_margin = (y_max - y_min) * 0.1

        ax.set_xlim(x_1d.min() - x_margin, x_1d.max() + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_xlabel(f'$x_{axis}$', fontsize=12)
        ax.set_ylabel('Output', fontsize=12)
        ax.set_title(f'Functional Convergence along $x_{axis}$ axis', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=11)

        def init():
            line_anim.set_data([], [])
            epoch_text.set_text('')
            return line_anim, epoch_text

        def update_frame(frame_idx):
            line_anim.set_data(x_1d, f_1d[frame_idx])
            # Display actual epoch number (from sampling), not frame index
            epoch_num = epochs[frame_idx] if frame_idx < len(epochs) else epochs[-1]
            epoch_text.set_text(f'Epoch: {epoch_num}')
            return line_anim, epoch_text

        anim = FuncAnimation(fig, update_frame, frames=len(epochs), init_func=init,
                            blit=True, interval=50)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        anim.save(output_file, writer='ffmpeg', fps=15)
        plt.close(fig)

    def render_3d_animation(self, pc1_frames, pc2_frames, f_test, y_test, in_domain,
                           epochs, mode, output_file, processor_metadata):
        """Render 3D PCA animation with domain coloring."""
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        colors = np.where(in_domain, 'blue', 'red')
        surf_plot = ax.plot_trisurf(pc1_frames[0], pc2_frames[0], y_test, cmap='viridis', alpha=0.3)

        scatter_in = ax.scatter([], [], [], c='blue', label='In-domain', s=30, alpha=0.8)
        scatter_out = ax.scatter([], [], [], c='red', label='Out-of-domain', s=30, alpha=0.8)
        epoch_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        def update_frame(frame_idx):
            nonlocal surf_plot
            pc1 = pc1_frames[frame_idx]
            pc2 = pc2_frames[frame_idx]
            z = f_test[frame_idx]
            if mode == 'procrustes':
                surf_plot.remove()
                surf_plot = ax.plot_trisurf(pc1, pc2, y_test, cmap='viridis', alpha=0.3,
                                            label='Ground Truth Surface')
            scatter_in._offsets3d = (pc1[in_domain], pc2[in_domain], z[in_domain])
            scatter_out._offsets3d = (pc1[~in_domain], pc2[~in_domain], z[~in_domain])
            # Display actual epoch number (from sampling), not frame index
            epoch_num = epochs[frame_idx] if frame_idx < len(epochs) else epochs[-1]
            epoch_text.set_text(f'Epoch: {epoch_num}')
            return scatter_in, scatter_out, epoch_text

        pc1_all = pc1_frames.flatten()
        pc2_all = pc2_frames.flatten()
        ax.set_xlim(pc1_all.min(), pc1_all.max())
        ax.set_ylim(pc2_all.min(), pc2_all.max())
        ax.set_zlim(min(f_test.min(), y_test.min()), max(f_test.max(), y_test.max()))
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('Output')
        ax.set_title(f'Hidden State Convergence in PCA Space ({mode} mode)', fontsize=14)
        ax.legend(fontsize=10, loc='upper right')

        anim = FuncAnimation(fig, update_frame, frames=len(epochs), interval=50)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        anim.save(output_file, writer='ffmpeg', fps=15)
        plt.close(fig)

    def render_3d_scatter(self, f_3d, y_3d, output_file):
        """Render 3D scatter plot (function space convergence)."""
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        epochs_range = np.arange(len(f_3d))
        ax.scatter(f_3d[:, 0], f_3d[:, 1], f_3d[:, 2], c=epochs_range, cmap='viridis',
                  s=30, label='Convergence path')
        ax.scatter(y_3d[0], y_3d[1], y_3d[2], c='red', s=100, marker='*', label='Target')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('Functional Convergence in PCA Space')
        ax.legend()

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        fig.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close(fig)

    def render_combined_loss_history(self, epoch_nums, all_train, all_test, title, xlabel, ylabel, output_file):
        """Render combined loss history with individual trials + mean ± std."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Pad all trials to same length
        max_len = max(len(train) for train in all_train)
        train_padded = np.array([np.pad(t, (0, max_len - len(t)), mode='edge') for t in all_train])
        test_padded = np.array([np.pad(t, (0, max_len - len(t)), mode='edge') for t in all_test])

        x = epoch_nums

        # Individual trial curves (light alpha)
        for train_curve in train_padded:
            ax.plot(x, train_curve, 'b-', alpha=0.2, linewidth=0.5)
        for test_curve in test_padded:
            ax.plot(x, test_curve, 'r-', alpha=0.2, linewidth=0.5)

        # Mean curves
        train_mean = np.mean(train_padded, axis=0)
        test_mean = np.mean(test_padded, axis=0)
        train_std = np.std(train_padded, axis=0)
        test_std = np.std(test_padded, axis=0)

        ax.plot(x, train_mean, 'b-', linewidth=2.5, label='Mean Train Loss')
        ax.plot(x, test_mean, 'r-', linewidth=2.5, label='Mean Test Loss')

        # Shaded std
        ax.fill_between(x, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
        ax.fill_between(x, test_mean - test_std, test_mean + test_std, alpha=0.2, color='red')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=11)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        fig.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close(fig)

    def render_multi_1d_animation(self, trials_data, output_file):
        """Render 1D convergence animation grid (one panel per trial)."""
        n_trials = len(trials_data)
        n_cols = min(n_trials, 3)
        n_rows = (n_trials + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)

        # Set up all axes first
        all_lines = []
        all_texts = []
        for trial_idx, (x_1d, y_1d, f_1d, epochs, axis, x_range, metadata) in enumerate(trials_data):
            row, col = trial_idx // n_cols, trial_idx % n_cols
            ax = axes[row, col]

            ax.plot(x_1d, y_1d, 'k--', linewidth=2, label='Ground Truth', zorder=2)
            line, = ax.plot([], [], 'b-', linewidth=2, label='Prediction', zorder=1)
            epoch_text = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            if x_range:
                ax.axvspan(xmin=x_range[0], xmax=x_range[1], color='blue', alpha=0.1, label='Training domain')

            y_min, y_max = float(min(y_1d.min(), f_1d.min())), float(max(y_1d.max(), f_1d.max()))
            y_margin = (y_max - y_min) * 0.1
            ax.set_xlim(x_1d.min() - 0.2, x_1d.max() + 0.2)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            ax.set_xlabel(f'$x_{axis}$', fontsize=11)
            ax.set_ylabel('Output', fontsize=11)
            ax.set_title(f'Trial {trial_idx}', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(fontsize=9)

            all_lines.append((line, f_1d, epochs))
            all_texts.append(epoch_text)

        # Hide unused subplots
        for trial_idx in range(n_trials, n_rows * n_cols):
            row, col = trial_idx // n_cols, trial_idx % n_cols
            axes[row, col].axis('off')

        def update_frame(frame_idx):
            artists = []
            for idx, (line, f_1d, epochs) in enumerate(all_lines):
                if frame_idx < len(f_1d):
                    line.set_data(trials_data[0][0], f_1d[frame_idx])
                    artists.append(line)
                if frame_idx < len(epochs):
                    all_texts[idx].set_text(f'Epoch: {epochs[frame_idx]}')
                else:
                    all_texts[idx].set_text(f'Epoch: {epochs[-1]}')
                artists.append(all_texts[idx])
            return artists

        from matplotlib.animation import FuncAnimation
        n_frames = max(len(epochs) for _, _, _, epochs, _, _, _ in trials_data)  # max epochs across trials
        anim = FuncAnimation(fig, update_frame, frames=n_frames, interval=50, blit=True)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        anim.save(output_file, writer='ffmpeg', fps=15)
        plt.close(fig)

    def render_multi_3d_animation(self, trials_data, output_file):
        """Render 3D PCA animation grid (one panel per trial)."""
        from mpl_toolkits.mplot3d import Axes3D

        n_trials = len(trials_data)
        n_cols = min(n_trials, 3)
        n_rows = (n_trials + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(6*n_cols, 5*n_rows))

        all_ax = []
        all_scatter_in = []
        all_scatter_out = []
        all_texts = []
        all_surf = []

        all_surf_is_trisurf = []  # track whether each trial's surface is trisurf or scatter

        for trial_idx, (pc1_frames, pc2_frames, f_test, y_test, in_domain, epochs, mode, metadata) in enumerate(trials_data):
            ax = fig.add_subplot(n_rows, n_cols, trial_idx + 1, projection='3d')
            all_ax.append(ax)

            try:
                surf_plot = ax.plot_trisurf(pc1_frames[0], pc2_frames[0], y_test, cmap='viridis', alpha=0.3)
                all_surf_is_trisurf.append(True)
            except Exception:
                surf_plot = ax.scatter(pc1_frames[0], pc2_frames[0], y_test,
                                       c=y_test, cmap='viridis', alpha=0.3, s=5)
                all_surf_is_trisurf.append(False)
            scatter_in = ax.scatter([], [], [], c='blue', label='In-domain', s=20, alpha=0.8)
            scatter_out = ax.scatter([], [], [], c='red', label='Out-of-domain', s=20, alpha=0.8)
            epoch_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=9,
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            pc1_all = pc1_frames.flatten()
            pc2_all = pc2_frames.flatten()
            ax.set_xlim(pc1_all.min(), pc1_all.max())
            ax.set_ylim(pc2_all.min(), pc2_all.max())
            ax.set_zlim(min(f_test.min(), y_test.min()), max(f_test.max(), y_test.max()))
            ax.set_xlabel('PC1', fontsize=9)
            ax.set_ylabel('PC2', fontsize=9)
            ax.set_zlabel('Output', fontsize=9)
            ax.set_title(f'Trial {trial_idx} ({mode})', fontsize=10)
            ax.legend(fontsize=8, loc='upper right')

            all_scatter_in.append(scatter_in)
            all_scatter_out.append(scatter_out)
            all_texts.append(epoch_text)
            all_surf.append((surf_plot, trials_data[trial_idx]))

        def update_frame(frame_idx):
            artists = []
            for trial_idx, (pc1_frames, pc2_frames, f_test, y_test, in_domain, epochs, mode, metadata) in enumerate(trials_data):
                if frame_idx >= len(pc1_frames):
                    continue  # Skip if this trial has fewer frames

                ax = all_ax[trial_idx]
                pc1 = pc1_frames[frame_idx]
                pc2 = pc2_frames[frame_idx]
                z = f_test[frame_idx]

                if mode == 'procrustes':
                    all_surf[trial_idx][0].remove()
                    if all_surf_is_trisurf[trial_idx]:
                        try:
                            surf_plot = ax.plot_trisurf(pc1, pc2, y_test, cmap='viridis', alpha=0.3)
                        except Exception:
                            surf_plot = ax.scatter(pc1, pc2, y_test, c=y_test, cmap='viridis', alpha=0.3, s=5)
                            all_surf_is_trisurf[trial_idx] = False
                    else:
                        surf_plot = ax.scatter(pc1, pc2, y_test, c=y_test, cmap='viridis', alpha=0.3, s=5)
                    all_surf[trial_idx] = (surf_plot, all_surf[trial_idx][1])

                all_scatter_in[trial_idx]._offsets3d = (pc1[in_domain], pc2[in_domain], z[in_domain])
                all_scatter_out[trial_idx]._offsets3d = (pc1[~in_domain], pc2[~in_domain], z[~in_domain])
                all_texts[trial_idx].set_text(f'Epoch: {epochs[min(frame_idx, len(epochs)-1)]}')

                artists.extend([all_scatter_in[trial_idx], all_scatter_out[trial_idx], all_texts[trial_idx]])

            return artists

        from matplotlib.animation import FuncAnimation
        n_frames = max(len(trial_data[0]) for trial_data in trials_data)  # max frames across all trials
        anim = FuncAnimation(fig, update_frame, frames=n_frames, interval=50)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        anim.save(output_file, writer='ffmpeg', fps=15)
        plt.close(fig)

    def render_multi_3d_scatter(self, trials_data, output_file):
        """Render 3D scatter plot grid (one panel per trial)."""
        from mpl_toolkits.mplot3d import Axes3D

        n_trials = len(trials_data)
        n_cols = min(n_trials, 3)
        n_rows = (n_trials + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(6*n_cols, 5*n_rows))

        for trial_idx, (f_3d, y_3d) in enumerate(trials_data):
            ax = fig.add_subplot(n_rows, n_cols, trial_idx + 1, projection='3d')

            epochs_range = np.arange(len(f_3d))
            ax.scatter(f_3d[:, 0], f_3d[:, 1], f_3d[:, 2], c=epochs_range, cmap='viridis',
                      s=20, label='Convergence path')
            ax.scatter(y_3d[0], y_3d[1], y_3d[2], c='red', s=60, marker='*', label='Target')

            ax.set_xlabel('PC1', fontsize=9)
            ax.set_ylabel('PC2', fontsize=9)
            ax.set_zlabel('PC3', fontsize=9)
            ax.set_title(f'Trial {trial_idx}', fontsize=10)
            ax.legend(fontsize=8)

        # Hide unused subplots
        for trial_idx in range(n_trials, n_rows * n_cols):
            ax = fig.add_subplot(n_rows, n_cols, trial_idx + 1)
            ax.axis('off')

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        fig.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close(fig)


    def render_intermediate_loss_history(self, all_step_train, all_step_test,
                                         train_sequence_lengths, output_file,
                                         all_epochs=None, max_train_len=None):
        """Render per-time-step loss curves with per-trial panels.

        Blue lines = training (dark→light by step).
        Red lines = testing (dark→light by step).

        Line styles for test:
        - Solid ('-'): t+1 in train_sequence_lengths (terminal training length)
        - Dashed ('--'): t+1 ≤ max(train_sequence_lengths) but not terminal (intermediate)
        - Dotted (':'): t+1 > max(train_sequence_lengths) (length OOD)
        """
        n_trials = len(all_step_test)
        n_cols = min(n_trials, 3)
        n_rows = (n_trials + n_cols - 1) // n_cols

        has_train = all_step_train is not None and len(all_step_train) > 0

        # Compute T (max steps across all trials and both train/test)
        T_test = max(arr.shape[1] for arr in all_step_test) if all_step_test else 0
        T_train = max(arr.shape[1] for arr in all_step_train) if has_train else 0
        T = max(T_test, T_train)

        # Colors: Blues for train, Reds for test
        train_colors = plt.cm.Blues(np.linspace(0.85, 0.35, T))  # dark → light
        test_colors = plt.cm.Reds(np.linspace(0.85, 0.35, T))

        # Compute max training length for length-OOD marker
        if train_sequence_lengths:
            max_train_len = max(train_sequence_lengths)

        fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes_grid = np.array([[axes_grid]])
        elif n_rows == 1 or n_cols == 1:
            axes_grid = axes_grid.reshape(n_rows, n_cols)

        for trial_idx in range(n_trials):
            row, col = trial_idx // n_cols, trial_idx % n_cols
            ax = axes_grid[row, col]

            step_test = all_step_test[trial_idx]  # (epochs, T)
            step_train = all_step_train[trial_idx] if has_train else None  # (epochs, T)

            epochs = (all_epochs[trial_idx] if all_epochs is not None
                      else np.arange(step_test.shape[0]))

            # Plot train curves (solid lines)
            if step_train is not None:
                for t in range(step_train.shape[1]):
                    ax.plot(epochs, step_train[:, t], color=train_colors[t], linestyle='-',
                           alpha=0.8, linewidth=1.5, label=f'Train $x_{{{t}}}$' if t == 0 else '')

            # Plot test curves with per-step line styles (length OOD distinction)
            for t in range(step_test.shape[1]):
                # Determine line style based on length OOD
                if train_sequence_lengths is not None:
                    if (t + 1) in train_sequence_lengths:
                        ls = '-'  # Terminal training length
                    elif max_train_len is not None and (t + 1) <= max_train_len:
                        ls = '--'  # Intermediate
                    else:
                        ls = ':'  # Length OOD
                else:
                    ls = '-'  # Default to solid if no metadata

                ax.plot(epochs, step_test[:, t], color=test_colors[t], linestyle=ls,
                       alpha=0.8, linewidth=1.5, label=f'Test $x_{{{t}}}$' if t == 0 else '')

            # Add vertical line at length-OOD boundary if available
            if max_train_len is not None and max_train_len < T:
                ax.axvline(x=max_train_len - 0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

            ax.set_yscale('log')
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.set_title(f'Trial {trial_idx}', fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.4)

            # Compact legend
            if trial_idx == 0:
                from matplotlib.lines import Line2D
                handles = [
                    Line2D([0], [0], color=train_colors[0], ls='-', linewidth=2, label='Train (early)'),
                    Line2D([0], [0], color=train_colors[-1], ls='-', linewidth=2, label='Train (late)'),
                    Line2D([0], [0], color=test_colors[0], ls='-', linewidth=2, label='Test in-len'),
                    Line2D([0], [0], color=test_colors[-1], ls=':', linewidth=2, label='Test len-OOD'),
                ]
                ax.legend(handles=handles, fontsize=8, loc='upper right', ncol=2)

        # Hide unused subplots
        for trial_idx in range(n_trials, n_rows * n_cols):
            row, col = trial_idx // n_cols, trial_idx % n_cols
            axes_grid[row, col].axis('off')

        fig.tight_layout()
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def render_loss_by_sequence_length(self, all_step_train, all_step_test,
                                        train_sequence_lengths, checkpoint_epochs,
                                        output_file, checkpoint_epoch_labels=None,
                                        max_train_len=None):
        """Render loss-vs-sequence-length curves with per-trial panels.

        One line per selected epoch checkpoint (dark→light by epoch).
        Test lines solid, train lines dashed. Vertical marker at the
        length-OOD boundary (max_train_len) when applicable.
        """
        if checkpoint_epoch_labels is None:
            checkpoint_epoch_labels = checkpoint_epochs
        n_trials = len(all_step_test)
        n_cols = min(n_trials, 3)
        n_rows = (n_trials + n_cols - 1) // n_cols

        has_train = all_step_train is not None and len(all_step_train) > 0

        if train_sequence_lengths:
            max_train_len = max(train_sequence_lengths)

        n_checkpoints = len(checkpoint_epochs)
        checkpoint_colors = plt.cm.viridis(np.linspace(0.85, 0.15, n_checkpoints))

        fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes_grid = np.array([[axes_grid]])
        elif n_rows == 1 or n_cols == 1:
            axes_grid = axes_grid.reshape(n_rows, n_cols)

        for trial_idx in range(n_trials):
            row, col = trial_idx // n_cols, trial_idx % n_cols
            ax = axes_grid[row, col]

            step_test = all_step_test[trial_idx]  # (epochs, T)
            step_train = all_step_train[trial_idx] if has_train else None  # (epochs, T)

            T = step_test.shape[1]
            seq_lengths = np.arange(1, T + 1)
            train_seq_lengths_arr = (np.arange(1, step_train.shape[1] + 1)
                                      if step_train is not None else None)

            for ci, epoch_idx in enumerate(checkpoint_epochs):
                if epoch_idx >= step_test.shape[0]:
                    continue
                color = checkpoint_colors[ci]
                if step_train is not None and epoch_idx < step_train.shape[0]:
                    ax.plot(train_seq_lengths_arr, step_train[epoch_idx], color=color, linestyle='--',
                           alpha=0.8, linewidth=1.5)
                ax.plot(seq_lengths, step_test[epoch_idx], color=color, linestyle='-',
                       alpha=0.9, linewidth=1.8, label=f'Epoch {checkpoint_epoch_labels[ci]}')

            if max_train_len is not None and max_train_len < T:
                ax.axvline(x=max_train_len + 0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

            ax.set_yscale('log')
            ax.set_xlabel('Sequence length', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.set_title(f'Trial {trial_idx}', fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.4)

            if trial_idx == 0:
                ax.legend(fontsize=8, loc='upper left', ncol=1)

        for trial_idx in range(n_trials, n_rows * n_cols):
            row, col = trial_idx // n_cols, trial_idx % n_cols
            axes_grid[row, col].axis('off')

        fig.tight_layout()
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def render_multi_axis_1d_animation(self, grid_data, output_file):
        """Render multi-axis 1D convergence animation grid (axes × trials).

        grid_data[axis_idx][trial_idx] = (x_1d, y_1d, f_1d, epochs, axis, x_range, metadata)
        - n_rows = len(grid_data) (number of axes)
        - n_cols = len(grid_data[0]) (number of trials)
        - Layout: 1 trial → vertical stack; N trials → grid
        """
        n_axes = len(grid_data)
        n_trials = len(grid_data[0]) if grid_data else 0

        if n_axes == 0 or n_trials == 0:
            print("⊘ Empty grid_data for multi-axis animation")
            return

        n_rows = n_axes
        n_cols = n_trials
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))

        # Handle single row/col case (plt.subplots returns 1D array or scalar)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)

        # Set up all axes first
        all_lines = []
        all_texts = []
        for axis_idx, axis_group in enumerate(grid_data):
            for trial_idx, (x_1d, y_1d, f_1d, epochs, axis, x_range, metadata) in enumerate(axis_group):
                ax = axes[axis_idx, trial_idx] if n_rows > 1 and n_cols > 1 else \
                     axes[axis_idx, 0] if n_cols == 1 else axes[0, trial_idx]

                ax.plot(x_1d, y_1d, 'k--', linewidth=2, label='Ground Truth', zorder=2)
                line, = ax.plot([], [], 'b-', linewidth=2, label='Prediction', zorder=1)
                epoch_text = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                if x_range:
                    ax.axvspan(xmin=x_range[0], xmax=x_range[1], color='blue', alpha=0.1, label='Training domain')

                y_min, y_max = float(min(y_1d.min(), f_1d.min())), float(max(y_1d.max(), f_1d.max()))
                y_margin = (y_max - y_min) * 0.1
                ax.set_xlim(x_1d.min() - 0.2, x_1d.max() + 0.2)
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
                ax.set_xlabel(f'$x_{{{axis}}}$', fontsize=11)
                ax.set_ylabel('Output', fontsize=11)
                ax.set_title(f'Axis {axis}, Trial {trial_idx}', fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.4)
                ax.legend(fontsize=9)

                all_lines.append((line, f_1d, epochs, x_1d))
                all_texts.append(epoch_text)

        def update_frame(frame_idx):
            artists = []
            for idx, (line, f_1d, epochs, x_1d) in enumerate(all_lines):
                if frame_idx < len(f_1d):
                    line.set_data(x_1d, f_1d[frame_idx])
                    artists.append(line)
                if frame_idx < len(epochs):
                    all_texts[idx].set_text(f'Epoch: {epochs[frame_idx]}')
                else:
                    all_texts[idx].set_text(f'Epoch: {epochs[-1]}')
                artists.append(all_texts[idx])
            return artists

        from matplotlib.animation import FuncAnimation
        n_frames = max(len(grid_data[0][0][3]) for grid_data_axis in grid_data for _ in grid_data_axis)
        anim = FuncAnimation(fig, update_frame, frames=n_frames, interval=50, blit=True)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        anim.save(output_file, writer='ffmpeg', fps=15)
        plt.close(fig)


# ============================================================================
# Plotting Backend
# ============================================================================

class PlotBackend:
    """Matplotlib-based plotting backend for stable, reliable visualization."""
    _renderer = None

    @classmethod
    def set_backend(cls, backend: str = 'matplotlib', use_gpu: bool = False, offscreen: bool = False) -> None:
        """
        Set the plotting backend. Now only supports matplotlib for stability.

        Args:
            backend: Ignored (matplotlib only)
            use_gpu: Ignored (deprecated, no longer supported)
            offscreen: Ignored (deprecated, no longer supported)
        """
        if backend != 'matplotlib':
            warnings.warn(
                f"Backend '{backend}' no longer supported. Using matplotlib.\n"
                "For fast video creation on HPC, matplotlib is sufficient - FFmpeg will be used for encoding."
            )
        if use_gpu or offscreen:
            warnings.warn(
                "GPU rendering (fastplotlib) has been removed for stability.\n"
                "All rendering now uses matplotlib with FFmpeg for efficient video encoding."
            )
        cls._renderer = MatplotlibRenderer()

    @classmethod
    def get_backend(cls) -> str:
        return 'matplotlib'

    @classmethod
    def get_renderer(cls) -> RendererBackend:
        """Get the matplotlib renderer instance."""
        if cls._renderer is None:
            cls._renderer = MatplotlibRenderer()
        return cls._renderer


# ============================================================================
# Base Visualization Class - Users inherit from this

# ============================================================================

class Visualization(ABC):
    """Base class for all visualizations. Users implement update() and finalize()."""

    def __init__(self, name: str, sampling: int = 1, device: str = 'cpu'):
        self.name = name
        self.sampling = sampling
        self.device = device
        self.frames = []
        # Centralized epoch bookkeeping (do not duplicate in subclasses).
        # Populated exclusively by Visualizer.update()/next_trial() via
        # record_epoch()/commit_epoch_trial(), so it always matches exactly
        # which epochs a subclass's update() was actually called with.
        self.epoch_numbers = []      # actual epoch numbers for current trial
        self.all_epoch_numbers = []  # per-trial lists of actual epoch numbers

    def record_epoch(self, epoch: int):
        """Called by Visualizer once per (sampled) epoch, before update()."""
        self.epoch_numbers.append(epoch)

    # ------------------------------------------------------------------
    # Batch-level hooks (optional). Called by Processor.train_epoch via
    # Visualizer for every training epoch/batch, regardless of sampling.
    # Default no-ops; batch-granular visualizations (e.g. batch-composition
    # or sampler diagnostics) override what they need.
    # ------------------------------------------------------------------
    def begin_epoch(self, processor, epoch: int):
        pass

    def record_batch(self, processor, epoch: int, batch: dict):
        """batch: {'indices': slice|LongTensor, 'weights': Tensor|None,
        'loss': float} for one training batch."""
        pass

    def end_epoch(self, processor, epoch: int):
        pass

    def commit_epoch_trial(self):
        """Called by Visualizer at trial boundaries, before next_trial()."""
        if self.epoch_numbers:
            self.all_epoch_numbers.append(self.epoch_numbers[:])
        self.epoch_numbers = []

    def epoch_axis(self, trial_idx: int, n_frames: int) -> np.ndarray:
        """Real epoch numbers for a trial's collected frames.

        Falls back to a sampling-based reconstruction (with a warning) only
        if epoch tracking is missing or desynced from the frame count.
        """
        if trial_idx < len(self.all_epoch_numbers):
            epochs = self.all_epoch_numbers[trial_idx]
            if len(epochs) == n_frames:
                return np.array(epochs)
        warnings.warn(
            f"{self.name}: epoch tracking missing/desynced for trial {trial_idx} "
            f"({n_frames} frames) - falling back to sampling-based reconstruction."
        )
        return np.arange(0, n_frames * self.sampling, self.sampling)

    @abstractmethod
    def update(self, processor, epoch: int):
        """
        Called each epoch to append frame data.
        Users extract whatever they need from processor.

        Args:
            processor: Processor instance with access to logs, metadata, model
            epoch: current epoch number
        """
        pass

    def next_trial(self):
        """
        Called by Visualizer after each trial to mark trial boundary.
        Override in subclasses to accumulate per-trial data.
        """
        pass

    # ------------------------------------------------------------------
    # Cross-run state persistence (used to merge prior trials on extend).
    # Subclasses list their committed per-trial attributes in _STATE_ATTRS;
    # the base save/load pickles those plus the shared epoch bookkeeping.
    # PCA3D overrides these to also persist its streamed hidden-state HDF5.
    # ------------------------------------------------------------------
    _STATE_ATTRS: Tuple[str, ...] = ()

    def _state_dict(self) -> dict:
        d = {'all_epoch_numbers': [list(e) for e in self.all_epoch_numbers]}
        for a in self._STATE_ATTRS:
            d[a] = getattr(self, a)
        return d

    def _apply_state(self, d: dict):
        self.all_epoch_numbers = [list(e) for e in d.get('all_epoch_numbers', [])]
        for a in self._STATE_ATTRS:
            if a in d:
                setattr(self, a, d[a])

    def save_state(self, state_dir: str):
        """Persist committed per-trial data to ``state_dir`` for later merge."""
        os.makedirs(state_dir, exist_ok=True)
        with open(os.path.join(state_dir, f'{self.name}.state.pkl'), 'wb') as f:
            pickle.dump(self._state_dict(), f)

    def load_state(self, state_dir: str) -> bool:
        """Restore committed per-trial data saved by ``save_state``.

        Returns True if a saved state was found and applied.
        """
        path = os.path.join(state_dir, f'{self.name}.state.pkl')
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            self._apply_state(pickle.load(f))
        return True

    def cleanup(self):
        """Release any persistent resources (open files/temp data).

        No-op by default; overridden by visualizations that hold open handles
        (e.g. PCA3D's streamed hidden-state HDF5). Called by the Visualizer once
        all finalization is complete.
        """
        pass

    @abstractmethod
    def finalize(self, output_dir: str, prefix: str):
        """
        Called after training to create output files (animations, plots, etc).

        Args:
            output_dir: where to save output
            prefix: filename prefix
        """
        pass


class LossHistoryPlot(Visualization):
    """Plot training and test loss across all trials with mean and std."""

    _STATE_ATTRS = ('all_train', 'all_test')

    def __init__(self, device='cpu'):
        super().__init__('loss_history', device=device)
        self.train_losses = []  # current trial
        self.test_losses = []   # current trial
        self.all_train = []     # list of per-trial loss arrays
        self.all_test = []

    def update(self, processor, epoch: int):
        """Extract loss values from processor."""
        if processor.logs['train_loss']:
            self.train_losses.append(processor.logs['train_loss'][-1])
        if processor.logs['test_loss']:
            self.test_losses.append(processor.logs['test_loss'][-1])

    def next_trial(self):
        """Save current trial data and reset for next trial."""
        if self.train_losses:
            self.all_train.append(np.array(self.train_losses))
            self.all_test.append(np.array(self.test_losses))
        self.train_losses = []
        self.test_losses = []

    def finalize(self, output_dir: str, prefix: str):
        """Create combined loss plot with individual trials + mean ± std."""
        # Commit any un-finalized trial
        self.commit_epoch_trial()
        if self.train_losses:
            self.all_train.append(np.array(self.train_losses))
            self.all_test.append(np.array(self.test_losses))

        if not self.all_train or not self.all_test:
            print(f"⊘ Skipping {self.name}: no loss data")
            return

        # Use actual epoch numbers from the first trial
        epoch_nums = self.epoch_axis(0, len(self.all_train[0]))

        output_file = f'{output_dir}/{prefix}_loss_history.png'
        renderer = PlotBackend.get_renderer()
        renderer.render_combined_loss_history(
            epoch_nums, self.all_train, self.all_test,
            title='Training Loss History',
            xlabel='Epoch',
            ylabel='MSE Loss',
            output_file=output_file
        )
        print(f"✓ Loss history plot saved to '{output_file}'")


class IntermediateLossHistory(Visualization):
    """Per-time-step loss history with dark→light color gradient across time steps."""

    _STATE_ATTRS = ('all_train', 'all_test', 'train_seq_lengths')

    def __init__(self, device='cpu'):
        super().__init__('intermediate_loss_history', device=device)
        self.step_train: List[np.ndarray] = []   # (T,) per epoch, current trial
        self.step_test:  List[np.ndarray] = []
        self.all_train:  List[np.ndarray] = []   # (epochs, T) per trial
        self.all_test:   List[np.ndarray] = []
        self.train_seq_lengths = None

    def update(self, processor, epoch: int):
        logs = processor.logs
        if logs.get('step_train_losses'):
            self.step_train.append(logs['step_train_losses'][-1])
        if logs.get('step_test_losses'):
            self.step_test.append(logs['step_test_losses'][-1])
        if self.train_seq_lengths is None:
            self.train_seq_lengths = getattr(processor, 'train_sequence_lengths', None)

    def next_trial(self):
        if self.step_train or self.step_test:
            if self.step_train:
                self.all_train.append(np.array(self.step_train))
            if self.step_test:
                self.all_test.append(np.array(self.step_test))
        self.step_train, self.step_test = [], []

    def finalize(self, output_dir: str, prefix: str):
        self.commit_epoch_trial()
        if self.step_train or self.step_test:
            self.next_trial()
        if not self.all_test:
            print(f"⊘ Skipping {self.name}: no step loss data")
            return
        output_file = f'{output_dir}/{prefix}_intermediate_loss_history.png'
        renderer = PlotBackend.get_renderer()
        # Compute max training length for visual marker
        max_train_len = max(self.train_seq_lengths) if self.train_seq_lengths else None
        all_epochs = [self.epoch_axis(i, arr.shape[0]) for i, arr in enumerate(self.all_test)]
        renderer.render_intermediate_loss_history(
            all_step_train=self.all_train if self.all_train else None,
            all_step_test=self.all_test,
            all_epochs=all_epochs,
            train_sequence_lengths=self.train_seq_lengths,
            max_train_len=max_train_len,
            output_file=output_file,
        )
        print(f"✓ Intermediate loss history saved to '{output_file}'")


class SequenceLengthLossPlot(Visualization):
    """Loss-vs-sequence-length curves, one line per epoch checkpoint (per-trial panels).

    Inverse of IntermediateLossHistory: x-axis is sequence length, lines are a
    handful of epoch checkpoints, instead of x-axis epoch with lines per length.
    """

    N_CHECKPOINTS = 5
    _STATE_ATTRS = ('all_train', 'all_test', 'train_seq_lengths')

    def __init__(self, device='cpu'):
        super().__init__('sequence_length_loss', device=device)
        self.step_train: List[np.ndarray] = []   # (T,) per epoch, current trial
        self.step_test:  List[np.ndarray] = []
        self.all_train:  List[np.ndarray] = []   # (epochs, T) per trial
        self.all_test:   List[np.ndarray] = []
        self.train_seq_lengths = None

    def update(self, processor, epoch: int):
        logs = processor.logs
        if logs.get('step_train_losses'):
            self.step_train.append(logs['step_train_losses'][-1])
        if logs.get('step_test_losses'):
            self.step_test.append(logs['step_test_losses'][-1])
        if self.train_seq_lengths is None:
            self.train_seq_lengths = getattr(processor, 'train_sequence_lengths', None)

    def next_trial(self):
        if self.step_train or self.step_test:
            if self.step_train:
                self.all_train.append(np.array(self.step_train))
            if self.step_test:
                self.all_test.append(np.array(self.step_test))
        self.step_train, self.step_test = [], []

    def finalize(self, output_dir: str, prefix: str):
        self.commit_epoch_trial()
        if self.step_train or self.step_test:
            self.next_trial()
        if not self.all_test:
            print(f"⊘ Skipping {self.name}: no step loss data")
            return
        output_file = f'{output_dir}/{prefix}_sequence_length_loss.png'
        renderer = PlotBackend.get_renderer()
        max_train_len = max(self.train_seq_lengths) if self.train_seq_lengths else None
        n_epochs = max(arr.shape[0] for arr in self.all_test)
        n_checkpoints = min(self.N_CHECKPOINTS, n_epochs)
        # Row indices into the (epochs, T) arrays - used to slice self.all_test/all_train
        checkpoint_epochs = sorted(set(
            np.linspace(0, n_epochs - 1, n_checkpoints, dtype=int).tolist()
        ))
        # Real epoch numbers (for display labels), from the first trial's epoch axis
        epoch_axis = self.epoch_axis(0, n_epochs)
        checkpoint_epoch_labels = [int(epoch_axis[i]) for i in checkpoint_epochs]
        renderer.render_loss_by_sequence_length(
            all_step_train=self.all_train if self.all_train else None,
            all_step_test=self.all_test,
            train_sequence_lengths=self.train_seq_lengths,
            checkpoint_epochs=checkpoint_epochs,
            checkpoint_epoch_labels=checkpoint_epoch_labels,
            max_train_len=max_train_len,
            output_file=output_file,
        )
        print(f"✓ Sequence-length loss plot saved to '{output_file}'")


class Convergence1D(Visualization):
    """1D functional convergence animation across trials."""

    _STATE_ATTRS = ('all_trials', 'axis')

    def __init__(self, axis: int = 0, sampling: int = 1, device: str = 'cpu'):
        super().__init__('convergence_1d', sampling, device=device)
        self.axis = axis
        self.f_test_frames = []      # current trial
        self.all_trials = []         # list of per-trial frame arrays

    def update(self, processor, epoch: int):
        """Extract f_test predictions."""
        if processor.logs.get('f_test'):
            f_test = processor.logs['f_test'][-1]
            self.f_test_frames.append(f_test)

    def next_trial(self):
        """Save current trial data and reset for next trial."""
        if self.f_test_frames:
            self.all_trials.append(np.array(self.f_test_frames))
        self.f_test_frames = []

    def finalize(self, output_dir: str, prefix: str):
        """Create 1D convergence animation grid using renderer backend."""
        processor_logs = getattr(self, '_processor_logs', {})
        processor_metadata = getattr(self, '_processor_metadata', {})

        x_test = processor_logs.get('x_test')
        y_test = processor_logs.get('y_test')

        if x_test is None or y_test is None:
            print(f"⊘ Skipping {self.name}: missing reference data")
            return

        # Commit current trial
        self.commit_epoch_trial()
        if self.f_test_frames:
            self.all_trials.append(np.array(self.f_test_frames))

        if not self.all_trials:
            print(f"⊘ Skipping {self.name}: no trial data")
            return

        x_test = np.nan_to_num(x_test, nan=np.nanmin(x_test)).reshape(x_test.shape[0], -1)

        # Extract 1D slice using stored axis bounds (or fallback to mask-based search)
        # Bug B fix: axis data is INTERLEAVED, not contiguous. E.g., for d=10, axis k is at
        # positions [axis_start + k, axis_start + k + d, axis_start + k + 2d, ...]
        dataset_meta = processor_metadata.get('dataset', {})
        axis_start = dataset_meta.get('x_test_axis_start')
        axis_end = dataset_meta.get('x_test_axis_end')
        x_test_axis_d = dataset_meta.get('x_test_axis_d')

        if axis_start is not None and axis_end is not None and x_test_axis_d is not None:
            # Use stored bounds with correct interleaved indexing
            d = x_test_axis_d
            n_per_axis = (axis_end - axis_start) // d
            # Axis k is interleaved at stride d: positions [axis_start+k, axis_start+k+d, ...]
            filter_indices = axis_start + np.arange(n_per_axis) * d + self.axis
        else:
            # Fallback: mask-based search (for backward compatibility with old data)
            condition = np.all(np.delete(x_test, self.axis, axis=-1) == 0, axis=1)
            filter_indices = np.where(condition)[0]

        if len(filter_indices) == 0:
            print(f"⊘ No 1D data found along axis {self.axis}")
            return

        x_1d = x_test[filter_indices, self.axis]
        y_1d = y_test[filter_indices]
        sort_indices = np.argsort(x_1d)
        x_1d, y_1d = x_1d[sort_indices], y_1d[sort_indices]

        # Prepare data for all trials
        trials_data = []
        for trial_idx, f_test in enumerate(self.all_trials):
            epochs = self.epoch_axis(trial_idx, len(f_test))

            f_1d = f_test[:, filter_indices]
            frame_indices = np.arange(len(epochs))
            f_1d = f_1d[np.ix_(frame_indices, sort_indices)]
            trials_data.append((x_1d, y_1d, f_1d, epochs, self.axis,
                                dataset_meta.get('x_range'), processor_metadata))

        output_file = f'{output_dir}/{prefix}_1d_convergence.mp4'
        renderer = PlotBackend.get_renderer()
        renderer.render_multi_1d_animation(trials_data, output_file)
        print(f"✓ 1D convergence animation grid saved to '{output_file}'")


class MultiAxisConvergence1D(Visualization):
    """Multi-axis 1D functional convergence animation across trials.

    Renders every `step`-th x-axis (0, step, 2*step, ...) as a grid animation:
    - Rows = axes
    - Columns = trials
    - Perfect for visualizing length OOD degradation in sequence models.
    """

    _STATE_ATTRS = ('all_trials', 'step')

    def __init__(self, step: int = 5, sampling: int = 1, device: str = 'cpu'):
        super().__init__('multi_axis_convergence_1d', sampling, device=device)
        self.step = step
        self.f_test_frames = []      # current trial
        self.all_trials = []         # list of per-trial frame arrays

    def update(self, processor, epoch: int):
        """Extract f_test predictions."""
        if processor.logs.get('f_test'):
            f_test = processor.logs['f_test'][-1]
            self.f_test_frames.append(f_test)

    def next_trial(self):
        """Save current trial data and reset for next trial."""
        if self.f_test_frames:
            self.all_trials.append(np.array(self.f_test_frames))
        self.f_test_frames = []

    def finalize(self, output_dir: str, prefix: str):
        """Create multi-axis 1D convergence animation grid using renderer backend."""
        processor_logs = getattr(self, '_processor_logs', {})
        processor_metadata = getattr(self, '_processor_metadata', {})

        x_test = processor_logs.get('x_test')
        y_test = processor_logs.get('y_test')

        if x_test is None or y_test is None:
            print(f"⊘ Skipping {self.name}: missing reference data")
            return

        # Commit current trial
        self.commit_epoch_trial()
        if self.f_test_frames:
            self.all_trials.append(np.array(self.f_test_frames))

        if not self.all_trials:
            print(f"⊘ Skipping {self.name}: no trial data")
            return

        x_test = np.nan_to_num(x_test, nan=np.nanmin(x_test)).reshape(x_test.shape[0], -1)

        # Get axis metadata
        dataset_meta = processor_metadata.get('dataset', {})
        axis_start = dataset_meta.get('x_test_axis_start')
        axis_end = dataset_meta.get('x_test_axis_end')
        x_test_axis_d = dataset_meta.get('x_test_axis_d')

        if axis_start is None or axis_end is None or x_test_axis_d is None:
            print(f"⊘ Skipping {self.name}: missing axis metadata (x_test_axis_start, x_test_axis_end, x_test_axis_d)")
            return

        # Compute number of points per axis
        d = x_test_axis_d
        n_per_axis = (axis_end - axis_start) // d

        # Determine which axes to show (every step-th axis)
        axes_to_show = list(range(0, d, self.step))
        if not axes_to_show:
            print(f"⊘ Skipping {self.name}: no axes selected (d={d}, step={self.step})")
            return

        # Determine in-domain vs out-of-domain samples
        x_range = dataset_meta.get('x_range', (-8, 8))
        if x_range:
            in_domain = np.all(
                (x_test >= x_range[0]) & (x_test <= x_range[1]),
                axis=1
            )
        else:
            in_domain = np.ones(len(y_test), dtype=bool)

        # Build grid_data[axis_idx][trial_idx]
        grid_data = []
        for axis_k in axes_to_show:
            axis_trials_data = []
            for trial_idx, f_test in enumerate(self.all_trials):
                # Extract this axis's data (interleaved indexing)
                filter_indices = axis_start + np.arange(n_per_axis) * d + axis_k
                x_1d = x_test[filter_indices, axis_k]
                y_1d = y_test[filter_indices]
                sort_indices = np.argsort(x_1d)
                x_1d, y_1d = x_1d[sort_indices], y_1d[sort_indices]

                # Get epoch numbers for this trial
                epochs = self.epoch_axis(trial_idx, len(f_test))

                # Slice and sort f_1d
                f_1d = f_test[:, filter_indices]
                frame_indices = np.arange(len(epochs))
                f_1d = f_1d[np.ix_(frame_indices, sort_indices)]

                axis_trials_data.append((x_1d, y_1d, f_1d, epochs, axis_k,
                                        dataset_meta.get('x_range'), processor_metadata))

            grid_data.append(axis_trials_data)

        output_file = f'{output_dir}/{prefix}_multi_axis_1d_convergence.mp4'
        renderer = PlotBackend.get_renderer()
        renderer.render_multi_axis_1d_animation(grid_data, output_file)
        print(f"✓ Multi-axis 1D convergence animation grid saved to '{output_file}'")


class PCA3D(Visualization):
    """3D PCA visualization of hidden states with domain coloring across trials."""

    _STATE_ATTRS = ('all_f_test',)

    def __init__(self, pca_epoch: int = -1, sampling: int = 1, mode: str = 'anchor', device: str = 'cpu'):
        super().__init__(f'pca_3d_{mode}', sampling, device=device)
        self.pca_epoch = pca_epoch
        self.mode = mode
        self.f_test_frames = []      # current trial
        self.all_f_test = []         # list of per-trial f_test arrays
        # HDF5 streaming for hidden states
        self._h5_path = None
        self._h5_file = None
        self._trial_idx = 0
        self._frame_idx = 0
        self._h5_trial_frames = []  # track frame keys per trial for loading

    def update(self, processor, epoch: int):
        """Extract f_test and hidden states, streaming hidden states to HDF5."""
        if processor.logs.get('f_test'):
            self.f_test_frames.append(processor.logs['f_test'][-1])
        if processor.logs.get('hidden_states'):
            # Initialize HDF5 file on first write
            if self._h5_file is None:
                import tempfile
                import uuid
                # Write to the config's output dir (roomy project/scratch FS),
                # not /tmp (small/RAM-backed on HPC nodes).
                scratch_dir = getattr(self, '_scratch_dir', None) or os.getcwd()
                os.makedirs(scratch_dir, exist_ok=True)
                self._h5_path = tempfile.mktemp(
                    dir=scratch_dir, suffix=f'_pca_{uuid.uuid4().hex}.h5')
                self._h5_file = h5py.File(self._h5_path, 'w')
                self._h5_trial_frames.append([])  # add list for first trial

            # Write hidden states to HDF5 (gzip-compressed to cut disk footprint)
            hidden = processor.logs['hidden_states'][-1]
            key = f'trial_{self._trial_idx}/frame_{self._frame_idx}'
            self._h5_file.create_dataset(key, data=hidden,
                                         compression='gzip', compression_opts=4)
            self._h5_trial_frames[-1].append(key)
            self._frame_idx += 1

    def next_trial(self):
        """Save current trial data and reset for next trial."""
        if self.f_test_frames:
            self.all_f_test.append(np.array(self.f_test_frames))
        self.f_test_frames = []
        # Prepare for next trial in HDF5
        if self._h5_file is not None:
            self._trial_idx += 1
            self._frame_idx = 0
            self._h5_trial_frames.append([])

    def finalize(self, output_dir: str, prefix: str):
        """Create 3D PCA animation grid across trials using parallel frame computation."""
        processor_logs = getattr(self, '_processor_logs', {})
        processor_metadata = getattr(self, '_processor_metadata', {})

        x_test = processor_logs.get('x_test')
        y_test = processor_logs.get('y_test')
        x_range = processor_metadata.get('dataset', {}).get('x_range', (-8, 8))

        # Commit current trial if any
        self.commit_epoch_trial()
        if self.f_test_frames:
            self.all_f_test.append(np.array(self.f_test_frames))

        if not self.all_f_test:
            print(f"⊘ No frames for {self.name}")
            return

        if x_test is None or y_test is None:
            print(f"⊘ Skipping {self.name}: missing reference data")
            return

        x_test = np.nan_to_num(x_test, nan=np.nanmin(x_test)).reshape(x_test.shape[0], -1)

        # Determine in-domain vs out-of-domain samples
        if x_range:
            in_domain = np.all(
                (x_test >= x_range[0]) & (x_test <= x_range[1]),
                axis=1
            )
        else:
            in_domain = np.ones(len(y_test), dtype=bool)

        # Use process-based parallelization for CPU-bound frame computation
        queue = FrameComputationQueue()
        trials_data = []

        try:
            for trial_idx, f_test in enumerate(self.all_f_test):
                # Load hidden states from HDF5 or use fallback
                if self._h5_file is not None and trial_idx < len(self._h5_trial_frames):
                    frame_keys = self._h5_trial_frames[trial_idx]
                else:
                    # Fallback: shouldn't happen in normal flow
                    print(f"⊘ Warning: No HDF5 data for trial {trial_idx}")
                    continue

                # Get epoch numbers
                epochs = self.epoch_axis(trial_idx, len(frame_keys))

                # frame_keys already reflects Visualizer-level sampling (update()
                # is only ever called on sampled epochs), so use every frame here -
                # do not re-apply self.sampling, or these indices desync from the
                # (unresampled) epochs/f_test arrays below.
                frame_indices = np.arange(len(frame_keys))

                if self.mode == 'procrustes':
                    # Parallel PCA fitting across sampled epochs (lazy load frames)
                    def _fit_and_transform_pca(frame_idx):
                        h = self._h5_file[frame_keys[int(frame_idx)]][()]
                        n_comps = min(2, h.shape[0], h.shape[1])
                        if n_comps < 1:
                            return np.zeros((h.shape[0], 2))
                        pca = PCA(n_components=n_comps)
                        pca.fit(h)
                        result = pca.transform(h)
                        if n_comps < 2:
                            result = np.pad(result, ((0, 0), (0, 2 - n_comps)))
                        return result

                    with ThreadPoolExecutor(max_workers=max(1, multiprocessing.cpu_count() - 1)) as executor:
                        hidden_2d_list = list(executor.map(_fit_and_transform_pca, frame_indices))

                    # Queue Procrustes alignment to run in background
                    task_id = f'procrustes_trial_{trial_idx}'
                    queue.submit_procrustes_alignment(task_id, hidden_2d_list)
                    hidden_2d_aligned = queue.get_result(task_id)

                    if len(hidden_2d_aligned) == 0:
                        print(f"⊘ Warning: No frames for procrustes trial {trial_idx}, skipping")
                        continue

                    pc1_frames = np.array([h[:, 0] for h in hidden_2d_aligned])
                    pc2_frames = np.array([h[:, 1] for h in hidden_2d_aligned])
                else:
                    # Anchor mode: fit PCA once at reference epoch, then transform all frames lazily
                    pca_frame_idx = self.pca_epoch if self.pca_epoch >= 0 else len(frame_keys) - 1
                    reference = self._h5_file[frame_keys[pca_frame_idx]][()]
                    n_comps = min(2, reference.shape[0], reference.shape[1])
                    if n_comps < 1:
                        del reference
                        continue
                    pca = PCA(n_components=n_comps)
                    pca.fit(reference)
                    del reference

                    # Transform all frames (lazy load one at a time)
                    pc1_frames, pc2_frames = [], []
                    for key in [frame_keys[i] for i in range(len(frame_keys))]:
                        h = self._h5_file[key][()]
                        t = pca.transform(h)
                        if n_comps < 2:
                            t = np.pad(t, ((0, 0), (0, 2 - n_comps)))
                        pc1_frames.append(t[:, 0])
                        pc2_frames.append(t[:, 1])
                    pc1_frames = np.array(pc1_frames)
                    pc2_frames = np.array(pc2_frames)

                trials_data.append((pc1_frames, pc2_frames, f_test, y_test, in_domain,
                                    epochs, self.mode, processor_metadata))

            output_file = f'{output_dir}/{prefix}_pca_3d_{self.mode}.mp4'
            renderer = PlotBackend.get_renderer()
            renderer.render_multi_3d_animation(trials_data, output_file)
            print(f"✓ PCA 3D animation grid ({self.mode}) saved to '{output_file}'")
        finally:
            queue.shutdown()
            # NOTE: the streamed hidden-state HDF5 is intentionally NOT closed or
            # deleted here — finalize() may be called again (live refresh renders
            # animations after round 0 and at the end) and additional trials may
            # still be streamed into the same file. It is released by cleanup(),
            # which the Visualizer calls once all finalization is complete.

    def _state_dict(self) -> dict:
        d = super()._state_dict()
        # HDF5 frame-key bookkeeping needed to resume streaming on extend.
        d['_h5_trial_frames'] = [list(t) for t in self._h5_trial_frames]
        d['_trial_idx'] = self._trial_idx
        return d

    def _apply_state(self, d: dict):
        super()._apply_state(d)
        self._h5_trial_frames = [list(t) for t in d.get('_h5_trial_frames', [])]
        self._trial_idx = d.get('_trial_idx', len(self.all_f_test))

    def save_state(self, state_dir: str):
        os.makedirs(state_dir, exist_ok=True)
        with open(os.path.join(state_dir, f'{self.name}.state.pkl'), 'wb') as f:
            pickle.dump(self._state_dict(), f)
        # Persist the streamed hidden-state HDF5 alongside the pickle so a later
        # extend can reload prior trials' hidden states and append new ones.
        if self._h5_file is not None and self._h5_path and os.path.exists(self._h5_path):
            self._h5_file.flush()
            shutil.copyfile(self._h5_path, os.path.join(state_dir, f'{self.name}_hidden.h5'))

    def load_state(self, state_dir: str) -> bool:
        path = os.path.join(state_dir, f'{self.name}.state.pkl')
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            self._apply_state(pickle.load(f))
        h5_src = os.path.join(state_dir, f'{self.name}_hidden.h5')
        if os.path.exists(h5_src):
            # Copy into the scratch dir and reopen in append mode so new trials'
            # hidden states stream into the same file (keys trial_{_trial_idx}/...).
            scratch_dir = getattr(self, '_scratch_dir', None) or os.getcwd()
            os.makedirs(scratch_dir, exist_ok=True)
            import uuid, tempfile
            self._h5_path = tempfile.mktemp(
                dir=scratch_dir, suffix=f'_pca_{uuid.uuid4().hex}.h5')
            shutil.copyfile(h5_src, self._h5_path)
            self._h5_file = h5py.File(self._h5_path, 'a')
            self._frame_idx = 0
        return True

    def cleanup(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
                if self._h5_path and os.path.exists(self._h5_path):
                    os.remove(self._h5_path)
            except Exception as e:
                print(f"⊘ Warning: Failed to clean up HDF5 file: {e}")
            finally:
                self._h5_file = None


class FunctionSpaceConvergence(Visualization):
    """Convergence in PCA-projected function space across trials."""

    _STATE_ATTRS = ('all_trials',)

    def __init__(self, sampling: int = 1, device: str = 'cpu'):
        super().__init__('function_space', sampling, device=device)
        self.f_test_frames = []      # current trial
        self.all_trials = []         # list of per-trial frame arrays

    def update(self, processor, epoch: int):
        """Extract f_test predictions."""
        if processor.logs.get('f_test'):
            self.f_test_frames.append(processor.logs['f_test'][-1])

    def next_trial(self):
        """Save current trial data and reset for next trial."""
        if self.f_test_frames:
            self.all_trials.append(np.array(self.f_test_frames))
        self.f_test_frames = []

    def finalize(self, output_dir: str, prefix: str):
        """Create function space convergence plot grid with GPU-accelerated PCA."""
        # Commit current trial
        if self.f_test_frames:
            self.all_trials.append(np.array(self.f_test_frames))

        if not self.all_trials:
            print(f"⊘ No frames for {self.name}")
            return

        y_test = self._processor_logs.get('y_test')

        # Parallel GPU-accelerated PCA across trials using thread pool
        trials_data = []
        with ThreadPoolExecutor(max_workers=max(1, multiprocessing.cpu_count() - 1)) as executor:
            futures = [executor.submit(_compute_pca_projection, f_test, y_test)
                      for f_test in self.all_trials]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    trials_data.append(result)

        if not trials_data:
            print(f"⊘ Skipping {self.name}: unable to prepare data")
            return

        output_file = f'{output_dir}/{prefix}_function_space.png'
        renderer = PlotBackend.get_renderer()
        renderer.render_multi_3d_scatter(trials_data, output_file)
        print(f"✓ Function space plot grid saved to '{output_file}'")


# ============================================================================
# Main Visualizer Class
# ============================================================================

class Visualizer:
    """
    Simplified visualizer that works with processor directly.
    Users just implement their own Visualization subclasses.
    Supports GPU-accelerated visualization via device parameter.

    Supports background finalization for parallel execution with training:
        visualizer.finalize(background=True)  # Returns immediately, renders in thread pool
        visualizer.wait_for_background()      # Wait for background tasks to complete
    """

    def __init__(self, name, output_dir='visualizations', sampling: int = 1, device: str = 'cpu'):
        self.name = name or "model"
        self.output_dir = output_dir
        self.sampling = sampling
        self.device = device
        self.visualizations: Dict[str, Visualization] = {}
        self.processor = None
        self.logs = {}
        self.metadata = {}
        self._finalization_executor = ThreadPoolExecutor(max_workers=1)
        self._finalization_futures = []

    def register(self, visualization: Visualization):
        """Register a visualization instance."""
        # Propagate device to visualization
        visualization.device = self.device
        # Propagate sampling so fallback epoch numbering is correct
        visualization.sampling = self.sampling
        # Propagate scratch directory (config output_dir) so visualizations that
        # stream temp files write to the roomy project/scratch FS, not /tmp.
        # Use abspath so relative output_dir is anchored to submission cwd, not
        # whatever cwd SLURM sets when the job actually runs.
        visualization._scratch_dir = os.path.abspath(self.output_dir)
        self.visualizations[visualization.name] = visualization

    def attach_processor(self, processor):
        """Attach processor for dynamic data access."""
        self.processor = processor

    @classmethod
    def from_processor_data(cls, processor, name: str = None):
        """
        Create a visualizer from already-trained processor data.
        Useful for post-training visualization or standalone visualization tasks.

        Args:
            processor: Processor instance with completed training
            name: Optional name for the visualizer

        Returns:
            Visualizer instance with processor data attached
        """
        visualizer = cls(name=name or processor.__class__.__name__)
        visualizer.processor = processor
        if hasattr(processor, 'logs'):
            visualizer.logs = processor.logs
        if hasattr(processor, 'metadata'):
            visualizer.metadata = processor.metadata
        return visualizer

    def update(self, epoch: int):
        """Update all registered visualizations."""
        if self.processor is None:
            raise RuntimeError("Processor not attached. Call visualizer.attach_processor(processor)")

        # Sample epochs
        if epoch % self.sampling == 0:
            for viz in self.visualizations.values():
                viz.record_epoch(epoch)
                viz.update(self.processor, epoch)

    def begin_epoch(self, epoch: int):
        """Forward the start-of-train-epoch hook to all visualizations."""
        for viz in self.visualizations.values():
            viz.begin_epoch(self.processor, epoch)

    def record_batch(self, epoch: int, batch: dict):
        """Forward one training-batch record to all visualizations."""
        for viz in self.visualizations.values():
            viz.record_batch(self.processor, epoch, batch)

    def end_epoch(self, epoch: int):
        """Forward the end-of-train-epoch hook to all visualizations."""
        for viz in self.visualizations.values():
            viz.end_epoch(self.processor, epoch)

    def next_trial(self):
        """Mark end of trial for all registered visualizations (accumulate trial data)."""
        for viz in self.visualizations.values():
            viz.commit_epoch_trial()
            viz.next_trial()

    def detach_processor(self):
        """Stash the live processor's logs/metadata onto each registered
        visualization, then drop the processor reference so its (GPU) model can
        be freed. Called before rendering (finalize) and by the experiment's
        snapshot-and-detach path after each per-config state snapshot. The
        stashed logs/metadata are what save_state() and the geometric finalizers
        read once the live processor is gone.
        """
        if self.processor is None:
            return
        for viz in self.visualizations.values():
            if hasattr(self.processor, 'logs'):
                viz._processor_logs = self.processor.logs
            if hasattr(self.processor, 'metadata'):
                viz._processor_metadata = self.processor.metadata
        self.processor = None

    def finalize(self, prefix: Optional[str] = None, background: bool = False):
        """
        Finalize all visualizations.

        Args:
            prefix: Output filename prefix (defaults to visualizer name)
            background: If True, render in background thread pool (returns immediately)
                       Call wait_for_background() to block until complete
        """
        os.makedirs(self.output_dir, exist_ok=True)
        if prefix is None:
            prefix = self.name

        # Stash processor logs/metadata onto each viz and release the processor
        # (frees the GPU model); _finalize_visualizations only reads the stash.
        self.detach_processor()

        if background:
            # Submit finalization to background thread pool
            future = self._finalization_executor.submit(
                self._finalize_visualizations, prefix
            )
            self._finalization_futures.append(future)
            print(f"✓ Visualization finalization queued for background execution")
        else:
            # Finalize immediately
            self._finalize_visualizations(prefix)

    def _finalize_visualizations(self, prefix: str):
        """Internal method to finalize all visualizations (runs in thread pool if background=True)."""
        # Serialize all matplotlib rendering: pyplot's global figure state and the
        # mathtext parser are not thread-safe, so concurrent finalizations of
        # multiple Visualizers (each on its own background thread) would otherwise
        # race and raise spurious mathtext ParseExceptions.
        with _RENDER_LOCK:
            # Use non-interactive backend for thread-safe matplotlib operations
            original_backend = plt.get_backend()
            try:
                if original_backend not in ['Agg', 'agg']:
                    plt.switch_backend('Agg')

                for viz_name, viz in self.visualizations.items():
                    print(f"Finalizing {viz_name}...")
                    try:
                        viz.finalize(self.output_dir, prefix)
                    except Exception as e:
                        print(f"✗ Error finalizing {viz_name}: {e}, traceback: {traceback.format_exc()}")
            finally:
                # Restore original backend
                try:
                    plt.switch_backend(original_backend)
                except:
                    pass

    def wait_for_background(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for any background finalization tasks to complete.

        Args:
            timeout: Maximum seconds to wait (None = wait indefinitely)

        Returns:
            True if all tasks completed, False if timeout occurred
        """
        if not self._finalization_futures:
            return True

        try:
            for future in as_completed(self._finalization_futures, timeout=timeout):
                future.result()  # Re-raise any exceptions from background tasks
            self._finalization_futures.clear()
            return True
        except TimeoutError:
            print(f"⚠ Timeout waiting for background visualization tasks")
            return False
        finally:
            pass

    # ========================================================================
    # Cross-run state persistence (for post-hoc extension / merge)
    # ========================================================================
    def save_state(self, state_dir: Optional[str] = None):
        """Persist every registered visualization's committed per-trial data to
        ``state_dir`` (default ``output_dir/viz_state``) so a later extend can
        reload prior trials and render combined figures alongside new trials."""
        state_dir = state_dir or os.path.join(self.output_dir, 'viz_state')
        os.makedirs(state_dir, exist_ok=True)

        # Reference data (x_test/y_test/metadata) needed by geometric finalizers
        # when re-rendering without a live processor. Prefer the live processor;
        # fall back to whatever was stashed on a viz during the last finalize.
        logs, metadata = None, None
        if self.processor is not None and getattr(self.processor, 'logs', None) is not None:
            logs, metadata = self.processor.logs, getattr(self.processor, 'metadata', {})
        else:
            for viz in self.visualizations.values():
                if getattr(viz, '_processor_logs', None) is not None:
                    logs = viz._processor_logs
                    metadata = getattr(viz, '_processor_metadata', {})
                    break
        if logs is not None:
            ref = {
                'x_test': np.asarray(logs['x_test']) if logs.get('x_test') is not None else None,
                'y_test': np.asarray(logs['y_test']) if logs.get('y_test') is not None else None,
                'metadata': metadata or {},
            }
            with open(os.path.join(state_dir, 'reference.pkl'), 'wb') as f:
                pickle.dump(ref, f)

        for viz in self.visualizations.values():
            try:
                viz.save_state(state_dir)
            except Exception as e:
                print(f"⊘ Warning: failed to save state for {viz.name}: {e}")

    def load_state(self, state_dir: Optional[str] = None) -> bool:
        """Restore per-visualization state previously written by save_state.

        Returns True if a state directory was found. Sets each viz's reference
        logs/metadata so figures can be rendered even before any new trial runs.
        """
        state_dir = state_dir or os.path.join(self.output_dir, 'viz_state')
        if not os.path.isdir(state_dir):
            return False
        ref = None
        ref_path = os.path.join(state_dir, 'reference.pkl')
        if os.path.exists(ref_path):
            with open(ref_path, 'rb') as f:
                ref = pickle.load(f)
        for viz in self.visualizations.values():
            try:
                viz.load_state(state_dir)
            except Exception as e:
                print(f"⊘ Warning: failed to load state for {viz.name}: {e}")
            if ref is not None:
                viz._processor_logs = {'x_test': ref.get('x_test'), 'y_test': ref.get('y_test')}
                viz._processor_metadata = ref.get('metadata', {})
        return True

    def cleanup(self):
        """Release persistent resources (open HDF5 handles) held by any viz."""
        for viz in self.visualizations.values():
            try:
                viz.cleanup()
            except Exception as e:
                print(f"⊘ Warning: cleanup failed for {getattr(viz, 'name', '?')}: {e}")

    # ========================================================================
    # Convenience registration methods
    # ========================================================================
    def register_defaults(self):
        """Register default visualizations."""
        self.register_loss_history()
        self.register_function_space_convergence()
        self.register_intermediate_loss_history()
        self.register_sequence_length_loss()
        self.register_multi_axis_convergence_1d(step=5)
        self.register_pca_3d_procrustes()

    def register_loss_history(self):
        """Register loss history plot."""
        self.register(LossHistoryPlot(device=self.device))

    def register_intermediate_loss_history(self):
        """Register per-time-step loss history plot (requires cot > 0)."""
        self.register(IntermediateLossHistory(device=self.device))

    def register_sequence_length_loss(self):
        """Register loss-vs-sequence-length plot (requires cot > 0)."""
        self.register(SequenceLengthLossPlot(device=self.device))

    def register_function_space_convergence(self):
        self.register(FunctionSpaceConvergence(device=self.device))

    def register_convergence_1d(self, axis: int = 0):
        """Register 1D convergence visualization."""
        self.register(Convergence1D(axis=axis, device=self.device))

    def register_pca_3d(self, pca_epoch: int = -1):
        """Register 3D PCA visualization (anchor mode)."""
        self.register(PCA3D(pca_epoch=pca_epoch, mode='anchor', device=self.device))

    def register_pca_3d_procrustes(self):
        """Register 3D PCA visualization (Procrustes mode)."""
        self.register(PCA3D(pca_epoch='all', mode='procrustes', device=self.device))

    def register_multi_axis_convergence_1d(self, step: int = 5):
        """Register multi-axis 1D convergence visualization (every step-th axis)."""
        self.register(MultiAxisConvergence1D(step=step, device=self.device))

    def __del__(self):
        """Cleanup thread pools and any open viz resources on deletion."""
        try:
            self.cleanup()
        except:
            pass
        try:
            self._finalization_executor.shutdown(wait=False)
        except:
            pass

    def load_training_data(self, h5_filename='train_out/training_data.h5'):
        """Load training data from HDF5 file."""
        self.logs = {}
        self.metadata = {}
        with h5py.File(h5_filename, 'r') as hf:
            self.logs = {key: hf['logs'][key][()] for key in hf['logs'].keys()}
            self.metadata = {key: hf['metadata'][key][()] for key in hf['metadata'].keys()}

    def visualize_full(self, name):
        """Legacy method for loading and visualizing from file."""
        self.load_training_data(f'train_out/{name}.h5')
        # Register default visualizations
        self.register_loss_history()
        self.register_convergence_1d(axis=0)
        self.register_pca_3d(pca_epoch=-1)
        # Create mock processor for finalization
        # This is for backward compatibility
