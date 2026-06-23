"""
Simplified Visualization System
Visualizer takes processor as parameter and dynamically extracts data needed.
Each visualization is simple: just implement update() and finalize() methods.
"""
import traceback
import warnings
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
import h5py
import os

# ============================================================================
# Base Visualization Class - Users inherit from this
# ============================================================================

class Visualization(ABC):
    """Base class for all visualizations. Users implement update() and finalize()."""

    def __init__(self, name: str, sampling: int = 1):
        self.name = name
        self.sampling = sampling
        self.frames = []

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
    """Plot training and test loss."""

    def __init__(self):
        super().__init__('loss_history')
        self.train_losses = []
        self.test_losses = []

    def update(self, processor, epoch: int):
        """Extract loss values from processor."""
        if processor.logs['train_loss']:
            self.train_losses.append(processor.logs['train_loss'][-1])
        if processor.logs['test_loss']:
            self.test_losses.append(processor.logs['test_loss'][-1])

    def finalize(self, output_dir: str, prefix: str):
        """Create loss plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.train_losses, linewidth=2, label='Train Loss', alpha=0.8)
        ax.plot(self.test_losses, linewidth=2, label='Test Loss', alpha=0.8)
        ax.set_title('Training Loss History', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MSE Loss', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=11)
        fig.tight_layout()

        output_file = f'{output_dir}/{prefix}_loss_history.png'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_file, dpi=150)
        plt.close(fig)
        print(f"✓ Loss history plot saved to '{output_file}'")


class Convergence1D(Visualization):
    """1D functional convergence animation."""

    def __init__(self, axis: int = 0, sampling: int = 1):
        super().__init__('convergence_1d', sampling)
        self.axis = axis
        self.f_test_frames = []

    def update(self, processor, epoch: int):
        """Extract f_test predictions."""
        if processor.logs.get('f_test'):
            f_test = processor.logs['f_test'][-1]  # Last appended frame
            self.f_test_frames.append(f_test)

    def finalize(self, output_dir: str, prefix: str):
        """Create 1D convergence animation."""
        processor_logs = getattr(self, '_processor_logs', {})
        processor_metadata = getattr(self, '_processor_metadata', {})

        x_test = processor_logs.get('x_test')
        y_test = processor_logs.get('y_test')

        if x_test is None or y_test is None:
            print(f"⊘ Skipping {self.name}: missing reference data")
            return

        f_test = np.array(self.f_test_frames)
        epochs = np.arange(0, len(f_test), self.sampling)

        # Extract 1D slice
        condition = np.all(np.delete(x_test, self.axis, axis=-1) == 0, axis=1)
        filter_indices = np.where(condition)[0]

        if len(filter_indices) == 0:
            print(f"⊘ No 1D data found along axis {self.axis}")
            return

        x_1d = x_test[filter_indices, self.axis]
        y_1d = y_test[filter_indices]
        f_1d = f_test[:, filter_indices]

        sort_indices = np.argsort(x_1d)
        x_1d, y_1d = x_1d[sort_indices], y_1d[sort_indices]
        f_1d = f_1d[np.ix_(epochs, sort_indices)]

        # Create animation
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(x_1d, y_1d, 'k--', linewidth=3, label='Ground Truth', zorder=2)
        line_anim, = ax.plot([], [], 'b-', linewidth=2, label='Network Prediction', zorder=1)
        epoch_text = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        x_min, x_max = processor_metadata.get('x_range', (x_1d.min(), x_1d.max()))
        ax.axvspan(xmin=x_min, xmax=x_max, color='blue', alpha=0.1, label='Training domain')

        x_margin = (x_1d.max() - x_1d.min()) * 0.05
        y_min, y_max = float(min(y_1d.min(), f_1d.min())), float(max(y_1d.max(), f_1d.max()))
        y_margin = (y_max - y_min) * 0.1

        ax.set_xlim(x_1d.min() - x_margin, x_1d.max() + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_xlabel(f'$x_{self.axis}$', fontsize=12)
        ax.set_ylabel('Output', fontsize=12)
        ax.set_title(f'Functional Convergence along $x_{self.axis}$ axis', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.4)
        epoch_text = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.legend(fontsize=11)

        def init():
            line_anim.set_data([], [])
            epoch_text.set_text('')
            return line_anim, epoch_text

        def update_frame(frame_idx):
            line_anim.set_data(x_1d, f_1d[frame_idx])
            epoch_text.set_text(f'Epoch: {frame_idx * self.sampling}')
            return line_anim, epoch_text

        anim = FuncAnimation(fig, update_frame, frames=len(epochs), init_func=init,
                            blit=True, interval=50)

        output_file = f'{output_dir}/{prefix}_1d_convergence.mp4'
        os.makedirs(output_dir, exist_ok=True)
        anim.save(output_file, writer='ffmpeg', fps=15)
        plt.close(fig)
        print(f"✓ 1D convergence animation saved to '{output_file}'")


class PCA3D(Visualization):
    """3D PCA visualization of hidden states with domain coloring."""

    def __init__(self, pca_epoch: int = -1, sampling: int = 1, mode: str = 'anchor'):
        super().__init__(f'pca_3d_{mode}', sampling)
        self.pca_epoch = pca_epoch
        self.mode = mode  # 'anchor' or 'procrustes'
        self.f_test_frames = []
        self.hidden_frames = []

    def update(self, processor, epoch: int):
        """Extract f_test and hidden states."""
        if processor.logs.get('f_test'):
            self.f_test_frames.append(processor.logs['f_test'][-1])
        if processor.logs.get('hidden_states'):
            self.hidden_frames.append(processor.logs['hidden_states'][-1])

    def finalize(self, output_dir: str, prefix: str):
        """Create 3D PCA animation with in-domain/out-of-domain coloring."""
        x_test = self._processor_logs.get('x_test')
        y_test = self._processor_logs.get('y_test')
        x_range = self._processor_metadata.get('x_range', (-8, 8))

        if not self.f_test_frames or not self.hidden_frames:
            print(f"⊘ No frames for {self.name}")
            return

        f_test = np.array(self.f_test_frames) # (E, N)
        hidden_states = np.array(self.hidden_frames)
        epochs = np.arange(0, len(f_test), self.sampling)

        # Determine in-domain vs out-of-domain samples
        if x_test is not None and x_range:
            in_domain = np.all(
                (x_test >= x_range[0]) & (x_test <= x_range[1]),
                axis=1
            )
        else:
            in_domain = np.ones(len(y_test), dtype=bool)

        # PCA fitting
        if self.mode == 'procrustes':
            pca_list = []
            hidden_2d_list = []
            for epoch_idx in epochs:
                pca = PCA(n_components=2)
                pca.fit(hidden_states[epoch_idx])
                hidden_2d = pca.transform(hidden_states[epoch_idx])
                pca_list.append(pca)
                hidden_2d_list.append(hidden_2d)

            # Procrustes alignment
            for epoch_idx in range(1, len(epochs)):
                R, _ = orthogonal_procrustes(hidden_2d_list[epoch_idx], hidden_2d_list[epoch_idx - 1])
                hidden_2d_list[epoch_idx] @= R

            pc1_frames = np.array([h[:, 0] for h in hidden_2d_list])
            pc2_frames = np.array([h[:, 1] for h in hidden_2d_list])
            pc1_anchor = pc1_frames[0]
            pc2_anchor = pc2_frames[0]
        else:
            pca = PCA(n_components=2)
            pca.fit(hidden_states[self.pca_epoch])
            pc1_frames = np.array([pca.transform(hidden_states[e])[:, 0] for e in range(len(self.f_test_frames))])
            pc2_frames = np.array([pca.transform(hidden_states[e])[:, 1] for e in range(len(self.f_test_frames))])
            hidden_2d_anchor = pca.transform(hidden_states[self.pca_epoch])
            pc1_anchor = hidden_2d_anchor[:, 0]
            pc2_anchor = hidden_2d_anchor[:, 1]

        # Create 3D animation
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface with colors
        colors = np.where(in_domain, 'blue', 'red')  # blue for in-domain, red for out
        surf_plot = ax.plot_trisurf(pc1_anchor, pc2_anchor, y_test, cmap='viridis', alpha=0.3)

        scatter_in = ax.scatter([], [], [], c='blue', label='In-domain', s=30, alpha=0.8)
        scatter_out = ax.scatter([], [], [], c='red', label='Out-of-domain', s=30, alpha=0.8)
        epoch_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        def update_frame(frame_idx):
            pc1 = pc1_frames[frame_idx]
            pc2 = pc2_frames[frame_idx]
            z = f_test[frame_idx]
            if self.mode == 'procrustes':
                nonlocal surf_plot
                surf_plot.remove()
                surf_plot = ax.plot_trisurf(pc1, pc2, y_test, cmap='viridis', alpha=0.3,
                                            label='Ground Truth Surface')
            scatter_in._offsets3d = (pc1[in_domain], pc2[in_domain], z[in_domain])
            scatter_out._offsets3d = (pc1[~in_domain], pc2[~in_domain], z[~in_domain])
            epoch_text.set_text(f'Epoch: {frame_idx * self.sampling}')
            return scatter_in, scatter_out, epoch_text

        # Set limits and labels
        pc1_all = pc1_frames.flatten()
        pc2_all = pc2_frames.flatten()
        ax.set_xlim(pc1_all.min(), pc1_all.max())
        ax.set_ylim(pc2_all.min(), pc2_all.max())
        ax.set_zlim(min(f_test.min(), y_test.min()), max(f_test.max(), y_test.max()))
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('Output')
        ax.set_title(f'Hidden State Convergence in PCA Space ({self.mode} mode)', fontsize=14)
        ax.legend(fontsize=10, loc='upper right')

        anim = FuncAnimation(fig, update_frame, frames=len(epochs), interval=50)

        output_file = f'{output_dir}/{prefix}_pca_3d_{self.mode}.mp4'
        os.makedirs(output_dir, exist_ok=True)
        anim.save(output_file, writer='ffmpeg', fps=15)
        plt.close(fig)
        print(f"✓ PCA 3D animation ({self.mode}) saved to '{output_file}'")


class FunctionSpaceConvergence(Visualization):
    """Convergence in PCA-projected function space."""

    def __init__(self, sampling: int = 1):
        super().__init__('function_space', sampling)
        self.f_test_frames = []

    def update(self, processor, epoch: int):
        """Extract f_test predictions."""
        if processor.logs.get('f_test'):
            self.f_test_frames.append(processor.logs['f_test'][-1])

    def finalize(self, output_dir: str, prefix: str):
        """Create function space convergence plot."""
        if not self.f_test_frames:
            print(f"⊘ No frames for {self.name}")
            return

        f_test = np.array(self.f_test_frames)
        y_test = self._processor_logs.get('y_test')

        pca = PCA(n_components=3)
        pca.fit(f_test)
        f_3d = pca.transform(f_test)
        y_3d = pca.transform(y_test[None, :])[0]

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

        output_file = f'{output_dir}/{prefix}_function_space.png'
        os.makedirs(output_dir, exist_ok=True)
        fig.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close(fig)
        print(f"✓ Function space plot saved to '{output_file}'")


# ============================================================================
# Main Visualizer Class
# ============================================================================

class Visualizer:
    """
    Simplified visualizer that works with processor directly.
    Users just implement their own Visualization subclasses.
    """

    def __init__(self, name, sampling: int = 1):
        self.name = name or "model"
        self.sampling = sampling
        self.visualizations: Dict[str, Visualization] = {}
        self.processor = None

    def register(self, visualization: Visualization):
        """Register a visualization instance."""
        self.visualizations[visualization.name] = visualization

    def attach_processor(self, processor):
        """Attach processor for dynamic data access."""
        self.processor = processor

    def update(self, epoch: int):
        """Update all registered visualizations."""
        if self.processor is None:
            raise RuntimeError("Processor not attached. Call visualizer.attach_processor(processor)")

        # Sample epochs
        if epoch % self.sampling == 0:
            for viz in self.visualizations.values():
                viz.update(self.processor, epoch)

    def finalize(self, output_dir: str = 'visualizations/topk-sum', prefix: Optional[str] = None):
        """Finalize all visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        if prefix is None:
            prefix = self.name

        # Store processor data for access in finalize
        for viz in self.visualizations.values():
            if hasattr(self.processor, 'logs'):
                viz._processor_logs = self.processor.logs
            if hasattr(self.processor, 'metadata'):
                viz._processor_metadata = self.processor.metadata

        for viz_name, viz in self.visualizations.items():
            print(f"Finalizing {viz_name}...")
            try:
                viz.finalize(output_dir, prefix)
            except Exception as e:
                print(f"✗ Error finalizing {viz_name}: {e}, traceback: {traceback.format_exc()}")

    # ========================================================================
    # Convenience registration methods
    # ========================================================================

    def register_loss_history(self):
        """Register loss history plot."""
        self.register(LossHistoryPlot())

    def register_convergence_1d(self, axis: int = 0):
        """Register 1D convergence visualization."""
        self.register(Convergence1D(axis=axis))

    def register_pca_3d(self, pca_epoch: int = -1):
        """Register 3D PCA visualization (anchor mode)."""
        self.register(PCA3D(pca_epoch=pca_epoch, mode='anchor'))

    def register_pca_3d_procrustes(self):
        """Register 3D PCA visualization (Procrustes mode)."""
        self.register(PCA3D(pca_epoch='all', mode='procrustes'))


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


def main():
    """Example: Using the new simplified system."""
    from train import Processor
    from models import MLP
    from datasets import topksubset
    import torch.optim as optim

    # Create visualizer with desired visualizations
    visualizer = Visualizer(sampling=10)
    visualizer.register_loss_history()
    visualizer.register_convergence_1d(axis=0)

    # Create processor (doesn't need to know about visualization details)
    processor = Processor(
        model=MLP(input_dim=10),
        epochs=100,
        visualizer=visualizer
    )

    # visualizer is used during training
    # After training, finalize outputs
    visualizer.finalize()


if __name__ == '__main__':
    main()
