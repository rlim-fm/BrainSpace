"""
Example demonstrating the new streaming visualization system.

This shows how to:
1. Create a Visualizer and register desired visualizations
2. Pass it to Processor for training
3. Processor logs only needed data during training
4. Visualizer generates animations automatically after training completes
5. Memory efficient - data is not stored between epochs for registered visualizations
"""

from models import MLP, SimpleTransformerModel
from datasets import topksubset
from train import Processor
from visualization import *
import torch.optim as optim
import torch.nn as nn

def example_streaming_training():
    """Train with streaming visualizations - memory efficient."""

    # Create visualizer with desired visualizations
    visualizer = Visualizer()
    visualizer.register_loss_history()  # Loss plot
    visualizer.register_convergence_1d(axis=0)  # 1D convergence
    visualizer.register_pca_3d()  # 3D PCA (anchor mode)
    visualizer.register_pca_3d_procrustes()  # 3D PCA (Procrustes mode)
    visualizer.register(FunctionSpaceConvergence())

    model = MLP(input_dim=10, dropout=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.8)

    processor = Processor(
        x_range=(-8, 8),
        data_dim=(10,),
        N=2048,
        ground_truth=topksubset(3, 1),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=1000,
        criterion=nn.MSELoss(reduction='mean'),
        visualizer=visualizer,
        seed=42
    )

    print("Training with streaming visualizations...")
    processor.run()
    processor.print_summary()

    # 4. Processor automatically calls visualizer.finalize_visualizations()
    #    This generates all animations from the buffered frames

    print("\n✓ Training and visualization complete!")
    print("Output files will be saved to 'visualizations/topk-sum/'")


def example_custom_visualization():
    """
    Example: Creating a custom visualization method.

    For future compatibility, just:
    1. Register with visualizer.register_visualization(name, data_types={...})
    2. Add a _finalize_* method to the Visualizer class
    3. Add case to finalize_visualizations() switch statement
    """

    # This is extensible! A future visualization like:
    # visualizer.register_visualization(
    #     name='my_custom_viz',
    #     data_types={DataType.F_TEST, DataType.HIDDEN_STATES}
    # )
    #
    # Would automatically receive those data types during training via visualizer.update(),
    # and you'd implement _finalize_my_custom_viz to process the buffered frames.
    pass


def example_traditional_logging():
    """Traditional approach - log everything, then visualize (uses more memory)."""

    # If you want the old behavior (log all data), just don't pass a visualizer
    model = MLP(input_dim=10, dropout=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    processor = Processor(
        x_range=(-8, 8),
        data_dim=(10,),
        N=2048,
        ground_truth=topksubset(3, 1),
        model=model,
        optimizer=optimizer,
        epochs=1000,
        seed=42
    )

    processor.run()
    # ... then create visualizer separately and visualize_full()


if __name__ == '__main__':
    example_streaming_training()
