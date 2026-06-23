"""
Example demonstrating the new streaming visualization system.

This shows how to:
1. Create a Visualizer and register desired visualizations
2. Pass it to Processor for training
3. Processor logs only needed data during training
4. Visualizer generates animations automatically after training completes
5. Memory efficient - data is not stored between epochs for registered visualizations
"""

from models import *
from datasets import topksubset
from train import Processor
from visualization import *
import torch.optim as optim
import torch.nn as nn

def main():

    # Create visualizer with desired visualizations
    visualizer = Visualizer(name='MHTA')
    visualizer.register_loss_history()  # Loss plot
    visualizer.register_convergence_1d(axis=0)  # 1D convergence
    visualizer.register_pca_3d()  # 3D PCA (anchor mode)
    visualizer.register_pca_3d_procrustes()  # 3D PCA (Procrustes mode)
    visualizer.register(FunctionSpaceConvergence())

    model = SimpleTransformerModel(input_dim=1, dropout=0.25, tropical=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.75)

    processor = Processor(
        x_range=(-8, 8),
        data_dim=(10, 1),
        N=2048,
        ground_truth=topksubset(3, 1),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=5000,
        criterion=nn.MSELoss(reduction='mean'),
        visualizer=visualizer,
        seed=42,
        device='mps'
    )

    print("Training with streaming visualizations...")
    processor.run()
    processor.print_summary()

    print("\n✓ Training and visualization complete!")
    print("Output files will be saved to 'visualizations/topk-sum/'")

if __name__ == '__main__':
    main()
