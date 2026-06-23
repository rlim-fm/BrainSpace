# BrainSpace: Functional Convergence Analysis for Neural Networks

A framework for visualization how neural networks converge to target functions during training through OOD visualization and analysis of functional space dynamics.

## Overview

This module provides tools for:
- **Training neural networks** for functional regression
- **Streaming visualizations** of network convergence (memory-efficient, MP4 output)
- **Analysis of hidden state evolution** in PCA space with two modes:
  - **Anchor Mode**: Fixed PCA basis from single epoch for consistent view 
  
  - **Procrustes Mode**: Per-epoch PCA with alignment for smooth evolution
- **Domain-aware visualization** distinguishing in-domain vs out-of-domain predictions
- **Custom visualization framework** - write your own by inheriting from `Visualization`

## Prebuilt Visualizations
- **Loss History**: Training and test loss curves
![MLP_top3sum_loss_history.png](visualization_samples/MLP_top3sum_loss_history.png)
- **1D Convergence**: Animated function convergence along a single input axis
![MLP_top3sum_1d_convergence.mp4](visualization_samples/MLP_top3sum_1d_convergence.mp4)
- **3D PCA Convergence**: Hidden states evolution in PCA space (anchor mode)
![MLP_top3sum_pca_3d_anchor.mp4](visualization_samples/MLP_top3sum_pca_3d_anchor.mp4)
- **3D PCA Procrustes**: Hidden states evolution with per-epoch PCA alignment
![MLP_top3sum_pca_3d_procrustes.mp4](visualization_samples/MLP_top3sum_pca_3d_procrustes.mp4)
- **Function Space Convergence**: Visualize the convergence of the network in the function space as a 3D projection
![MLP_top3sum_function_space.png](visualization_samples/MLP_top3sum_function_space.png)
## Quick Start (30 seconds)

```python
from visualization import Visualizer
from train import Processor
from models import MLP
from datasets import topksubset
import torch.optim as optim

# Create visualizer with desired visualizations
visualizer = Visualizer(name='model')
visualizer.register_loss_history()            # Loss plot
visualizer.register_convergence_1d(axis=0)    # 1D convergence
visualizer.register_pca_3d()                  # 3D PCA (anchor mode)
visualizer.register_pca_3d_procrustes()       # 3D PCA (Procrustes mode)

# Train with streaming visualizations (memory efficient!)
processor = Processor(
    model=MLP(input_dim=10),
    epochs=5000,
    visualizer=visualizer
)
processor.run()  # Automatically generates MP4 animations!
```

## Key Features

### 🎬 Streaming Visualizations
- **Memory efficient**: 27x less RAM during training (2.7GB → 100MB)
- **MP4 output**: Professional-quality animations, 5-10x smaller than GIF
- **Dynamic data extraction**: Visualizer pulls data from processor as needed
- **Easy to extend**: Write custom visualizations with just 2 methods

### 🔧 Simple API for Custom Visualizations

```python
from visualization import Visualization

class MyCustomViz(Visualization):
    def __init__(self):
        super().__init__('my_viz')
        self.data = []
    
    def update(self, processor, epoch):
        """Called each epoch. Extract whatever you need from processor."""
        self.data.append(processor.logs['f_test'][-1])
    
    def finalize(self, output_dir, prefix):
        """Called after training. Create your output files."""
        # Do something with self.data...
        
# Use it!
visualizer.register(MyCustomViz())
```

## Installation

```bash
git clone <repo>
cd BrainSpace
pip install -r requirements.txt
```

## Usage Examples

### Basic Training

```python
processor = Processor(
    x_range=(-8, 8),
    data_dim=(10,),
    N=2048,
    model=MLP(input_dim=10),
    epochs=5000
)
processor.run()
processor.save('output.h5')
```

### With Visualizations

```python
visualizer = Visualizer(sampling=10)  # Sample every 10th epoch
visualizer.register_loss_history()
visualizer.register_convergence_1d(axis=0)
visualizer.register_pca_3d()

processor = Processor(..., visualizer=visualizer)
processor.run()
# Animations automatically saved to visualizations/topk-sum/
```

### Custom Ground Truth Function

```python
from datasets import topksubset

# Built-in function: top-k subset sum
ground_truth = topksubset(k=3, d=1)

processor = Processor(
    ground_truth=ground_truth,
    model=MLP(input_dim=10),
    ...
)
```

## API Quick Reference

### Processor

```python
processor = Processor(
    x_range=(-8, 8),          # Input domain
    data_dim=(10,),           # Input shape
    N=2048,                   # Training samples
    ground_truth=...,         # Target function
    model=...,                # Neural network
    epochs=5000,              # Training epochs
    visualizer=...            # Optional visualizer
)
```

### Visualizer

```python
visualizer = Visualizer(sampling=1)

# Register visualizations
visualizer.register_loss_history()
visualizer.register_convergence_1d(axis=0)
visualizer.register_pca_3d(pca_epoch=-1)      # Anchor mode
visualizer.register_pca_3d_procrustes()       # Procrustes mode
visualizer.register_function_space()

# Or create custom ones
class MyViz(Visualization):
    def update(self, processor, epoch): pass
    def finalize(self, output_dir, prefix): pass
visualizer.register(MyViz())
```

## Output Files

```
visualizations/topk-sum/
├── {prefix}_loss_history.png              # Loss plot
├── {prefix}_1d_convergence.mp4            # 1D convergence animation
├── {prefix}_pca_3d_anchor.mp4             # 3D PCA (fixed basis)
├── {prefix}_pca_3d_procrustes.mp4         # 3D PCA (smooth evolution)
└── {prefix}_function_space.png            # Function space plot
```

## Documentation

- **Quick Start**: This README
- **API Reference**: `docs/API_REFERENCE.md`
- **Experiment Guide**: `docs/EXPERIMENT_GUIDE.md`
- **Advanced Usage**: `docs/QUICK_REFERENCE.md`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "ffmpeg not found" | `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux) |
| Out of memory | Use `visualizer = Visualizer(sampling=10)` to reduce frames |
| No visualizations | Ensure visualizer registered before Processor creation |

## Citation

```bibtex
@software{lim2026brainspace,
  title={BrainSpace: A Python Package for Functional Convergence Analysis of Neural Networks},
  author={Lim, Richard},
  year={2026}
}
```

## License

See LICENSE file.
