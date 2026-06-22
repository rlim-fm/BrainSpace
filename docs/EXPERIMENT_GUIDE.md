# Experiment Pipeline Guide

## Overview

The enhanced `experiment.py` now provides a complete training/visualization pipeline that:

1. **Trains models** based on a grid of experimental configurations
2. **Runs visualizations** for each trained model
3. **Collects results** and saves them in an organized manner
4. **Generates summaries** of experimental results (CSV and Markdown)

## Quick Start

### Basic Usage

```python
from experiment import Experiment
from models import MLP
from datasets import topksubset
import torch.nn as nn
import torch.optim as optim

# Define base configuration (common to all trials)
base_config = {
    'x_range': (-8, 8),
    'data_dim': (10, 1),
    'N': 2048,
    'ground_truth': topksubset(3, 1),
    'model': MLP(input_dim=1, hidden_sizes=(64, 64)),
    'criterion': nn.MSELoss(),
    'optimizer': optim.AdamW(MLP(input_dim=1, hidden_sizes=(64, 64)).parameters(), lr=0.001),
}

# Define independent variables (what to vary)
ivs = {
    'epochs': [100, 500],      # Test with 100 and 500 epochs
    'scheduler': [None, ...],  # Or any other parameters
}

# Create experiment
experiment = Experiment(
    base_config=base_config,
    ivs=ivs,
    trials=3,              # 3 trials per configuration
    output_root='results'  # Results directory
)

# Run the full pipeline
experiment.run_grid(visualize=True, save_visualizations=True)
```

## Features

### 1. Model Training

- Trains models for each configuration and seed combination
- Automatically sets seeds for reproducibility (seed = trial_idx)
- Captures final and best test losses during training

### 2. Visualization

For each trained model, the pipeline generates:
- **Loss History** (`*_loss_history.png`): Training and test loss curves
- **1D Convergence** (`*_1d_convergence.gif`): Animated function convergence along x₀ axis
- **3D PCA Convergence** (`*_pca_3d_convergence.gif`): Hidden states evolution (fixed PCA anchor)
- **3D PCA Procrustes** (`*_pca_3d_convergence_procrustes.gif`): Hidden states with per-epoch PCA alignment

### 3. Result Organization

All outputs are organized hierarchically:

```
results/
├── results.csv                 # Detailed results table
├── summary.md                  # Summary report
├── config_0/
│   ├── trial_0/
│   │   ├── training_data.h5    # Training logs and metadata
│   │   ├── model.pt            # Model weights
│   │   ├── config.json         # Configuration for this trial
│   │   └── visualizations/
│   │       ├── trial_0_loss_history.png
│   │       ├── trial_0_1d_convergence.gif
│   │       ├── trial_0_pca_3d_convergence.gif
│   │       └── trial_0_pca_3d_convergence_procrustes.gif
│   └── trial_1/
│       └── ...
└── config_1/
    └── ...
```

### 4. Results Summary

#### CSV Output (`results.csv`)

A detailed table with one row per trial containing:
- `config_idx`, `trial_idx`: Configuration and trial identifiers
- `config_name`: Human-readable configuration name
- `seed`: Random seed used
- `final_train_loss`: Training loss at final epoch
- `final_test_loss`: Test loss at final epoch
- `best_test_loss`: Best test loss achieved during training
- `best_test_epoch`: Epoch where best test loss was achieved
- `epochs`: Number of training epochs
- Per-parameter columns (e.g., `config_epochs`, `config_scheduler`, etc.)

#### Markdown Summary (`summary.md`)

A formatted report including:
- **Overview**: Number of configurations, trials, and total runs
- **Loss Statistics**: Min/max/mean/std of final test losses
- **Results by Configuration**: Detailed tables for each configuration
- **Output Organization**: Directory structure reference

## Advanced Usage

### Custom Configuration Names

Configuration names are automatically generated from independent variables:
```
"epochs=100", "epochs=500", etc.
```

For custom naming, override `_config_to_name()` method.

### Conditional Visualization

Run training without visualization:
```python
experiment.run_grid(visualize=False)
```

Or run visualization only on specific configs:
```python
experiment.run_grid(visualize=True, save_visualizations=True)
```

### Multiple Independent Variables

```python
ivs = {
    'epochs': [100, 500],
    'model': [MLP(...), Transformer(...)],
    'ground_truth': [topksubset(3, 1), topksubset(5, 2)],
}
# This will create 2 × 2 × 2 = 8 configurations
```

## Integration with Existing Code

### Backward Compatibility

The enhancement maintains full backward compatibility:
- **train.py**: No changes required. `Processor` works exactly as before.
- **visualization.py**: No changes required. `Visualizer` works exactly as before.
- Both can still be used independently for single experiments.

### Accessing Results Programmatically

```python
experiment = Experiment(...)
experiment.run_grid()

# Access results
for result in experiment.results:
    print(f"Config {result['config_idx']}, Trial {result['trial_idx']}")
    print(f"Final Test Loss: {result['final_test_loss']:.6f}")
    print(f"Best Test Loss: {result['best_test_loss']:.6f}")
```

## Key Methods

### `Experiment.run_grid(visualize=True, save_visualizations=True)`

Executes the full pipeline:
1. Creates configuration directories
2. For each config × trial:
   - Trains model with unique seed
   - Collects metrics
   - Optionally runs visualizations
   - Saves all outputs
3. Generates summary files

**Parameters:**
- `visualize`: Whether to generate visualizations (default: True)
- `save_visualizations`: Whether to save visualizations to disk (default: True)

### `Experiment._create_configs()`

Generates all configuration combinations from `ivs` using Cartesian product.

### `Experiment._save_results_summary()`

Creates:
- `results.csv`: Machine-readable results table
- `summary.md`: Human-readable summary report

## Example: Comparing Model Architectures

```python
from models import MLP, SimpleTransformerModel
import torch.nn as nn
import torch.optim as optim

base_config = {
    'x_range': (-8, 8),
    'data_dim': (10, 1),
    'N': 2048,
    'ground_truth': topksubset(3, 1),
    'criterion': nn.MSELoss(),
    'epochs': 1000,
}

mlp_model = MLP(input_dim=1, hidden_sizes=(64, 64))
mlp_optimizer = optim.AdamW(mlp_model.parameters(), lr=0.001)

transformer_model = SimpleTransformerModel(input_dim=1)
transformer_optimizer = optim.AdamW(transformer_model.parameters(), lr=0.001)

experiment = Experiment(
    base_config=base_config,
    ivs={
        'model': [mlp_model, transformer_model],
    },
    trials=5,
    output_root='model_comparison'
)

experiment.run_grid()
```

After completion, check `model_comparison/summary.md` and `model_comparison/results.csv` to compare architectures!

## Troubleshooting

### Out of Memory

If encounters GPU/CPU memory issues:
1. Reduce `N` (number of samples)
2. Reduce `data_dim` complexity
3. Reduce batch size (if using in future implementations)

### Visualization Failures

If visualization fails but training succeeds, check:
1. `hidden_states` dimension compatibility
2. PCA convergence properties
3. matplotlib/PIL availability

The pipeline gracefully handles visualization errors and continues.

### Missing Dependencies

Ensure all dependencies are installed:
```bash
pip install torch numpy scipy scikit-learn matplotlib h5py
```

## Performance Tips

1. **Parallelize**: Consider using `joblib.Parallel` for multiple configs (requires modifications)
2. **Reduce sampling**: Decrease visualization sampling rate for large datasets
3. **Batch processing**: Group related configs to share data preparation
