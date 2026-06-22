# Quick Reference: Experiment Pipeline

## 🚀 Quick Start (30 seconds)

```python
from experiment import Experiment
from models import SimpleTransformerModel
from datasets import topksubset
import torch.optim as optim
import torch.nn as nn

# Setup
experiment = Experiment(
    base_config={
        'x_range': (-8, 8),
        'data_dim': (10, 1),
        'N': 2048,
        'ground_truth': topksubset(3, 1),
        'model': SimpleTransformerModel(input_dim=1),
        'criterion': nn.MSELoss(),
        'optimizer': optim.AdamW(SimpleTransformerModel(input_dim=1).parameters(), lr=0.001),
    },
    ivs={'epochs': [100, 500, 1000]},
    trials=3,
    output_root='results'
)

# Run
experiment.run_grid(visualize=True, save_visualizations=True)

# Results in: results/results.csv and results/summary.md
```

## 📁 Output Structure

```
results/
├── results.csv              # Detailed metrics table
├── summary.md               # Formatted report
├── config_0/trial_0/
│   ├── training_data.h5     # Logs
│   ├── model.pt             # Weights
│   ├── config.json          # Configuration
│   └── visualizations/      # 4 visualization files
└── ... (more configs/trials)
```

## 📊 Results CSV Columns

- `config_idx`, `trial_idx`: IDs
- `config_name`: Human-readable name
- `seed`: Random seed
- `final_train_loss`: Final training loss
- `final_test_loss`: **Final test loss (MAIN METRIC)**
- `best_test_loss`: Best test loss achieved
- `best_test_epoch`: When best loss occurred
- `epochs`: Training epochs
- `config_*`: All config parameters

## 📈 Summary Markdown Contents

1. **Overview**: Configs, trials, runs count
2. **Loss Statistics**: Min/max/mean/std of final test losses
3. **Per-Config Tables**: Detailed results for each configuration
4. **Directory Structure**: Reference diagram

## ⚙️ Configuration Examples

### Test Different Epochs
```python
ivs={'epochs': [100, 500, 1000]}
```

### Test Different Models
```python
from models import MLP, SimpleTransformerModel

ivs={
    'model': [
        MLP(input_dim=1, hidden_sizes=(64, 64)),
        SimpleTransformerModel(input_dim=1)
    ]
}
```

### Test Multiple Parameters
```python
ivs={
    'epochs': [500, 1000],
    'N': [1024, 2048],
    'x_range': [(-4, 4), (-8, 8)]
}
# Creates 2 × 2 × 2 = 8 configurations
```

## 🔧 Training Only (No Visualization)

```python
experiment.run_grid(visualize=False)
```
Saves: CSV, summary, training data, models only
Time: ~70% faster

## 📊 Accessing Results Programmatically

```python
for result in experiment.results:
    config_id = result['config_idx']
    trial_id = result['trial_idx']
    test_loss = result['final_test_loss']
    print(f"Config {config_id}, Trial {trial_id}: {test_loss:.6f}")
```

## 📋 Visualizations Generated (per trial)

1. **loss_history.png**: Training/test loss curves
2. **1d_convergence.gif**: Function convergence animation
3. **pca_3d_convergence.gif**: Hidden state evolution (fixed PCA)
4. **pca_3d_convergence_procrustes.gif**: Hidden state evolution (aligned PCA)

## ⏱️ Typical Runtime

- Training only: ~5 min for 3 configs × 3 trials × 100 epochs
- + Visualizations: ~10-15 min (adds 1-2 min per trial)

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `N`, reduce `epochs`, or disable visualizations |
| Slow execution | Set `visualize=False` or reduce `trials` |
| Shape mismatch error | Ensure `model` input_dim matches your data_dim |
| Missing visualizations | Check that hidden states have correct dimensions |

## 📚 Documentation Files

- **EXPERIMENT_GUIDE.md**: Full documentation (380+ lines)
- **IMPLEMENTATION_SUMMARY.md**: Technical details (150+ lines)
- **test_experiment.py**: Working example and test suite
- **experiment.py**: Source code (344 lines)

## ✅ Verification

Run the test suite:
```bash
python test_experiment.py
```

Expected output: `✓ ALL TESTS PASSED!`

## 🔄 Reproducibility

Each trial:
- Seed = trial_idx (e.g., trial_0 uses seed=0)
- config.json saved with all parameters
- Model weights saved to model.pt
- Training logs saved to training_data.h5

Reproduce any trial:
```python
import json
with open('results/config_0/trial_0/config.json') as f:
    config = json.load(f)
processor = Processor(**config)
processor.run()
```

## 💡 Pro Tips

1. Start with `trials=2` and `visualize=False` for testing
2. Use `results.csv` for spreadsheet analysis
3. Compare configs by looking at mean/std in summary.md
4. Organize experiments: `output_root='exp_2024_06_18'`
5. Version your configs in a separate JSON file

## 🎯 Next Steps

1. Customize `base_config` for your experiment
2. Define `ivs` (what you want to test)
3. Adjust `trials` (more trials = more confidence)
4. Run: `experiment.run_grid()`
5. Analyze results in CSV/Markdown

---

**Need help?** See EXPERIMENT_GUIDE.md for detailed documentation.
