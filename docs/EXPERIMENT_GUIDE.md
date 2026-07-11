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
from config import RunConfig, DatasetConfig, ModelConfig, TrainConfig
from models import LSTM
from datasets import topksubset
import torch.nn as nn
import torch.optim as optim

# Define base configuration (common to all trials)
base_config = RunConfig(
    data=DatasetConfig(ground_truth=topksubset(3), data_dim=(10, 1), N=2048),
    model=ModelConfig(arch=LSTM, hidden_dim=64, num_layers=2),
    train=TrainConfig(epochs=100, criterion=nn.MSELoss(),
                      optimizer_cls=optim.AdamW, lr=1e-3),
)

# Define independent variables (what to vary)
ivs = {
    'epochs': [100, 500],
    'hidden_dim': [32, 64],
}

# Create experiment
experiment = Experiment(
    base_config=base_config,
    ivs=ivs,
    name='epochs_sweep',    # logical name (registry key — not a directory)
    trials=3,               # 3 trials per configuration
    global_seed=42,
    results_root='results'  # the flat results store (default)
)

# Run the full pipeline (cells already in the store are skipped; force=True re-runs)
experiment.run_grid(visualize=True)
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

### 3. Result Organization: the flat results store

All output accumulates in **one flat store** (default `results/`) — there are no
per-experiment subdirectories, and every run ever executed is recorded. Each
distinct configuration gets a folder named by its **global content-hash config
id** (`cfg_<sha1[:8]>` of the config's complete field set, excluding `seed` and
`device`); a lookup table in `registry.json` maps ids to full field sets and
cosmetic descriptive names.

```
results/
├── registry.json   # lookup table: config_id → full fields + name,
│                   # plus one record per experiment (ivs, trials, config ids)
├── index.md        # human-readable index (configs + experiments tables)
├── runs.csv        # APPEND-ONLY log of every run ever executed
├── results.csv     # canonical rows for statistics, keyed (config_id, seed)
├── summary.md      # pooled statistics over the whole store
├── manifests.pkl   # per-experiment manifests (for extension)
└── cfg_a3f9d2c4/   # one folder per distinct config
    ├── config.json                       # full flattened configuration
    ├── config.pkl                        # exact RunConfig (faithful rerun)
    ├── viz_state/                        # per-trial viz data (merge-on-extend)
    ├── cfg_a3f9d2c4_loss_history.png     # combined over trials
    └── cfg_a3f9d2c4_pca_3d_procrustes.mp4 ...
```

Because identity is content-addressed, **adding new IVs or IV values never
disturbs existing results** (new combinations mint new ids), and re-running a
previously-seen config — in *any* experiment — maps back to the same folder and
rows, enabling cell-level caching. Use `--results-root sandbox/my_test` on any
CLI (or `results_root=` in code) for a fully self-contained ad-hoc store.

### 4. Results Summary

#### CSV Outputs

`results.csv` — the canonical table with one row per `(config_id, seed)` cell:
- `config_id`: Global content-hash config identifier (folder name)
- `experiment`: Name of the experiment that (last) produced the row
- `config_idx`, `trial_idx`: Per-experiment identifiers
- `config_name`: Cosmetic human-readable configuration name
- `seed`: Random seed used
- `final_train_loss` / `final_test_loss`: Losses at the final epoch
- `best_test_loss` / `best_test_epoch`: Best test loss and when it occurred
- `epochs`: Number of training epochs
- Full per-field columns (`config_data_*`, `config_model_*`, `config_train_*`)

`runs.csv` — the append-only history: one row per actual training run ever
executed (including failures and `--force` re-runs), with timestamp, experiment,
status, and metrics. It is never rewritten.

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

### `Experiment.run_grid(visualize=True, sampling=5, device='cpu', force=False)`

Executes the full pipeline:
1. Registers each config in the store's lookup table and creates its `cfg_*` directory
2. For each trial round × config:
   - Skips the cell if the store already holds its `(config_id, seed)` result (unless `force=True`)
   - Otherwise trains the model with its recorded seed, collects metrics, and logs the run to `runs.csv`
3. Refreshes `results.csv`/`summary.md` each round and renders visualizations on the first + last round

**Parameters:**
- `visualize`: Whether to generate visualizations (default: True)
- `sampling`: Save visualization data every Nth epoch (default: 5)
- `device`: Device for visualization PCA (default: 'cpu')
- `force`: Re-run cells even if the store already has their results (default: False)

### `Experiment._create_configs()`

Generates all configuration combinations from `ivs` using Cartesian product.

### `Experiment._save_results_summary()`

Creates:
- `results.csv`: Machine-readable results table
- `summary.md`: Human-readable summary report

## Example: Comparing Model Architectures

```python
from experiment import Experiment
from config import RunConfig, DatasetConfig, ModelConfig, TrainConfig
from models import GRU, MHTA
from datasets import topksubset
import torch.nn as nn
import torch.optim as optim

base_config = RunConfig(
    data=DatasetConfig(ground_truth=topksubset(3), data_dim=(10, 1), N=2048),
    model=ModelConfig(arch=GRU, hidden_dim=64, num_layers=2),
    train=TrainConfig(epochs=1000, criterion=nn.MSELoss(),
                      optimizer_cls=optim.AdamW, lr=1e-3),
)

experiment = Experiment(
    base_config=base_config,
    ivs={'arch': [GRU, MHTA]},
    name='model_comparison',
    trials=5,
    global_seed=42,
)

experiment.run_grid()
```

After completion, check the store's `summary.md`/`index.md`, or build a
browsable view: `python view.py --experiment model_comparison --label compare`.

## The Results Store: Registry, Caching, Extension, and Post-hoc Analysis

> The API below is the current `RunConfig`-based interface.

### How experiments run (trial-outer)

`run_grid` runs **trial-outer / config-inner**: it completes trial 0 for every
config, then trial 1 for every config, and so on. Because the per-trial seed is
`global_seed + trial_idx` (constant across configs), every config that shares a
`DatasetConfig` + `cot` for a given trial uses the *same* dataset — so each
distinct dataset is generated **once per round** and reused across configs. This
is bit-identical to the old config-outer order for any given seed; it only
removes redundant dataset regeneration and lets partial results appear early.

Refresh cadence during a run:

- **Every round**: the store's `results.csv` and pooled `summary.md` (including
  the statistical analysis) are rewritten in a background thread pool.
- **First and last round**: the expensive per-config visualizations
  (loss history, 1D/PCA animations, etc.) are (re)rendered — so after round 0
  you already have a full set of figures built from one trial each.

### Config identity and the lookup table

Every configuration is identified by a **global content-hash id**:

```
config_id = 'cfg_' + sha1(canonical-json(identity))[:8]
```

where the identity is the config's **complete, fully-resolved field set** —
every field of `DatasetConfig` / `ModelConfig` / `TrainConfig`, defaults
included, rendered with `registry.format_config_value` — minus `train_seed`
(varies per trial) and `train_device` (execution environment). The lookup table
in `registry.json` maps each id to its full field set, a cosmetic descriptive
name (e.g. `archGRU_h16_cot0.5`), and a creation stamp; `index.md` renders it
for humans.

Consequences of content-addressing:

- **New IVs / IV values just work**: new combinations mint new ids; existing
  ids, folders, and rows are untouched. Nothing is ever renumbered.
- **Cross-experiment identity**: two experiments that construct the same config
  share one folder, one viz state, and one set of result rows.
- **Schema changes need `migrate_registry.py`** (see below): renaming a config
  field or an architecture class, or adding a new config field, changes the
  canonical form.
- `registry.json` is only an index — it can be rebuilt from the config folders
  with `refresh_results.py --rebuild`, and hash-named folders mean concurrent
  jobs never collide on ids.

Experiments themselves are **logical records**, not directories: an experiment
name keys its registry record (IVs, trials, config ids, headline metrics), its
manifest in `manifests.pkl`, and the `experiment` column in result rows.

Read the store programmatically:

```python
from internal import registry

reg = registry.load_registry('results')
for cid, entry in reg['configs'].items():
    print(cid, entry['name'])
for e in reg['experiments']:
    print(e['name'], e['n_configs'], e['trials'], e['metrics']['mean_best_test_loss'])

rows = registry.load_results('results')   # canonical rows, ready for stats.py
```

### Cell-level caching (and `--force`)

Before training a `(config, trial)` cell, `run_grid`/`extend` check the store
for a successful row with the same `(config_id, seed)` — from *any* experiment.
If found, the cell is skipped and the stored row adopted (printed as
`↩ cached`); its visualizations are already in the shared config folder. Failed
rows are never treated as cache hits. Pass `force=True` (CLI `--force`) to
retrain cached cells; the append-only `runs.csv` keeps every actual training
run either way.

### Extending an experiment

An experiment is fully reconstructable from its manifest, so you can add trials
and/or configs later without re-running finished work. Extension runs only the
missing `(config, trial)` cells (the cache skips anything the store already
has), reloads prior trials' visualization state from each config's `viz_state/`
so combined figures show old **and** new trials, and then recomputes
`results.csv`, `summary.md`, and the registry.

**CLI** (`experiment.py`):

```bash
# Grow every existing config up to 10 trials (existing configs run trials [old..10))
python experiment.py --name grid_search --trials 10

# Add architectures (each new config runs the full trial set from scratch)
python experiment.py --name grid_search --add-arch GRU,MHA

# Add arbitrary IV values (repeatable; values coerced to int/float/bool)
python experiment.py --name grid_search --set hidden_dim=32,128 --set cot=0.9

# Combine: add an arch AND grow to 10 trials in one call
python experiment.py --name grid_search --add-arch GRU --trials 10
```

Flags: `--name` (experiment), `--results-root` (default `results`), `--trials N`
(new total, ≥ current), `--add-arch A,B` (resolve class names from `models.py`),
`--set key=v1,v2` (repeatable), `--force`, `--no-viz`, `--sampling`, `--device`.

**Programmatic** (`Experiment.load` + `extend`):

```python
from experiment import Experiment
from models import GRU

exp = Experiment.load('grid_search')             # exact configs, seeds, prior rows
print(exp.trials, len(exp.configs), len(exp.results))

exp.extend(trials=10)                            # +trials for existing configs
exp.extend(add_ivs={'arch': [GRU]})              # +configs (run full trial set)
exp.extend(trials=10, add_ivs={'arch': [GRU]})   # both at once
```

Notes:
- Config indices are stable across extends: new configs are **appended** with new
  `config_idx` (existing rows never shift), because the manifest stores the
  ordered per-config IV assignment rather than relying on product ordering. The
  manifest also pins each `config_idx` to its global `config_id`.
- Seeds are preserved: trial `t` always uses the same seed it would have in the
  original run, so re-running a cell reproduces it exactly.
- Only dict-form `seed_mapping` round-trips through the manifest; `global_seed`
  and the default fallback always do. Custom **callable** seed mappings can't be
  re-pickled, so reload uses the fallback for those.

### Post-hoc statistics on a subset (`analyze.py`)

Recompute statistics over any slice of the store — one experiment, several, or
everything — without re-running training:

```bash
# Everything in the store
python analyze.py

# A single experiment
python analyze.py --experiment grid_search

# Filter: only GRU/MHTA archs with cot in {0.5, 0.7}
python analyze.py --experiment grid_search --arch GRU,MHTA --set cot=0.5,0.7

# Restrict to specific configs / trials
python analyze.py --config-id cfg_a3f9d2c4,cfg_0b7e11ff --trials 0-4
```

Filters:
- `--experiment E1,E2` — keep rows from these experiments.
- `--config-id cfg_x,cfg_y` — keep rows for these global config ids.
- `--arch A,B` — keep rows whose architecture is in the list.
- `--set key=v1,v2` — keep rows whose IV `key` is in the values (IV columns are
  resolved as `config_{data,model,train}_{key}`, e.g. `hidden_dim`, `cot`).
- `--where col=v1,v2` — keep rows by exact results.csv column name.
- `--config-idx 0,2` and `--trials 0-4` — index/range sets.

Output goes to `<results-root>/analysis_summary.md` (loss tables + full LMM /
Friedman+Nemenyi / Spearman analysis over the subset).

**Missing cells → generated `.sh`.** If you request a trial range that hasn't been
run (e.g. `--trials 0-9` when only 0–4 exist), `analyze.py` does not compute
partial statistics. Instead it prints which experiments are short and writes a
runnable `run_missing.sh` that calls `experiment.py` (extend) to fill the gap:

```bash
python analyze.py --experiment grid_search --trials 0-9
# ⚠ needs 10 trials → writes results/run_missing.sh
bash results/run_missing.sh        # extends the experiment(s) to 10 trials
python analyze.py --experiment grid_search --trials 0-9   # now succeeds
```

Pass `--no-emit-missing` to only report the gap without writing the script.

### Browsing a subset (`view.py`)

Config folders are opaque hashes; to *look at* results, assemble a view — a
temporary folder where each selected config appears under its **descriptive
name** with copies of its rendered visualizations (nothing is regenerated), plus
an `analysis_summary.md` recomputed for exactly that subset:

```bash
python view.py --experiment grid_search --label check
python view.py --arch GRU --set cot=0.5 --trials 0-4 --label gru_cot05
```

```
views/2026-07-06_check/
├── analysis_summary.md
├── archGRU_h16_cot0.5/        # = cfg_a3f9d2c4
│   ├── config.json
│   └── *.png / *.mp4 / *.gif  # byte-identical copies from the store
└── archMHTA_h16_cot0.5/
    └── ...
```

`view.py` accepts the same filters as `analyze.py`. `views/` is gitignored and
safe to delete anytime.

### Upgrading stored results in place (`refresh_results.py`)

After a major repo update — a new standard visualization in
`register_defaults()`, a renderer/style change, a summary or config.json schema
tweak — refresh every already-run config without retraining:

```bash
python refresh_results.py                     # whole store
python refresh_results.py --experiment grid_search
python refresh_results.py --config-id cfg_a3f9d2c4
```

For each config it regenerates `config.json` from the exact pickled RunConfig,
re-renders every visualization from the saved `viz_state/` with current code,
and finally rewrites `results.csv`, the pooled `summary.md`, and `index.md`.

A genuinely **new** visualization cannot be rendered from old saved state (its
per-epoch data was never collected). The default pass skips those and prints
which configs are affected; `--rerun-missing` retrains them exactly (from
`config.pkl` + the recorded seeds) so the new visualization's data gets
collected. `--rebuild` reconstructs `registry.json`'s config lookup from the
`cfg_*` folders (disaster recovery, or after merging two stores).

### Schema migrations (`migrate_registry.py`)

Because config ids are content hashes, changes to the config *schema* change
the canonical form and therefore the ids. `migrate_registry.py` rewrites the
whole store consistently — lookup table, `results.csv`/`runs.csv` columns and
values, per-config `config.json`/`config.pkl`, `manifests.pkl` — and re-hashes
ids, renaming the `cfg_*` folders. Always preview with `--dry-run`:

```bash
# A config field was renamed in code (e.g. TrainConfig.cot_sup → cot)
python migrate_registry.py --rename-field train.cot_sup=cot --dry-run
python migrate_registry.py --rename-field train.cot_sup=cot

# A model class was renamed (old pickles are re-read through a rename shim)
python migrate_registry.py --rename-value arch:OldClass=NewClass

# A new config field was added; backfill its value into existing entries
python migrate_registry.py --fill-field model.new_field=default
```

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
