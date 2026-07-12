# BrainSpace: Functional Convergence Analysis for Neural Networks

A core framework for training neural networks on functional-regression tasks,
visualizing how they converge to target functions, and running reproducible
grid-search experiments into a content-addressed results store.

BrainSpace is designed to be used two ways:

1. **Standalone** — train models, render convergence animations, run experiments.
2. **As a domain core** — research repos (e.g.
   [Tropical-RNN](https://github.com/rlim-fm/Tropical-RNN),
   [ClusterGAR](https://github.com/rlim-fm/ClusterGAR)) consume this repo as a git
   submodule pinned to an exact commit and plug their specializations in through
   the extension points documented below, each keeping its **own** results store.

The pre-extraction standalone codebase is preserved on the
[`legacy`](../../tree/legacy) branch.

## Overview

- **Training** (`brainspace.train`): `Processor` orchestrates data generation,
  training/evaluation loops, logging, and optional HDF5 checkpoints; `SeedManager`
  gives bit-reproducible runs across numpy/torch/Python.
- **Streaming visualizations** (`brainspace.visualization`): memory-efficient MP4
  animations of loss history, 1D functional convergence, hidden-state PCA
  (anchor and Procrustes modes), and function-space trajectories; a two-method
  `Visualization` ABC for custom ones, plus batch-level hooks.
- **Declarative configs** (`brainspace.config`): type-safe dataclasses
  (`RunConfig`/`DatasetConfig`/`ModelConfig`/`TrainConfig`), YAML loading, and
  extensible name registries.
- **Experiments & results store** (`brainspace.experiment`): grid search over
  independent variables into a flat, content-addressed store with cross-experiment
  caching, extension (`Experiment.load(...).extend(...)`), and faithful trial reruns.
- **Statistics** (`brainspace.internal.stats`): pooled/pairwise analysis over the
  store, in-/out-of-distribution loss views.
- **Model zoo** (`brainspace.models`): `MLP`, `LSTM`/`GRU`, transformer blocks —
  all with subclassing hooks that preserve seeded weight init.

## Installation

```bash
git clone https://github.com/rlim-fm/BrainSpace.git
cd BrainSpace
pip install -e .            # installs the `brainspace` package + store CLIs
pip install -r requirements-dev.txt && pytest -m "not slow"   # optional: fast tests
```

As a submodule core in a domain repo (mount at `core/`, **not** `brainspace/` —
that would shadow the package):

```bash
git submodule add https://github.com/rlim-fm/BrainSpace.git core
pip install -e ./core
```

---

# Standalone use

## Quick start (30 seconds)

```python
from brainspace.datasets import Dataset, cumsum
from brainspace.models import LSTM
from brainspace.train import Processor
from brainspace.visualization import Visualizer

visualizer = Visualizer(name='demo', output_dir='visualizations')
visualizer.register_loss_history()            # loss curves (PNG)
visualizer.register_convergence_1d(axis=0)    # 1D convergence animation (MP4)
visualizer.register_pca_3d()                  # hidden-state PCA, anchor mode (MP4)

dataset = Dataset(cumsum(), x_range=(-8, 8), data_dim=(10, 1), N=2048, seed=42)
model = LSTM(input_dim=1, hidden_dim=64, output_dim=1, num_layers=2)

processor = Processor(dataset=dataset, model=model, epochs=500,
                      visualizer=visualizer, seed=42)
processor.run()
visualizer.finalize()        # renders the MP4s/PNGs
```

## Declarative configs

The same run, as data — this is the form experiments and the results store use:

```python
from brainspace.config import RunConfig, DatasetConfig, ModelConfig, TrainConfig
from brainspace.datasets import cumsum
from brainspace.models import LSTM
from brainspace.train import Processor
import torch.optim as optim

config = RunConfig(
    data=DatasetConfig(ground_truth=cumsum(), x_range=(-8, 8), data_dim=(10, 1), N=2048),
    model=ModelConfig(arch=LSTM, hidden_dim=64, num_layers=2),
    train=TrainConfig(epochs=500, optimizer_cls=optim.AdamW, lr=1e-3, seed=42),
)
processor = Processor.from_run_config(config)
processor.run()
```

Configs also load from YAML (`load_run_config` / `load_experiment_spec`), with
strings resolved through the name registries (`ARCH_REGISTRY`,
`GROUND_TRUTH_REGISTRY`, `CRITERION_REGISTRY`, `OPTIMIZER_REGISTRY`,
`SCHEDULER_REGISTRY`).

## Grid-search experiments and the results store

```python
from brainspace.experiment import Experiment

experiment = Experiment(
    base_config=config,
    ivs={'hidden_dim': [32, 64, 128], 'lr': [1e-3, 1e-2]},   # any config field
    name='hidden_dim_sweep',     # logical name — not a directory
    trials=5,
    global_seed=42,
    results_root='results',
)
experiment.run_grid(visualize=True)     # cached (config, seed) cells are skipped

# Later: add trials or IV values, running only the missing cells
Experiment.load('hidden_dim_sweep', results_root='results').extend(trials=10)
```

Everything accumulates in **one flat store** (default `results/`):

```
results/
├── registry.json   # config_id → full field set + descriptive name; experiment records
├── index.md        # human-readable index
├── runs.csv        # append-only log of every run ever executed
├── results.csv     # canonical rows for statistics, keyed (config_id, seed)
├── summary.md      # pooled statistics
└── cfg_<hash8>/    # one folder per distinct config: config.json/.pkl,
                    # viz_state/, rendered figures
```

`config_id` is a content hash of the fully-resolved config field set, so re-running
the same config anywhere maps to the same folder (cross-experiment caching), and new
IV values just mint new ids without renumbering anything. Point `results_root` at
e.g. `sandbox/my_test` for a fully isolated throwaway store.

Store CLIs (installed as console scripts, or `python -m brainspace.<name>`):

```bash
brainspace-analyze --experiment hidden_dim_sweep --set lr=0.001   # post-hoc stats
brainspace-view    --experiment hidden_dim_sweep --label check    # browsable views/
brainspace-refresh                                                # re-render store after code updates
brainspace-migrate --rename-field train.old=new --dry-run         # schema migrations
```

Because domain repos `pip install -e ./core`, these console scripts land on that
environment's `PATH` — run them from the domain repo's own working directory
(e.g. `~/Labs/Tropical-RNN`, `~/Labs/ClusterGAR`), against that repo's own
`results/`. No need to `cd core/` or `python -m` inside the submodule.

See `docs/EXPERIMENT_GUIDE.md` for the full store/extension/analysis workflow.

## Custom visualizations

Two methods; data collection is separated from rendering:

```python
from brainspace.visualization import Visualization

class MyViz(Visualization):
    def __init__(self):
        super().__init__('my_viz')
        self.data = []

    def update(self, processor, epoch):
        """Called each (sampled) epoch — extract whatever you need."""
        self.data.append(processor.logs['test_loss'][-1])

    def finalize(self, output_dir, prefix):
        """Called after training — write your output files."""
        ...

visualizer.register(MyViz())
```

Prebuilt registrations: `register_loss_history`,
`register_intermediate_loss_history`, `register_sequence_length_loss`,
`register_convergence_1d(axis)`, `register_multi_axis_convergence_1d`,
`register_pca_3d(pca_epoch)` (anchor mode), `register_pca_3d_procrustes`,
`register_function_space_convergence`, or `register_defaults()` for the standard
set. Use `Visualizer(sampling=k)` to record every k-th epoch on long runs.
See `docs/CUSTOM_VIZ.md` for the tutorial.

---

# Hooking a domain in

Domain repos add fields, architectures, losses, samplers, and schedules **without
touching the core** — this keeps every stored `config_id` stable (identity hashes
over config field names/values, and classes render by `__name__` only). The golden
rule: **never add fields to the core config dataclasses**; subclass them instead.

## 1. Config subclasses (domain fields → config identity)

Subclass under the **same class names** and install them; subclass fields flow into
config identity, YAML loading, and results-store columns automatically:

```python
# yourdomain/config.py — importing this module activates the domain
from dataclasses import dataclass, field
from brainspace.config import (
    DatasetConfig as _CoreDatasetConfig, ModelConfig as _CoreModelConfig,
    TrainConfig as _CoreTrainConfig, RunConfig as _CoreRunConfig,
    use_config_classes, register_arch, register_ground_truth,
)

@dataclass
class ModelConfig(_CoreModelConfig):
    arch: type = MyArch
    my_flag: bool = False          # reaches MyArch.__init__ if its signature names it

@dataclass
class TrainConfig(_CoreTrainConfig):
    my_schedule: object = None     # domain training knob

@dataclass
class RunConfig(_CoreRunConfig):
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

use_config_classes(run=RunConfig, model=ModelConfig, train=TrainConfig)
register_arch('MyArch', MyArch)                 # resolvable from YAML / CLIs
register_ground_truth('mytarget', MyTarget)
```

`ModelConfig.build` passes every dataclass field whose name appears in the arch's
`__init__` signature, so new fields plug in with zero core changes.

## 2. Per-epoch hooks (schedules, annealing)

Override `TrainConfig.build_epoch_hooks()` to run callables at the start of every
training epoch — e.g. Tropical-RNN wires tau annealing this way:

```python
@dataclass
class TrainConfig(_CoreTrainConfig):
    tau_schedule: object = None

    def build_epoch_hooks(self):
        hooks = super().build_epoch_hooks()
        if self.tau_schedule is not None:
            hooks.append(lambda proc, epoch: proc.model.set_tau(self.tau_schedule(epoch)))
        return hooks
```

(Direct-constructor use: `Processor(..., epoch_hooks=[my_hook])`.)

## 3. Batch samplers, per-batch stepping, weighted losses

The default training loop accumulates gradients over contiguous slices with one
optimizer step per epoch. Domains can replace batch selection and stepping — e.g.
ClusterGAR's k-NN importance sampling:

```python
@dataclass
class TrainConfig(_CoreTrainConfig):
    sampler: str = None
    step_granularity: str = 'epoch'    # 'batch' → per-batch SGD + scheduler steps

    def build_batch_sampler(self, dataset):
        if self.sampler != 'knn':
            return None                            # core default path
        return KNNBatchSampler(dataset.x_train, seed=self.seed)
```

A sampler is any iterable yielding index batches, or `(indices, probs)` pairs —
when probabilities are present the Processor calls `criterion(out, y, weights)`,
so pair it with a weighted criterion (e.g. inverse-probability / SNIS weighting).

## 4. Model hooks (seeded-init-safe subclassing)

`_BaseRNN._build_hidden_proj(hidden_dim)` and the transformer's
`attn_factory(d_model, n_heads, causal)` are called at fixed, RNG-order-preserving
points during construction — a domain subclass that injects layers there gets
**bit-identical** seeded init for all shared weights vs the unhooked class:

```python
class MyLSTM(_core_models.LSTM):
    def _build_hidden_proj(self, hidden_dim):
        return MyProjection(hidden_dim, hidden_dim)   # or None for the core default
```

## 5. Batch-level visualization hooks

`Visualization` exposes optional no-op hooks fired by the Processor for every
training batch — for sampler diagnostics, batch-composition heatmaps, etc.:

```python
class BatchViz(Visualization):
    def begin_epoch(self, processor, epoch): ...
    def record_batch(self, processor, epoch, batch): ...
        # batch = {'indices': slice | LongTensor, 'weights': Tensor | None, 'loss': float}
    def end_epoch(self, processor, epoch): ...
    def update(self, processor, epoch): pass
    def finalize(self, output_dir, prefix): ...
```

## 6. Descriptive-name slugs

Config folders get cosmetic names like `archLSTM_h64_lr0.01`. Register
abbreviations and value formatters for your fields:

```python
import brainspace.experiment as exp
exp.ABBREV_MAP.setdefault('my_flag', 'mf')
exp.VALUE_SLUG_FORMATTERS.setdefault('my_flag', lambda abbrev, v: 'on' if v else 'off')
```

## Submodule update workflow

Day-to-day: edit the core inside a domain repo's `core/` submodule on a real branch
(`cd core && git switch main`), commit + push here, then `git add core && git commit`
in the domain repo to advance the pin. Other domain repos pick the update up
explicitly with `git submodule update --remote core` + a pointer commit — nothing
moves under them silently. Fresh clones need `git clone --recurse-submodules`.

**Deploying to another machine (e.g. HPC)**: after every `git pull` in the domain
repo, sync the submodule checkout to the new pin — the pointer moves with the pull,
the checkout does not:

```bash
git pull
git submodule update --init core   # check out the pinned core commit
pip install -e ./core              # ONE TIME per environment (add --no-deps to
                                   # leave cluster-provided torch/numpy alone)
```

Because the install is editable, subsequent pulls only need the
`git submodule update --init core` step — no reinstall.

**Hash-stability contract** (domain stores with accumulated results depend on it):
never add fields to the core config dataclasses; keep field names, class names, and
value `__repr__`s stable (criterion/scheduler reprs feed the hash); route schema
changes through `brainspace-migrate` with `--dry-run` first.

---

## Testing

```bash
pytest                  # full suite, incl. slow training + FFmpeg rendering
pytest -m "not slow"    # fast core-logic tests (~25s)
python -m brainspace.internal.make_schema_fixture   # tiny real store for compat checks
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "ffmpeg not found" | `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux) |
| Out of memory during viz | `Visualizer(sampling=10)` to reduce frames |
| `brainspace` shadowed in a domain repo | Mount the submodule at `core/`, not `brainspace/` |
| Store CLIs on another store | All take `--results-root <path>` (default `results`) |

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
