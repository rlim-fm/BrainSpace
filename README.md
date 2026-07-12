# BrainSpace

Core framework for training, visualizing, and running reproducible experiments on
neural networks for functional regression. Domain research repos (e.g.
[Tropical-RNN](https://github.com/rlim-fm/Tropical-RNN),
[ClusterGAR](https://github.com/rlim-fm/ClusterGAR)) consume this repo as a git
submodule pinned to an exact commit and layer their specializations on top; each
domain keeps its **own** flat results store. The pre-extraction standalone codebase
is preserved on the [`legacy`](../../tree/legacy) branch.

## What the `brainspace` package provides

- **`train`** — `Processor` (data → train/test loop → logs/HDF5), `SeedManager`.
  Generic extension points: `epoch_hooks` (per-epoch callables), `batch_sampler`
  (index/`(indices, weights)` batches), `step_granularity` (`'epoch'` grad-accum
  default, `'batch'` per-batch SGD + scheduler), weighted criteria
  `criterion(out, y, weights)`, `StaticNN`/`DynamicNN` forward dispatch.
- **`config`** — declarative `RunConfig`/`DatasetConfig`/`ModelConfig`/`TrainConfig`
  dataclasses, YAML loading, and extensible registries (`register_arch`,
  `register_ground_truth`, `register_criterion`, …). Domains subclass the config
  dataclasses (extra fields flow into config identity automatically) and install
  them with `use_config_classes`; `TrainConfig.build_epoch_hooks()` /
  `build_batch_sampler()` are the domain override points for training-loop wiring.
- **`experiment`** — grid search over IVs into a **flat content-addressed results
  store** (`cfg_<hash8>` folders, global `registry.json`/`results.csv`/`runs.csv`),
  cell-level cross-experiment caching, `Experiment.load(...).extend(...)`,
  extensible descriptive-name slugs (`ABBREV_MAP`, `VALUE_SLUG_FORMATTERS`).
- **`visualization`** — renderer-backend architecture, `Visualization` ABC with
  epoch `update()`/`finalize()` plus batch-level hooks
  (`begin_epoch`/`record_batch`/`end_epoch`), `Visualizer` orchestration,
  detached snapshot-and-render pipeline.
- **`models` / `datasets`** — model zoo (MLP, LSTM/GRU with RNG-order-preserving
  subclassing hooks, transformer with `attn_factory`) and synthetic dataset
  generation (padding/masking, `GroundTruth` protocol).
- **Store CLIs** — `brainspace-analyze`, `brainspace-view`, `brainspace-refresh`,
  `brainspace-migrate` (also runnable as `python -m brainspace.<name>`).

See `docs/EXPERIMENT_GUIDE.md` for the store/extension/analysis workflow and
`docs/CUSTOM_VIZ.md` for custom visualizations.

## Installation

```bash
pip install -e .          # from a domain repo: pip install -e ./core
pytest -m "not slow"      # fast core-logic tests
```

## Using it as a domain core (submodule workflow)

```bash
# In a domain repo
git submodule add https://github.com/rlim-fm/BrainSpace.git core
pip install -e ./core
```

Mount the submodule at `core/` (not `brainspace/` — that would shadow the package).
Day-to-day: edit the core inside `core/` on a real branch
(`cd core && git switch main`), commit + push here, then `git add core && git commit`
in the domain repo to advance the pin. Other domain repos pick the update up
explicitly with `git submodule update --remote core` + a pointer commit.

**Hash-stability contract**: config ids are content hashes over the fully-resolved
config field set. Never add fields to the core config dataclasses (domain fields go
in subclasses); keep field names, class names, and value `__repr__`s stable; schema
changes go through `brainspace-migrate` (always `--dry-run` first).
