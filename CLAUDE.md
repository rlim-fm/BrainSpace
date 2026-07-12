# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

BrainSpace is the **generic core framework** extracted (2026-07) from the
Tropical-RNN research codebase: training pipeline, declarative configs +
registries, grid-search experiments over a flat content-addressed results store,
statistics, and the visualization system. It is consumed as a **git submodule**
(mounted at `core/`, installed with `pip install -e ./core`) by domain repos —
Tropical-RNN (tropical architectures) and ClusterGAR (k-NN batch sampling + GAR
losses) — each of which keeps its own separate results store. The pre-extraction
standalone code lives on the `legacy` branch.

**This repo must stay domain-agnostic.** No tropical or clustering specifics may
appear here; domains plug in through the extension points below.

## Extension points (how domains plug in)

- **Config subclassing** (`brainspace/config.py`): domains subclass
  `DatasetConfig`/`ModelConfig`/`TrainConfig`/`RunConfig` *under the same class
  names* and install them via `use_config_classes(...)`. `flatten_config`
  iterates `dataclasses.fields()`, so subclass fields flow into config identity
  automatically. `get_config_class('run'|'data'|'model'|'train')` returns the
  installed class — core code must use it instead of the literal classes
  whenever reconstructing configs (e.g. `migrate_registry.rebuild_run_config`).
- **Training-loop wiring**: `TrainConfig.build_epoch_hooks()` (per-epoch
  callables `hook(processor, epoch)`; e.g. tau annealing) and
  `TrainConfig.build_batch_sampler(dataset)` (returns an iterable of index or
  `(indices, weights)` batches, or None for the default contiguous-slice
  gradient-accumulation path). `Processor.from_run_config` passes both through,
  plus `step_granularity` via `getattr` (it is a domain-subclass field —
  deliberately NOT a core field, to preserve hashes).
- **Registries**: `register_arch/ground_truth/criterion/optimizer/scheduler`
  populate the YAML-resolution registries; `register_field_registry` adds
  registries for domain config fields (e.g. tau schedules).
- **Model hooks** (`brainspace/models.py`): `_BaseRNN._build_hidden_proj`
  and the transformer `attn_factory` are RNG-order-preserving construction
  points — domain subclasses inject custom layers with **bit-identical** seeded
  weight init vs the unhooked classes. Do not move these calls within
  `__init__` (comments mark them).
- **Visualization batch hooks**: `Visualization.begin_epoch/record_batch/
  end_epoch` (no-op defaults) are forwarded by `Visualizer` and called from
  `Processor.train_epoch` for batch-granular visualizations.
- **Slug cosmetics**: `experiment.ABBREV_MAP` and
  `experiment.VALUE_SLUG_FORMATTERS` are module-level extensible dicts.

## Hash-preservation rules (critical)

Config ids are `cfg_ + sha1(canonical-json(identity))[:8]` over the complete
resolved field set (see `brainspace/internal/registry.py`). Domain stores with
years of results depend on these staying stable:

- **Never add/rename/remove fields on the core config dataclasses** — that
  re-hashes every stored config in every domain. Domain fields go in subclasses.
- Keep `format_config_value`, class `__name__`s, and value `__repr__`s stable
  (classes render by name only, so moving code between packages is safe).
- Schema changes go through `brainspace/migrate_registry.py` (`--dry-run` first).
- Behavioral changes to `Processor` defaults must keep the gradient-accumulation
  equivalence test green and seeded runs bit-identical.

## Testing

```bash
pip install -r requirements-dev.txt
pytest                  # full suite incl. slow training/render tests
pytest -m "not slow"    # fast core logic (~25s)
```

`python -m brainspace.internal.make_schema_fixture` writes a tiny real store to
`sandbox/schema_fixture/` for store-compat checks. Domain repos run their own
suites against their pinned core; after changing the core, also run the
Tropical-RNN suite (`~/Labs/Tropical-RNN`, pin advanced via submodule pull)
before considering the change safe.

## Repo layout

```
brainspace/            # the package: config, train, models, datasets,
│                      # experiment, visualization, analyze/view/refresh/migrate,
│                      # internal/{registry,stats,util}
tests/                 # generic framework tests (domain tests live in domain repos)
docs/                  # EXPERIMENT_GUIDE.md, CUSTOM_VIZ.md
setup.py               # installs `brainspace` + brainspace-* console scripts
```
