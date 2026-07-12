from .train import Processor
from .visualization import Visualizer
from .config import RunConfig, ARCH_REGISTRY
from .internal import stats
from .internal import registry
from .internal.registry import format_config_value, format_duration
from itertools import product
from concurrent.futures import ThreadPoolExecutor
import argparse
import os
import gc
import json
import csv
import pickle
import shutil
import sys
import threading
import time
from datetime import datetime
import numpy as np
import torch
from dataclasses import fields, replace
from typing import Callable, List, Optional

# Backwards-compatible alias (the canonical implementation lives in registry.py
# because config formatting defines the content-hash config identity).
_format_config_value = format_config_value

# ---------------------------------------------------------------------------
# Descriptive-name (slug) formatting — purely cosmetic, used for the lookup
# table's descriptive names and view.py folder names. Domain packages extend
# these: add abbreviations to ABBREV_MAP, and per-field formatters
# ``fmt(abbrev, value) -> str`` to VALUE_SLUG_FORMATTERS (return '' to omit).
# ---------------------------------------------------------------------------

ABBREV_MAP = {
    'hidden_dim': 'h',
    'num_layers': 'L',
    'optimizer_type': 'opt',
    'architecture': 'arch',
    'rnn_type': 'rnn',
    'lr': 'lr',
    'dropout': 'drop',
    'epochs': 'ep',
}

VALUE_SLUG_FORMATTERS = {}


class _ThreadRoutedStdout:
    """Process-wide sys.stdout replacement that lets specific *threads*
    redirect their own prints to an alternate stream, without touching what
    other threads see. Used so Experiment's detached-render worker thread
    (which calls Visualizer.finalize(), full of its own ad hoc prints) can be
    routed to render.log while the main training thread keeps printing to the
    real terminal — the two would otherwise interleave unreadably since
    rendering runs concurrently with training."""

    def __init__(self, default_stream):
        self._default = default_stream
        self._local = threading.local()

    def route_to(self, stream):
        self._local.target = stream

    def reset(self):
        self._local.target = None

    def _target(self):
        return getattr(self._local, 'target', None) or self._default

    def write(self, s):
        return self._target().write(s)

    def flush(self):
        return self._target().flush()

    def isatty(self):
        t = self._target()
        return t.isatty() if hasattr(t, 'isatty') else False


def _install_stdout_router():
    """Idempotently wrap sys.stdout in a _ThreadRoutedStdout (once per process)."""
    if not isinstance(sys.stdout, _ThreadRoutedStdout):
        sys.stdout = _ThreadRoutedStdout(sys.stdout)
    return sys.stdout


# ============================================================================
# Reusable results-summary writers (module-level so analyze.py can share them)
# ============================================================================

def _native_scalar(s):
    """Coerce a CSV string back to the native type format_config_value emits
    for primitives (int/float/bool/None), leaving everything else a string.

    Ensures rows reloaded from results.csv carry the *same* Python types as
    freshly-computed rows, so mixing them (on extend / global merge) doesn't
    split otherwise equal IV values into distinct groups in the statistics.
    """
    if not isinstance(s, str):
        return s
    if s in ('', 'None'):
        return None
    if s == 'True':
        return True
    if s == 'False':
        return False
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def _coerce_loaded_result(row):
    """Normalize a reloaded result row so config_* fields match fresh-row types."""
    return {k: (_native_scalar(v) if k.startswith('config_') and k not in ('config_idx', 'config_id') else v)
            for k, v in row.items()}


def write_results_csv(results, csv_file):
    """Atomically write result-entry dicts to a CSV, unioning keys across rows."""
    os.makedirs(os.path.dirname(csv_file) or '.', exist_ok=True)
    tmp = csv_file + '.tmp'
    with open(tmp, 'w', newline='') as f:
        # Union of all result keys (success rows lack 'error'; failed rows may
        # lack loss fields) — dict.fromkeys preserves insertion order.
        all_fields = list(dict.fromkeys(k for r in results for k in r.keys()))
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    os.replace(tmp, csv_file)


def _best_config_key(results):
    """The config_id (or config_idx) with the lowest mean best_test_loss among
    configs with at least one successful row, or None if none succeeded."""
    by_config = {}
    for r in results:
        if r.get('best_test_loss') is None:
            continue
        key = r.get('config_id') or r['config_idx']
        by_config.setdefault(key, []).append(r['best_test_loss'])
    if not by_config:
        return None
    return min(by_config, key=lambda k: np.mean(by_config[k]))


def loss_statistics_md(results) -> str:
    """Aggregate final/best test-loss statistics as a markdown section."""
    out = ["## Loss Statistics\n\n"]
    best_key = _best_config_key(results)
    if best_key is not None:
        best_rows = [r for r in results if (r.get('config_id') or r.get('config_idx')) == best_key]
        best_name = best_rows[0].get('config_name', str(best_key))
        best_losses = [r['best_test_loss'] for r in best_rows if r.get('best_test_loss') is not None]
        out.append(f"🏆 **Best performing config:** `{best_key}` ({best_name}) — "
                   f"mean best test loss = {np.mean(best_losses):.6f}\n\n")
    out.append("| Metric | Value |\n")
    out.append("|--------|-------|\n")
    test_losses = [r['final_test_loss'] for r in results if r.get('final_test_loss') is not None]
    if test_losses:
        out.append(f"| Min Final Test Loss | {min(test_losses):.6f} |\n")
        out.append(f"| Max Final Test Loss | {max(test_losses):.6f} |\n")
        out.append(f"| Mean Final Test Loss | {np.mean(test_losses):.6f} |\n")
        out.append(f"| Std Final Test Loss | {np.std(test_losses):.6f} |\n")
        best_losses = [r['best_test_loss'] for r in results if r.get('best_test_loss') is not None]
        if best_losses:
            out.append(f"| Mean Best Test Loss | {np.mean(best_losses):.6f} |\n")
    out.append("\n")
    return "".join(out)


def config_tables_md(results) -> str:
    """Per-config results tables as a markdown section (grouped by global
    config_id when present, falling back to per-experiment config_idx)."""
    out = ["## Results by Configuration\n\n"]
    configs = {}
    for result in results:
        key = result.get('config_id') or result['config_idx']
        configs.setdefault(key, []).append(result)
    best_key = _best_config_key(results)

    for cfg_key in sorted(configs.keys(), key=str):
        results_for_config = configs[cfg_key]
        config_name = results_for_config[0].get('config_name', str(cfg_key))
        header = f"### Configuration `{cfg_key}`: {config_name}"
        if cfg_key == best_key:
            header = f"### 🏆 Configuration `{cfg_key}`: {config_name} — best performing"
        out.append(f"{header}\n\n")
        out.append("| Trial | Seed | Final Train Loss | Final Test Loss | Best Test Loss | Best Epoch | Status |\n")
        out.append("|-------|------|-----------------|-----------------|----------------|------------|--------|\n")
        for result in sorted(results_for_config,
                             key=lambda r: (r.get('trial_idx') is None, r.get('trial_idx'))):
            trial_idx = result.get('trial_idx')
            status = "✓" if result.get('final_test_loss') is not None else "✗"
            if result.get('final_test_loss') is not None:
                out.append(f"| {trial_idx} | {result.get('seed', '-')} | {result['final_train_loss']:.6f} | "
                           f"{result['final_test_loss']:.6f} | "
                           f"{result['best_test_loss']:.6f} | {result['best_test_epoch']} | {status} |\n")
            else:
                out.append(f"| {trial_idx} | {result.get('seed', '-')} | - | - | - | - | {status} |\n")
        test_losses = [r['final_test_loss'] for r in results_for_config if r.get('final_test_loss') is not None]
        if test_losses:
            out.append(f"\n**Summary:** Mean = {np.mean(test_losses):.6f} ± {np.std(test_losses):.6f}\n\n")
    return "".join(out)


def statistical_tests_md(results, ivs, ordinal_ivs, violin_pngs=None, pvalue_pngs=None) -> str:
    """LMM / Friedman+Nemenyi / Spearman analysis as a markdown section,
    optionally prefixed with embedded images: one pairwise IV violin-hue
    comparison per IV pair (violin_pngs) and one pairwise effect-size heatmap
    per non-ordinal IV (pvalue_pngs) -- each list of (label, relpath),
    already split into per-stratum panels within a single file per entry."""
    body = stats.run_statistical_tests(results, ivs, ordinal_ivs)
    images = "".join(f"![{label} violin]({path})\n\n" for label, path in (violin_pngs or []))
    images += "".join(f"![{label} effect size]({path})\n\n" for label, path in (pvalue_pngs or []))
    if not images:
        return body
    heading = "## Statistical Analysis\n\n"
    if body.startswith(heading):
        return heading + images + body[len(heading):]
    return images + body


def experiment_config_md(results, ivs, ordinal_ivs) -> str:
    """Summarize what the experiment actually swept: config fields that are
    fixed across every result row vs. ones that vary (the realized IVs),
    derived purely from the ``config_*`` columns already on each row so this
    works for both a live Experiment and an arbitrary post-hoc subset."""
    out = ["## Experiment Configuration\n\n"]
    if not results:
        out.append("_No results to summarize._\n\n")
        return "".join(out)

    sample_row = results[0]
    ordinal_cols = {stats._iv_column(k, sample_row) for k in ordinal_ivs}
    ordinal_cols.discard(None)

    config_keys = sorted(k for k in sample_row if k.startswith('config_')
                        and k not in ('config_id', 'config_idx', 'config_name'))
    fixed, varied = [], []
    for key in config_keys:
        values = {str(r.get(key)) for r in results if key in r}
        label = key[len('config_'):].replace('_', '.', 1)
        if len(values) <= 1:
            fixed.append((label, next(iter(values), '')))
        else:
            varied.append((key, label, sorted(values)))

    if ivs:
        out.append(f"- **Declared IVs:** {', '.join(sorted(ivs.keys()))}\n")
    if ordinal_ivs:
        out.append(f"- **Ordinal IVs:** {', '.join(sorted(ordinal_ivs))}\n")
    out.append("\n")

    if varied:
        out.append("**Varied parameters:**\n\n")
        out.append("| Parameter | Values |\n|-----------|--------|\n")
        for key, label, values in varied:
            tag = " _(ordinal)_" if key in ordinal_cols else ""
            out.append(f"| {label}{tag} | {', '.join(values)} |\n")
        out.append("\n")

    if fixed:
        out.append("**Fixed parameters:**\n\n")
        out.append("| Parameter | Value |\n|-----------|-------|\n")
        for label, value in fixed:
            out.append(f"| {label} | {value} |\n")
        out.append("\n")

    return "".join(out)


def _directory_structure_md(results_root) -> str:
    out = ["## Output Organization\n\n", f"```\n{results_root}/\n"]
    out.append("├── registry.json               # Config lookup table + experiment records\n")
    out.append("├── index.md                    # Human-readable index of configs/experiments\n")
    out.append("├── runs.csv                    # Append-only log of every run ever executed\n")
    out.append("├── results.csv                 # Canonical result rows (keyed config_id, seed)\n")
    out.append("├── summary.md                  # This file (pooled over the whole store)\n")
    out.append("├── manifests.pkl               # Per-experiment manifests (for extension)\n")
    out.append("└── cfg_<hash>/                 # One directory per distinct config\n")
    out.append("    ├── config.json             # Full flattened configuration\n")
    out.append("    ├── config.pkl              # Exact RunConfig (for faithful rerun)\n")
    out.append("    ├── viz_state/              # Committed per-trial visualization data\n")
    out.append("    └── <config_id>_*.png/.gif  # Rendered visualizations\n")
    out.append("```\n\n")
    return "".join(out)


def _infer_varied_ivs(results):
    """Fallback IV keys derived straight from the ``config_*`` columns that
    actually vary across ``results``. Used when the caller's declared ``ivs``
    dict is empty or incomplete -- e.g. a registry lookup for an experiment
    that hasn't finished (and so isn't registered) yet -- so the statistical
    tests never silently disagree with the "Varied parameters" table above
    them, which is already computed straight from the data."""
    if not results:
        return {}
    sample_row = results[0]
    config_keys = [k for k in sample_row if k.startswith('config_')
                  and k not in ('config_id', 'config_idx', 'config_name')]
    varied = {}
    for key in config_keys:
        values = {str(r.get(key)) for r in results if key in r}
        if len(values) > 1:
            label = key[len('config_'):].replace('_', '.', 1)
            varied[label] = sorted(values)
    return varied


def write_summary_md(md_file, results, ivs, ordinal_ivs, *, output_root,
                     trials=None, title="Experiment Summary", include_dir_structure=True):
    """Render the full markdown summary (overview + loss stats + per-config
    tables + statistical analysis) for a set of result rows.

    Config/trial counts are derived from ``results`` when not supplied, so this
    works for both a live Experiment and an arbitrary post-hoc subset.
    """
    ivs = dict(ivs)
    for label, values in _infer_varied_ivs(results).items():
        ivs.setdefault(label, values)
    config_keys = {r.get('config_id') or r.get('config_idx') for r in results}
    n_configs = len(config_keys)
    if trials is None:
        trials = 1 + max((int(r['trial_idx']) for r in results
                          if r.get('trial_idx') is not None), default=-1)

    md_dir = os.path.dirname(md_file) or '.'
    os.makedirs(md_dir, exist_ok=True)
    violin_pngs = stats.render_iv_pairs_violin_pngs(results, ivs, ordinal_ivs, md_dir)
    pvalue_pngs = stats.render_pvalue_heatmap_pngs(results, ivs, ordinal_ivs, md_dir)

    with open(md_file, 'w') as f:
        f.write(f"# {title}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Overview\n")
        f.write(f"- **Configurations:** {n_configs}\n")
        f.write(f"- **Trials per config:** {trials}\n")
        f.write(f"- **Total runs:** {len(results)}\n")
        f.write(f"- **Results store:** `{output_root}`\n\n")
        f.write(experiment_config_md(results, ivs, ordinal_ivs))
        f.write(statistical_tests_md(results, ivs, ordinal_ivs, violin_pngs, pvalue_pngs))
        f.write(loss_statistics_md(results))
        f.write(config_tables_md(results))
        if include_dir_structure:
            f.write(_directory_structure_md(output_root))


class Experiment:
    def __init__(
        self,
        base_config=None,
        ivs=None,
        *,
        name='experiment',
        trials=5,
        results_root='results',
        output_root=None,
        seed_mapping=None,
        global_seed=None,
        save_models=False,
        ordinal_ivs=None,
        extra_visualizations: Optional[Callable[[], List]] = None,
    ):
        """
        Initialize an Experiment for grid search with multiple trials.

        Results land in the flat global store at ``results_root``: one
        ``cfg_<hash>`` directory per distinct configuration (shared across
        experiments), a global results.csv/runs.csv, and a registry.json
        lookup table. Point ``results_root`` somewhere else (e.g.
        ``sandbox/my_test``) for a fully self-contained ad-hoc store.

        Args:
            base_config: Base RunConfig for all experiments
            ivs: Independent variables dict {param_name: [values]}
            name: Experiment name — the key for the registry record, the
                manifest (for later extension), and the ``experiment`` column
                in result rows. Purely logical; no directory is named after it.
            trials: Number of trials per configuration
            results_root: Root directory of the results store (default 'results')
            output_root: Deprecated alias for results_root
            seed_mapping: Optional function(config_idx, trial_idx) -> seed or dict {(cfg, trial): seed}
            global_seed: If provided, use this seed and increment for each trial/config
            save_models: If False (default), only save config.json and visualizations; if True, also save HDF5 training data and model weights
            ordinal_ivs: Subset of `ivs` keys that are ordinal (e.g. hidden_dim, cot).
                Used by the post-run statistical analysis (summary.md) to decide between a
                Spearman correlation (ordinal) or a Friedman/Nemenyi ranking (everything else,
                treated as non-ordinal/categorical). Defaults to none being ordinal.
            extra_visualizations: Optional zero-arg factory returning a fresh list of
                Visualization instances to register (in addition to
                Visualizer.register_defaults()) on every Visualizer this experiment
                creates. Called once per Visualizer so each config/trial gets its own
                instances (Visualization objects accumulate per-run state and must not
                be shared). Lets domain packages opt batch-granular or other
                domain-specific visualizations into grid runs without core knowing
                about them.
        """
        self.base_config = base_config or {}
        self.ivs = ivs or {}
        self.name = name
        self.trials = trials
        self.results_root = output_root if output_root is not None else results_root
        # Legacy alias, kept because docs/scripts referenced experiment.output_root.
        self.output_root = self.results_root
        self.seed_mapping = seed_mapping
        self.global_seed = global_seed
        self.save_models = save_models
        self.ordinal_ivs = list(ordinal_ivs or [])
        self.extra_visualizations = extra_visualizations
        self._force = False
        self._store_cache = {}
        # Detached figure-render pool (set up per run in run_grid/extend); defaults
        # keep _run_trial_rounds safe if ever called without that setup.
        self._render_executor = None
        self._render_futures = []
        self._create_configs()
        self.results = []

    def _get_seed(self, config_idx: int, trial_idx: int) -> int:
        """
        Determine the seed for a given config/trial combination.

        Precedence:
        1. seed_mapping (if dict with key (config_idx, trial_idx))
        2. seed_mapping (if callable)
        3. global_seed (if incremented per combination)
        4. trial_idx (default fallback)
        """
        if isinstance(self.seed_mapping, dict):
            return self.seed_mapping.get((config_idx, trial_idx), trial_idx)
        elif callable(self.seed_mapping):
            return self.seed_mapping(config_idx, trial_idx)
        elif self.global_seed is not None:
            # Vary seed only by trial index, so a given trial number shares
            # the same seed across all configs (isolates the IV's effect).
            return self.global_seed + trial_idx
        else:
            return trial_idx

    def _apply_iv(self, config: RunConfig, key: str, value) -> RunConfig:
        """
        Apply an IV to a config, searching sub-configs (data, model, train) in order.
        Supports dot notation (e.g., 'train.epochs').

        For dict fields like arch_kwargs, merges the new value into the existing dict.

        Args:
            config: RunConfig to modify
            key: IV key (simple name or dot notation)
            value: value to set

        Returns:
            New RunConfig with the IV applied (does not mutate original)
        """
        if '.' in key:
            # Dot notation: e.g., 'train.epochs'
            parts = key.split('.')
            if len(parts) != 2:
                raise ValueError(f"Only two-level dot notation supported, got: {key}")
            sub_name, attr = parts
            sub_config = getattr(config, sub_name, None)
            if sub_config is None:
                raise KeyError(f"Config has no sub-config '{sub_name}'")
            # Replace the sub-config with updated copy
            updated_sub = replace(sub_config, **{attr: value})
            return replace(config, **{sub_name: updated_sub})
        else:
            # Search sub-configs in order: data, model, train
            for sub_config_name in ['data', 'model', 'train']:
                sub_config = getattr(config, sub_config_name, None)
                if sub_config is None:
                    continue
                if hasattr(sub_config, key):
                    # Special handling for dict fields: merge instead of replace
                    if key == 'arch_kwargs' and isinstance(value, dict):
                        existing = getattr(sub_config, key, {})
                        merged = {**existing, **value}
                        updated_sub = replace(sub_config, **{key: merged})
                    else:
                        updated_sub = replace(sub_config, **{key: value})
                    return replace(config, **{sub_config_name: updated_sub})
            raise KeyError(f"IV '{key}' not found in any sub-config (data, model, train)")

    def _register_extra_visualizations(self, viz):
        """Register this experiment's extra_visualizations (if any) onto viz.

        Calls the factory once per Visualizer so each config/trial gets its
        own Visualization instances, matching register_defaults()'s pattern.
        """
        if self.extra_visualizations is not None:
            for v in self.extra_visualizations():
                viz.register(v)

    def _config_dir(self, config_idx, config):
        """Global content-addressed per-config output directory.

        Registers the config in the store's lookup table on first sight and
        returns (config_id, config_name, config_dir)."""
        config_name = self._config_to_name(config_idx, config)
        config_id = registry.get_or_register_config(self.results_root, config,
                                                    name=config_name)
        return config_id, config_name, registry.config_dir(self.results_root, config_id)

    def _prepare_configs(self, config_indices, visualize, sampling, device, load_viz_state=True):
        """Create per-config output dirs + (optionally) one long-lived Visualizer
        each, saving each config.json/pkl. Returns (config_meta, visualizers).

        config_meta: list of (config_idx, config, config_id, config_name, config_dir).
        visualizers: parallel list of Visualizer or None.

        When load_viz_state=True (default), any previously-saved viz_state in the
        config's directory is loaded so prior trials — from this experiment or any
        other that ran the same config — merge into the combined figures.
        """
        config_meta, visualizers = [], []
        for config_idx in config_indices:
            config = self.configs[config_idx]
            config_id, config_name, config_dir = self._config_dir(config_idx, config)
            os.makedirs(config_dir, exist_ok=True)
            self._save_config(config, config_dir)
            self._set_config_id(config_idx, config_id)

            viz = None
            if visualize:
                viz = Visualizer(name=config_id, output_dir=config_dir,
                                 sampling=sampling, device=device)
                viz.register_defaults()
                self._register_extra_visualizations(viz)
                if load_viz_state:
                    if viz.load_state():
                        print(f"  ↻ Loaded prior visualization state for {config_id}")
            config_meta.append((config_idx, config, config_id, config_name, config_dir))
            visualizers.append(viz)
        return config_meta, visualizers

    def _set_config_id(self, config_idx, config_id):
        """Record the global config_id for a config index (parallel list)."""
        while len(self.config_ids) <= config_idx:
            self.config_ids.append(None)
        self.config_ids[config_idx] = config_id

    def _load_store_cache(self):
        """Index of successful rows already in the store, for cell-level caching."""
        self._store_cache = {}
        for r in registry.load_results(self.results_root):
            loss = r.get('final_test_loss')
            ok = loss is not None and not (isinstance(loss, float) and np.isnan(loss))
            if ok and r.get('config_id') and r.get('seed') is not None:
                self._store_cache[(r['config_id'], r['seed'])] = _coerce_loaded_result(r)

    def _run_single_trial(self, config, config_idx, config_id, config_name,
                          config_dir, trial_idx, visualizer, dataset_cache, *,
                          config_pos=None, num_configs=None,
                          round_pos=None, num_rounds=None):
        """Train one (config, trial) cell, appending a result row to self.results.

        If the store already holds a successful row for this (config_id, seed)
        — from this experiment or any other — the stored row is adopted and
        training is skipped (unless force=True was passed to run_grid/extend).

        Reuses a cached Dataset when another config in this round already built an
        identical one (same DatasetConfig + cot + seed). Dataset generation
        touches no global RNG, so injecting the cache is bit-identical to letting
        from_run_config rebuild it (see Processor.from_run_config).

        config_pos/num_configs/round_pos/num_rounds (all 1-based) are purely
        cosmetic progress fractions for the start-of-cell print."""
        progress = ""
        if config_pos is not None and round_pos is not None:
            progress = f"  (config {config_pos}/{num_configs}, round {round_pos}/{num_rounds})"
        print(f"\n  [{config_id}.{trial_idx}] {config_name}{progress}...", end=" ", flush=True)
        seed = self._get_seed(config_idx, trial_idx)

        if not self._force:
            cached = self._store_cache.get((config_id, seed))
            if cached is not None:
                row = dict(cached)
                row.update(config_idx=config_idx, trial_idx=trial_idx,
                           config_name=config_name, config_id=config_id,
                           experiment=self.name)
                self._record_result(row)
                print(f"↩ cached (stored result for seed {seed} reused; force=True to re-run)",
                      flush=True)
                # No visualizer.next_trial(): this trial's viz data (if any)
                # already lives in the config dir's saved viz_state.
                self._report_grid_progress(config_name)
                return

        processor = None
        try:
            trial_config = self._apply_iv(config, 'seed', seed)

            # Build-or-reuse the dataset for this (data-config, cot, seed).
            cache_key = self._dataset_cache_key(trial_config, seed)
            dataset = dataset_cache.get(cache_key)
            if dataset is None:
                dataset = trial_config.data.build(
                    seed=seed,
                    device=Processor.resolve_device(trial_config.train.device),
                    cot=trial_config.train.cot,
                )
                dataset_cache[cache_key] = dataset

            processor = Processor.from_run_config(trial_config, visualizer=visualizer,
                                                  dataset=dataset)
            processor.run()

            final_train_loss = float(processor.logs['train_loss'][-1])
            final_test_loss = float(processor.logs['test_loss'][-1])
            best_test_loss = float(np.min(processor.logs['test_loss']))
            best_test_epoch = int(np.argmin(processor.logs['test_loss']))
            subset_losses = stats.compute_subset_test_losses(processor)

            result_entry = {
                'config_id': config_id,
                'experiment': self.name,
                'config_idx': config_idx,
                'trial_idx': trial_idx,
                'config_name': config_name,
                'seed': seed,
                'final_train_loss': final_train_loss,
                'final_test_loss': final_test_loss,
                'best_test_loss': best_test_loss,
                'best_test_epoch': best_test_epoch,
                'epochs': processor.epochs,
                'test_loss_all': subset_losses['all'],
                'test_loss_in_domain': subset_losses['in_domain'],
                'test_loss_domain_id': subset_losses['domain_id'],
                'test_loss_len_id': subset_losses['len_id'],
                'test_loss_domain_ood': subset_losses['domain_ood'],
                'test_loss_len_ood': subset_losses['len_ood'],
            }
            for field_name in ['data', 'model', 'train']:
                sub = getattr(config, field_name)
                for field in fields(sub):
                    result_entry[f'config_{field_name}_{field.name}'] = \
                        format_config_value(getattr(sub, field.name))

            self._record_result(result_entry)
            registry.append_run(self.results_root, {**result_entry, 'status': 'ok'})

            if self.save_models:
                model_filename = os.path.join(config_dir, f'trial_{trial_idx}.pt')
                processor.save(model_filename, output_dir=config_dir)

            print(f"✓ [{config_name}] Train Loss: {final_train_loss:.6f}, "
                  f"Test Loss: {final_test_loss:.6f}", flush=True)

        except Exception as e:
            print(f"✗ [{config_name}] FAILED: {e}", file=sys.stderr, flush=True)
            failed_entry = {
                'config_id': config_id, 'experiment': self.name,
                'config_idx': config_idx, 'trial_idx': trial_idx,
                'config_name': config_name, 'seed': seed,
                'final_train_loss': None, 'final_test_loss': None,
                'best_test_loss': None, 'best_test_epoch': None, 'epochs': None,
                'test_loss_all': None, 'test_loss_in_domain': None,
                'test_loss_domain_id': None, 'test_loss_len_id': None,
                'test_loss_domain_ood': None, 'test_loss_len_ood': None,
                'error': str(e),
            }
            self._record_result(failed_entry)
            registry.append_run(self.results_root, {**failed_entry, 'status': 'failed'})
        finally:
            # Advance the visualizer trial boundary even on failure so the next
            # trial writes to a fresh slot.
            if visualizer:
                visualizer.next_trial()
            if processor is not None:
                del processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._report_grid_progress(config_name)

    def _report_grid_progress(self, config_name):
        """Print cumulative elapsed/estimated-total wall time across the whole
        run_grid/extend invocation (all rounds, all configs, both phases of an
        extend). No-op if the grid-level tracker wasn't set up by the caller."""
        total_cells = getattr(self, '_grid_total_cells', None)
        if total_cells is None:
            return
        self._grid_cells_done += 1
        elapsed = time.time() - self._grid_start_time
        total_estd = elapsed / self._grid_cells_done * total_cells
        print(f"  Total: {format_duration(elapsed)}/{format_duration(total_estd)} estd "
              f"(cell {self._grid_cells_done}/{total_cells}, last: {config_name})", flush=True)

    def _record_result(self, result_entry):
        """Append a result row, replacing any existing row for the same
        (config_idx, trial_idx) so re-runs / extends stay idempotent."""
        key = (result_entry['config_idx'], result_entry['trial_idx'])
        self.results = [r for r in self.results
                        if (r['config_idx'], r['trial_idx']) != key]
        self.results.append(result_entry)

    def _dataset_cache_key(self, config, seed):
        """Value-based key identifying datasets that are identical across configs
        (so archs/hidden_dims sharing a DatasetConfig + cot + seed reuse one
        generated dataset within a trial round)."""
        dc = config.data
        field_vals = tuple(str(format_config_value(getattr(dc, f.name)))
                           for f in fields(dc))
        return field_vals + (str(format_config_value(config.train.cot)), seed)

    def _run_trial_rounds(self, trial_indices, config_meta, visualizers, *,
                          parallel_viz=True, animate=True):
        """Run trials **round by round** (trial-outer, config-inner) so each
        distinct dataset is built once per round and reused across configs, and
        so partial results appear as the run progresses.

        Refresh policy (per user request):
          - Every round: rewrite results.csv + summary.md in the background pool.
          - First and last round only: render the (expensive) per-config
            visualizations. Rendering is fully decoupled from training via
            snapshot-and-detach: the moment a config finishes training we
            snapshot its committed viz state to disk (cheap: pickle + gzip'd
            HDF5 copy) and hand a *fresh* Visualizer, loaded from that snapshot,
            to the shared render pool. Training never waits on rendering — the
            live Visualizer keeps accumulating the next round's trials while the
            snapshot renders. All renders are drained once, at the end of
            run_grid/extend (see _drain_render_pool).
        """
        trial_indices = list(trial_indices)
        for pos, trial_idx in enumerate(trial_indices):
            is_first, is_last = pos == 0, pos == len(trial_indices) - 1
            print(f"\n{'─'*80}\nTRIAL ROUND {trial_idx} ({pos + 1}/{len(trial_indices)})\n{'─'*80}",
                  flush=True)

            # Expensive animation refresh on first + last round only.
            do_finalize = animate and (is_first or is_last)
            if do_finalize:
                self._render_log(f"Snapshotting + dispatching visualizations for round "
                                  f"{trial_idx} (rendering runs detached; training does "
                                  f"not wait)...")

            dataset_cache = {}
            for list_pos, ((config_idx, config, config_id, config_name, config_dir), viz) \
                    in enumerate(zip(config_meta, visualizers)):
                self._run_single_trial(config, config_idx, config_id, config_name,
                                       config_dir, trial_idx, viz, dataset_cache,
                                       config_pos=list_pos + 1, num_configs=len(config_meta),
                                       round_pos=pos + 1, num_rounds=len(trial_indices))
                # Snapshot this config's freshly-committed viz state and dispatch
                # a detached render, so it renders while the remaining configs —
                # and every later round — keep training. save_state is a cheap
                # pickle + gzip'd-HDF5 copy; detach_processor then frees the GPU
                # model (and stashes reference logs for the final save_state).
                if do_finalize and viz is not None:
                    self._dispatch_detached_render(viz, config_id, config_dir, trial_idx)
                # Cheap refresh after every trial-config cell (background thread
                # pool) so the registry reflects live progress, not just the
                # state after the experiment fully completes.
                self._schedule_registry_refresh()
            dataset_cache.clear()

            # Cheap refresh every round (background thread pool).
            self._schedule_summary_refresh()

    def _render_log_stream(self):
        """Lazily open results_root/render.log (append mode) and return its
        file handle, so both explicit _render_log() calls and the stdout
        router (for prints made deep inside Visualizer.finalize()) share one
        destination."""
        fh = getattr(self, '_render_log_fh', None)
        if fh is None:
            os.makedirs(self.results_root, exist_ok=True)
            fh = open(os.path.join(self.results_root, 'render.log'), 'a')
            self._render_log_fh = fh
        return fh

    def _render_log(self, msg):
        """Write a timestamped line to results_root/render.log instead of
        stdout, so training prints (stdout) stay uncluttered by finalization/
        render-pool activity that runs concurrently on background threads."""
        fh = self._render_log_stream()
        fh.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        fh.flush()

    def _close_render_log(self):
        fh = getattr(self, '_render_log_fh', None)
        if fh is not None:
            fh.close()
            self._render_log_fh = None

    def _dispatch_detached_render(self, viz, config_id, config_dir, trial_idx):
        """Snapshot a config's committed viz state to a distinct per-round dir and
        submit a detached render of it to the shared render pool. Runs on the
        training thread but does only cheap I/O; the expensive render is off-thread.
        """
        snapshot_dir = os.path.join(config_dir, '.render_snapshots', f'round{trial_idx}')
        shutil.rmtree(snapshot_dir, ignore_errors=True)  # clear any stale partial
        viz.save_state(snapshot_dir)   # pickle committed arrays + copy live HDF5
        viz.detach_processor()         # free GPU model; stash reference for final save_state
        if self._render_executor is not None:
            self._render_futures.append(
                self._render_executor.submit(
                    self._render_from_snapshot, config_id, config_dir, snapshot_dir))
        else:
            # No pool set up (e.g. _run_trial_rounds called directly): render inline.
            self._render_from_snapshot(config_id, config_dir, snapshot_dir)

    def _render_from_snapshot(self, config_id, config_dir, snapshot_dir):
        """Render one config's figures from a detached viz snapshot (no live viz
        or processor), then delete the snapshot dir. Mirrors
        refresh_results.render_config_figures; _RENDER_LOCK keeps matplotlib safe.

        Runs on the render-pool worker thread; routes this thread's stdout
        (including the many ad hoc prints inside Visualizer/Visualization
        .finalize()) to render.log so it doesn't interleave with the main
        thread's training prints.
        """
        router = _install_stdout_router()
        router.route_to(self._render_log_stream())
        try:
            viz = Visualizer(name=config_id, output_dir=config_dir,
                             sampling=getattr(self, '_viz_sampling', 5),
                             device=getattr(self, '_viz_device', 'cpu'))
            viz.register_defaults()
            self._register_extra_visualizations(viz)
            # Drop visualizations with no saved state (finalizing them would only
            # produce empty/broken figures).
            missing = [v.name for v in viz.visualizations.values()
                       if not os.path.exists(os.path.join(snapshot_dir, f'{v.name}.state.pkl'))]
            viz.load_state(snapshot_dir)
            for name in missing:
                viz.visualizations.pop(name, None)
            if viz.visualizations:
                viz.finalize()
            viz.cleanup()
        except Exception as e:
            self._render_log(f"✗ Detached render failed for {config_id}: {e}")
        finally:
            shutil.rmtree(snapshot_dir, ignore_errors=True)
            router.reset()

    def _drain_render_pool(self):
        """Wait for every detached figure render to finish, then shut the pool.
        Called once at the end of run_grid/extend, so training is never blocked
        on rendering during the run."""
        ex = getattr(self, '_render_executor', None)
        if ex is None:
            return
        futures = getattr(self, '_render_futures', [])
        if futures:
            self._render_log(f"Draining {len(futures)} detached visualization render(s)...")
        for fut in futures:
            try:
                fut.result()
            except Exception as e:
                self._render_log(f"✗ Detached render error: {e}")
        self._render_futures = []
        ex.shutdown(wait=True)
        self._render_executor = None
        if futures:
            self._render_log("✓ Visualization rendering complete")

    def _schedule_summary_refresh(self):
        """Dispatch results.csv + summary.md regeneration to the background pool
        with a snapshot of self.results (result dicts are never mutated after
        creation, so the snapshot is race-free)."""
        if not self.results:
            return
        snapshot = list(self.results)
        ex = getattr(self, '_summary_executor', None)
        if ex is None:
            self._write_summary_files(snapshot)
        else:
            ex.submit(self._write_summary_files, snapshot)

    def _merged_store_rows(self, results):
        """This experiment's rows merged into the store's canonical rows.

        Rows are keyed (config_id, seed): our snapshot replaces any stored row
        for the same cell (idempotent re-runs), everything else is preserved.
        """
        mine = {(r.get('config_id'), r.get('seed')) for r in results}
        existing = [_coerce_loaded_result(r)
                    for r in registry.load_results(self.results_root)
                    if (r.get('config_id'), r.get('seed')) not in mine]
        return existing + list(results)

    def _pooled_ivs(self):
        """Union of IVs/ordinals across every registered experiment plus this one
        (which may not be registered yet mid-run)."""
        ivs, ordinal = registry.union_ivs(self.results_root)
        for k, vals in self.ivs.items():
            merged = ivs.setdefault(k, [])
            for v in vals:
                s = str(format_config_value(v))
                if s not in merged:
                    merged.append(s)
        ordinal = sorted(set(ordinal) | set(self.ordinal_ivs))
        return ivs, ordinal

    def _write_summary_files(self, results):
        os.makedirs(self.results_root, exist_ok=True)
        try:
            merged = self._merged_store_rows(results)
            write_results_csv(merged, registry.results_csv_path(self.results_root))
            ivs, ordinal = self._pooled_ivs()
            write_summary_md(registry.summary_md_path(self.results_root), merged,
                             ivs, ordinal, output_root=self.results_root,
                             title="Results Store Summary (pooled)")
        except Exception as e:
            self._render_log(f"⊘ Warning: summary refresh failed: {e}")

    def run_grid(self, visualize=True, sampling=5, device='cpu', parallel_viz=True,
                 force=False):
        """
        Run the full experimental grid, trial-outer so each distinct dataset is
        generated once per round and reused across configs, with live-updating
        results and early (round-0) visualizations.

        Cells whose (config_id, seed) already have a successful row in the
        store — from any experiment — are skipped and the stored row reused,
        unless force=True.

        Args:
            visualize: Whether to run the visualization pipeline.
            sampling: Sampling rate for visualizations (every Nth epoch).
            device: Device for GPU-accelerated visualizations ('cpu' or 'cuda').
            parallel_viz: Retained for API compatibility (background finalization
                is always used for the per-round refresh).
            force: Re-run cells even if the store already has their results.
        """
        print(f"\n{'='*80}")
        print(f"Starting experiment '{self.name}' with {len(self.configs)} configs × {self.trials} trials")
        print(f"Store: '{self.results_root}' (flat, content-addressed config dirs)")
        print(f"Order: trial-outer (dataset built once per round, reused across configs)")
        print(f"Finalization/render log: {os.path.join(self.results_root, 'render.log')}")
        print(f"{'='*80}\n")

        self._viz_sampling, self._viz_device = sampling, device
        self._force = force
        self._load_store_cache()

        config_meta, visualizers = self._prepare_configs(
            range(len(self.configs)), visualize, sampling, device,
            load_viz_state=not force)
        self._save_manifest()
        self._register()  # visible in the registry from the start, even mid-run

        self._grid_start_time = time.time()
        self._grid_total_cells = len(config_meta) * self.trials
        self._grid_cells_done = 0

        self._summary_executor = ThreadPoolExecutor(max_workers=1)
        self._render_executor = ThreadPoolExecutor(max_workers=1)
        self._render_futures = []
        try:
            self._run_trial_rounds(range(self.trials), config_meta, visualizers,
                                   parallel_viz=parallel_viz)
        finally:
            self._summary_executor.shutdown(wait=True)
            self._summary_executor = None
            # Drain all detached figure renders (training never waited on them),
            # then persist per-config viz state for later extend + release handles.
            self._drain_render_pool()
            self._persist_visualizers(visualizers)
            self._close_render_log()

        self._save_results_summary()
        self._save_manifest()
        self._register()
        print(f"\n{'='*80}")
        print(f"Experiment '{self.name}' complete! Results saved to '{self.results_root}'")
        print(f"{'='*80}\n")

    def _save_results_summary(self):
        """Save results as CSV and markdown summary (synchronous, final)."""
        if not self.results:
            print("No results to summarize.")
            return
        self._write_summary_files(self.results)
        print(f"✓ Results CSV + summary.md saved to '{self.results_root}'")

    def _create_configs(self):
        """Create all configuration combinations from independent variables.

        Also records ``self.config_iv_values`` (a parallel list of the {iv_key:
        value} dict for each config) so config_idx can be persisted/extended
        independently of product ordering, and initializes ``self.config_ids``
        (global content-hash ids, filled in by _prepare_configs).
        """
        self.config_ids = []
        if not self.ivs:
            self.configs = [self.base_config]
            self.config_iv_values = [{}]
            return

        self.configs = []
        self.config_iv_values = []
        for val_comb in product(*self.ivs.values()):
            iv_values = dict(zip(self.ivs.keys(), val_comb))
            config = self.base_config
            for key, val in iv_values.items():
                config = self._apply_iv(config, key, val)
            self.configs.append(config)
            self.config_iv_values.append(iv_values)

    def _combo_key(self, iv_values):
        """Value-based canonical key for a config's IV assignment (so equal
        combos compare equal even across pickled/reloaded object identities)."""
        return tuple(sorted((k, str(format_config_value(v))) for k, v in iv_values.items()))

    def _rebuild_configs_from_specs(self, specs):
        """Rebuild self.configs / config_iv_values from persisted ordered IV-value
        dicts, so config_idx matches exactly what was saved (independent of any
        later product-order changes)."""
        self.config_iv_values = [dict(s) for s in specs]
        self.configs = []
        for iv_values in self.config_iv_values:
            config = self.base_config
            for key, val in iv_values.items():
                config = self._apply_iv(config, key, val)
            self.configs.append(config)

    # ------------------------------------------------------------------
    # Manifest persistence + reload + extension
    # ------------------------------------------------------------------
    def _save_manifest(self):
        """Persist everything needed to reconstruct/extend this experiment into
        the store's manifests.pkl (keyed by experiment name)."""
        manifest = {
            'name': self.name,
            'base_config': self.base_config,
            'ivs': self.ivs,
            'ordinal_ivs': self.ordinal_ivs,
            'trials': self.trials,
            # Callables can't be reliably re-pickled/reconstructed; only dict
            # seed mappings round-trip. global_seed / default fallback still do.
            'seed_mapping': self.seed_mapping if isinstance(self.seed_mapping, dict) else None,
            'global_seed': self.global_seed,
            'save_models': self.save_models,
            'config_iv_values': self.config_iv_values,
            # Persisted ids keep config_idx ↔ config_id stable on reload even if
            # identity-hashing details evolve between code versions.
            'config_ids': list(self.config_ids),
            'viz_sampling': getattr(self, '_viz_sampling', 5),
            'viz_device': getattr(self, '_viz_device', 'cpu'),
        }
        registry.save_manifest(self.results_root, self.name, manifest)

    def _register(self):
        """Upsert this experiment into the store's registry with current results."""
        self._register_snapshot(list(self.results), [c for c in self.config_ids if c])

    def _register_snapshot(self, results, config_ids):
        """Upsert this experiment into the store's registry using a fixed
        snapshot of results/config_ids (race-free when called from a
        background thread while training keeps appending to self.results)."""
        try:
            registry.register_experiment(
                self.results_root, name=self.name, ivs=self.ivs,
                ordinal_ivs=self.ordinal_ivs, trials=self.trials,
                global_seed=self.global_seed,
                config_ids=config_ids,
                results=results)
        except Exception as e:
            print(f"⊘ Warning: registry update failed: {e}", file=sys.stderr, flush=True)

    def _schedule_registry_refresh(self):
        """Dispatch a registry.json upsert reflecting current progress, so a
        still-running (not-yet-fully-registered) experiment is visible to
        analyze.py/view.py incrementally -- after every trial-config cell --
        instead of only once the whole grid finishes."""
        snapshot = list(self.results)
        config_ids = [c for c in self.config_ids if c]
        ex = getattr(self, '_summary_executor', None)
        if ex is None:
            self._register_snapshot(snapshot, config_ids)
        else:
            ex.submit(self._register_snapshot, snapshot, config_ids)

    def _persist_visualizers(self, visualizers):
        """Save per-config viz state (for later extend) and release open handles."""
        for viz in visualizers:
            if viz is not None:
                viz.save_state()
                viz.cleanup()

    @classmethod
    def load(cls, name, results_root='results'):
        """Reconstruct an Experiment from the store's manifest + global results.

        The returned Experiment has the exact configs/indices, IVs, seeds, and
        prior result rows (all rows in the store for its config ids), ready for
        ``extend(...)`` or summary regeneration.
        """
        manifests = registry.load_manifests(results_root)
        if name not in manifests:
            available = ', '.join(sorted(manifests)) or '(none)'
            raise KeyError(f"No experiment '{name}' in '{registry.manifests_path(results_root)}'. "
                           f"Available: {available}")
        m = manifests[name]
        exp = cls(
            base_config=m['base_config'],
            ivs=m['ivs'],
            name=name,
            trials=m['trials'],
            results_root=results_root,
            seed_mapping=m.get('seed_mapping'),
            global_seed=m.get('global_seed'),
            save_models=m.get('save_models', False),
            ordinal_ivs=m.get('ordinal_ivs'),
        )
        exp._rebuild_configs_from_specs(m['config_iv_values'])
        exp.config_ids = list(m.get('config_ids', []))
        exp._viz_sampling = m.get('viz_sampling', 5)
        exp._viz_device = m.get('viz_device', 'cpu')
        own_ids = {c for c in exp.config_ids if c}
        exp.results = [_coerce_loaded_result(r)
                       for r in registry.load_results(results_root)
                       if r.get('config_id') in own_ids]
        return exp

    def _add_configs(self, add_ivs):
        """Merge new IV values into self.ivs and append any resulting config
        combinations not already present. Returns the new config_idx list."""
        existing_keys = {self._combo_key(iv) for iv in self.config_iv_values}
        for key, vals in add_ivs.items():
            current = list(self.ivs.get(key, []))
            for v in vals:
                if not any(str(format_config_value(v)) == str(format_config_value(c))
                           for c in current):
                    current.append(v)
            self.ivs[key] = current

        new_indices = []
        next_idx = len(self.configs)
        for val_comb in product(*self.ivs.values()):
            iv_values = dict(zip(self.ivs.keys(), val_comb))
            key = self._combo_key(iv_values)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            config = self.base_config
            for k, v in iv_values.items():
                config = self._apply_iv(config, k, v)
            self.configs.append(config)
            self.config_iv_values.append(iv_values)
            new_indices.append(next_idx)
            next_idx += 1
        return new_indices

    def extend(self, trials=None, add_ivs=None, *, visualize=True,
               sampling=None, device=None, parallel_viz=True, force=False):
        """Extend a finished experiment in place: add trials to existing configs
        and/or add new configs, running only what's missing.

        Args:
            trials: new total trial count (>= current). Existing configs run the
                added rounds [old_trials, trials); new configs run [0, trials).
            add_ivs: {iv_key: [values]} of IV values to add (e.g. {'arch': [GRU]}).
            visualize/sampling/device/parallel_viz: as in run_grid.
            force: re-run cells even if the store already has their results.
        """
        sampling = sampling if sampling is not None else getattr(self, '_viz_sampling', 5)
        device = device if device is not None else getattr(self, '_viz_device', 'cpu')
        self._viz_sampling, self._viz_device = sampling, device
        self._force = force
        self._load_store_cache()

        old_trials = self.trials
        target_trials = trials if trials is not None else old_trials
        if target_trials < old_trials:
            raise ValueError(f"Cannot shrink trials from {old_trials} to {target_trials}")

        existing_indices = list(range(len(self.configs)))
        new_indices = self._add_configs(add_ivs) if add_ivs else []
        self.trials = target_trials

        print(f"\n{'='*80}")
        print(f"Extending '{self.name}' in '{self.results_root}': trials {old_trials}→{target_trials}, "
              f"+{len(new_indices)} new config(s)")
        print(f"Finalization/render log: {os.path.join(self.results_root, 'render.log')}")
        print(f"{'='*80}\n")

        # Total cells across both phases, computed up front so the progress
        # tracker spans the whole extend() call rather than resetting per phase.
        self._grid_start_time = time.time()
        self._grid_total_cells = (
            (len(existing_indices) * (target_trials - old_trials)
             if target_trials > old_trials and existing_indices else 0)
            + len(new_indices) * target_trials
        )
        self._grid_cells_done = 0

        self._summary_executor = ThreadPoolExecutor(max_workers=1)
        self._render_executor = ThreadPoolExecutor(max_workers=1)
        self._render_futures = []
        try:
            # Existing configs: run just the new rounds, merging prior viz state.
            if target_trials > old_trials and existing_indices:
                cmeta, vizs = self._prepare_configs(existing_indices, visualize,
                                                    sampling, device, load_viz_state=True)
                self._run_trial_rounds(range(old_trials, target_trials), cmeta, vizs,
                                       parallel_viz=parallel_viz)
                self._persist_visualizers(vizs)
            # New configs: run the full trial set from scratch (their dirs may
            # still carry prior state if another experiment ran them).
            if new_indices:
                cmeta, vizs = self._prepare_configs(new_indices, visualize,
                                                    sampling, device,
                                                    load_viz_state=not force)
                self._run_trial_rounds(range(target_trials), cmeta, vizs,
                                       parallel_viz=parallel_viz)
                self._persist_visualizers(vizs)
        finally:
            self._summary_executor.shutdown(wait=True)
            self._summary_executor = None
            self._drain_render_pool()
            self._close_render_log()

        self._save_results_summary()
        self._save_manifest()
        self._register()
        print(f"\n{'='*80}")
        print(f"Extension complete! Results updated in '{self.results_root}'")
        print(f"{'='*80}\n")

    def _save_config(self, config, config_dir):
        """Save configuration to config.json (full flattened field set) and
        config.pkl (exact RunConfig, for faithful rerun via rerun_trial())."""
        config_file = os.path.join(config_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(registry.flatten_config(config), f, indent=2)

        pkl_file = os.path.join(config_dir, 'config.pkl')
        with open(pkl_file, 'wb') as f:
            pickle.dump(config, f)

    def _config_to_name(self, config_idx, config):
        """Create a readable slug for a configuration (e.g., rnn_vanilla_h64_L2_adam_lr1e-3).

        Purely cosmetic: stored as the config's descriptive name in the lookup
        table and used by view.py for browsable folder names."""
        if not self.ivs:
            return "base"

        abbrev_map = ABBREV_MAP

        def format_value(key, value):
            """Format a key-value pair into a slug component."""
            abbrev = abbrev_map.get(key, key)

            formatter = VALUE_SLUG_FORMATTERS.get(key)
            if formatter is not None:
                return formatter(abbrev, value)

            if isinstance(value, bool):
                return abbrev if value else ''

            elif isinstance(value, float):
                if 0.0001 <= value < 0.1:
                    return f"{abbrev}{value:.0e}".replace('e-0', 'e-')
                else:
                    return f"{abbrev}{value}"

            elif isinstance(value, (int, str)):
                return f"{abbrev}{value}"

            elif callable(value):
                name = getattr(value, '__name__', 'callable')
                return f"{abbrev}_{name}"

            else:
                return f"{abbrev}_{value.__class__.__name__}"

        def get_iv_value(key):
            """Get IV value from RunConfig sub-configs."""
            # Search in sub-configs
            for sub_name in ['data', 'model', 'train']:
                sub = getattr(config, sub_name)
                if hasattr(sub, key):
                    return getattr(sub, key)
            return None

        # Get values for all IVs in key order
        parts = []
        for key in self.ivs.keys():
            val = get_iv_value(key)
            if val is not None:
                formatted = format_value(key, val)
                if formatted:  # Skip empty strings (e.g., tropical=False)
                    parts.append(formatted)

        return "_".join(parts) if parts else f"config_{config_idx}"


def resolve_config_id(results_root, config_id=None, experiment=None, config_idx=None):
    """Resolve a global config id from either an explicit id or an
    (experiment, config_idx) pair via the experiment's manifest."""
    if config_id is not None:
        return config_id
    if experiment is None or config_idx is None:
        raise ValueError("Provide either config_id or both experiment and config_idx")
    manifests = registry.load_manifests(results_root)
    if experiment not in manifests:
        raise KeyError(f"No experiment '{experiment}' in '{registry.manifests_path(results_root)}'")
    ids = manifests[experiment].get('config_ids', [])
    if config_idx >= len(ids) or ids[config_idx] is None:
        raise KeyError(f"Experiment '{experiment}' has no config_id for config_idx={config_idx}")
    return ids[config_idx]


def rerun_trial(config_id=None, *, experiment=None, config_idx=None, trial_idx=None,
                results_root='results', visualize=False, sampling=5, device='cpu'):
    """
    Re-run one or more trials of a stored config, reproducing them exactly.

    Loads the exact original RunConfig from the pickled `config.pkl` in the
    config's store directory, and the recorded seed(s) from the global
    `results.csv`, then rebuilds and trains via `Processor.from_run_config` -
    identical to how `Experiment.run_grid` originally built the trial.

    Args:
        config_id: Global config id (e.g. 'cfg_a3f9d2c4'). Alternatively pass
            experiment + config_idx to resolve it from a manifest.
        experiment/config_idx: Alternative addressing via an experiment record.
        trial_idx: Which trial to rerun. If None, reruns every recorded trial.
        results_root: Store root (default 'results').
        visualize: Attach a fresh Visualizer and finalize it for each rerun trial.
        sampling: Visualizer sampling rate (only used if visualize=True).
        device: Device for visualization PCA (only used if visualize=True).

    Returns:
        List of (config_id, trial_idx, processor) tuples, in the order rerun.
    """
    config_id = resolve_config_id(results_root, config_id, experiment, config_idx)
    cfg_dir = registry.config_dir(results_root, config_id)

    rows = [r for r in registry.load_results(results_root)
            if r.get('config_id') == config_id
            and (trial_idx is None or r.get('trial_idx') == trial_idx)]
    if not rows:
        raise ValueError(f"No results found for config_id={config_id}, trial_idx={trial_idx} "
                         f"in '{registry.results_csv_path(results_root)}'")

    pkl_file = os.path.join(cfg_dir, 'config.pkl')
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"No config.pkl found in '{cfg_dir}'")
    with open(pkl_file, 'rb') as f:
        config = pickle.load(f)

    results = []
    for row in rows:
        t_idx = row['trial_idx']
        seed = row['seed']
        trial_config = replace(config, train=replace(config.train, seed=seed))

        visualizer = None
        if visualize:
            visualizer = Visualizer(name=f'rerun_{config_id}_trial{t_idx}',
                                    output_dir=cfg_dir, sampling=sampling, device=device)
            visualizer.register_defaults()

        processor = Processor.from_run_config(trial_config, visualizer=visualizer)
        processor.run()

        if visualizer:
            visualizer.next_trial()
            visualizer.finalize()
            visualizer.cleanup()

        results.append((config_id, t_idx, processor))

    return results


# ---------------------------------------------------------------------------
# CLI: extend a finished experiment in place.
#
# Add more trials to every existing config, and/or add new configs (e.g. a new
# architecture), running only what's missing. Cells already present in the store
# (from this experiment or any other that ran the same config) are skipped via
# the (config_id, seed) cache unless --force is passed. Prior trials'
# visualization state is reloaded so combined figures show old + new trials
# together, and statistics / the registry are recomputed.
#
# Examples:
#     # Grow every config of experiment 'grid_search' up to 10 trials
#     python experiment.py --name grid_search --trials 10
#
#     # Add a GRU architecture (runs the full trial set for the new configs)
#     python experiment.py --name grid_search --add-arch GRU
#
#     # Add two archs AND grow to 10 trials in one call
#     python experiment.py --name grid_search --add-arch GRU,MHA --trials 10
#
#     # Add arbitrary IV values (repeatable); values are coerced to int/float/bool
#     python experiment.py --name grid_search --set hidden_dim=32,128 --set cot=0.9
#
#     # Extend an experiment living in a sandbox store
#     python experiment.py --name my_test --results-root sandbox/my_test --trials 5
# ---------------------------------------------------------------------------

def _coerce_value(s: str):
    """Coerce a CLI string to bool/int/float, falling back to the raw string."""
    low = s.strip().lower()
    if low in ('true', 'false'):
        return low == 'true'
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def _resolve_arch(name: str):
    cls = ARCH_REGISTRY.get(name)
    if cls is None:
        raise SystemExit(f"Unknown architecture '{name}' "
                         f"(not registered; available: {sorted(ARCH_REGISTRY)})")
    return cls


def build_add_ivs(add_arch, set_specs):
    """Assemble an add_ivs dict from --add-arch and repeated --set flags."""
    add_ivs = {}
    if add_arch:
        add_ivs['arch'] = [_resolve_arch(n) for n in add_arch.split(',') if n]
    for spec in set_specs or []:
        if '=' not in spec:
            raise SystemExit(f"--set expects key=v1,v2 (got '{spec}')")
        key, _, vals = spec.partition('=')
        add_ivs.setdefault(key.strip(), [])
        # 'arch' values are resolved to model classes; everything else is coerced.
        if key.strip() == 'arch':
            add_ivs[key.strip()].extend(_resolve_arch(v) for v in vals.split(',') if v)
        else:
            add_ivs[key.strip()].extend(_coerce_value(v) for v in vals.split(',') if v)
    return add_ivs


def main():
    parser = argparse.ArgumentParser(
        description="Extend a finished experiment in place: add trials and/or "
                    "configs, running only what's missing.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True,
                        help="Experiment name (key in the store's manifests.pkl)")
    parser.add_argument('--results-root', default='results',
                        help="Results store root (default: results)")
    parser.add_argument('--trials', type=int, default=None,
                        help="New total trial count (>= current). Existing configs run the added rounds.")
    parser.add_argument('--add-arch', default=None,
                        help="Comma-separated architecture class names to add (e.g. GRU,MHA)")
    parser.add_argument('--set', dest='set_specs', action='append', default=[],
                        help="Add IV values: key=v1,v2 (repeatable)")
    parser.add_argument('--force', action='store_true',
                        help="Re-run cells even if the store already holds their results")
    parser.add_argument('--no-viz', action='store_true', help="Disable visualization")
    parser.add_argument('--sampling', type=int, default=None, help="Visualization sampling rate")
    parser.add_argument('--device', default=None, help="Device for visualization PCA (cpu/cuda)")
    args = parser.parse_args()

    add_ivs = build_add_ivs(args.add_arch, args.set_specs)
    if args.trials is None and not add_ivs:
        parser.error("Nothing to do: pass --trials and/or --add-arch/--set.")

    exp = Experiment.load(args.name, results_root=args.results_root)
    print(f"Loaded experiment '{args.name}': {len(exp.configs)} configs, {exp.trials} trials, "
          f"{len(exp.results)} existing result rows.")
    exp.extend(trials=args.trials, add_ivs=add_ivs or None,
               visualize=not args.no_viz, sampling=args.sampling, device=args.device,
               force=args.force)


if __name__ == '__main__':
    main()
