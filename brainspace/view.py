"""Build a browsable, descriptively-named view of a subset of the results store.

The store keeps every config in an opaque content-hash folder (``cfg_<hash>``).
This script retrieves a subset of configs (by experiment, config id, IV values,
trials), re-runs the statistical analysis on that subset, and assembles a
temporary ``views/`` folder where each config appears under its *descriptive
name* with copies of its visualizations. Nothing is regenerated — files are
copied verbatim from the store, so the view is cheap to build and safe to
delete.

Output layout:

    views/<YYYY-MM-DD>_<label>/
    ├── analysis_summary.md        # stats recomputed over the selected subset
    ├── <descriptive-name-1>/      # e.g. archGRU_h16_cot0.5  (= cfg_a3f9d2c4)
    │   ├── config.json
    │   └── ... copied .png/.gif/.mp4 visualizations
    └── <descriptive-name-2>/
        └── ...

Examples:
    # Everything from one experiment
    python view.py --experiment grid_search --label gru_sweep

    # Only GRU configs with cot 0.5, trials 0-4
    python view.py --arch GRU --set cot=0.5 --trials 0-4 --label gru_cot05

    # Specific configs by global id
    python view.py --config-id cfg_a3f9d2c4,cfg_0b7e11ff

    # From a declarative experiment spec -- trains any missing cells (cache-
    # aware, via Experiment.run_grid) then assembles the view
    python view.py --config configs/grid_search.yaml
"""
import argparse
import os
import shutil
from datetime import datetime

from . import config
from .internal import registry
from .experiment import Experiment, write_summary_md
from .analyze import add_filter_args, filter_store_rows, infer_ivs

# Visualization/media files worth copying into a view (rendered outputs only —
# viz_state/ and *.pkl internals are deliberately excluded).
VIEW_EXTENSIONS = ('.png', '.gif', '.mp4', '.jpg', '.jpeg', '.svg', '.html')


def safe_slug(name: str) -> str:
    """A filesystem-safe folder name for a config's descriptive name."""
    slug = "".join(c if (c.isalnum() or c in '-_.=') else '_' for c in str(name))
    return slug.strip('._') or 'config'


def copy_config_view(store_dir, dest_dir):
    """Copy a config folder's rendered outputs + config.json into dest_dir.

    Returns the number of files copied. Copies only (never regenerates)."""
    os.makedirs(dest_dir, exist_ok=True)
    copied = 0
    for fname in sorted(os.listdir(store_dir)):
        src = os.path.join(store_dir, fname)
        if not os.path.isfile(src):
            continue
        if fname == 'config.json' or fname.lower().endswith(VIEW_EXTENSIONS):
            shutil.copy2(src, os.path.join(dest_dir, fname))
            copied += 1
    return copied


def build_view(results_root, subset, view_dir, ivs, ordinal):
    """Assemble the view folder: per-config descriptive subfolders + stats."""
    reg = registry.load_registry(results_root)
    config_ids = sorted({r.get('config_id') for r in subset if r.get('config_id')})

    os.makedirs(view_dir, exist_ok=True)
    used_names = set()
    for cid in config_ids:
        entry = reg['configs'].get(cid, {})
        name = safe_slug(entry.get('name') or
                         next((r.get('config_name') for r in subset
                               if r.get('config_id') == cid and r.get('config_name')), cid))
        # Disambiguate descriptive-name collisions by suffixing the id.
        if name in used_names:
            name = f"{name}_{cid}"
        used_names.add(name)

        store_dir = registry.config_dir(results_root, cid)
        if not os.path.isdir(store_dir):
            print(f"⊘ {cid}: no store directory, skipped")
            continue
        n = copy_config_view(store_dir, os.path.join(view_dir, name))
        print(f"  {cid} → {name}/ ({n} files)")

    write_summary_md(os.path.join(view_dir, 'analysis_summary.md'), subset,
                     ivs, ordinal, output_root=results_root,
                     title="View Analysis", include_dir_structure=False)


def build_view_from_config(config_path, views_root, label):
    """Load a declarative experiment spec, ensure its results exist (running
    only missing cells -- Experiment.run_grid's cell-level caching skips
    anything already in the store), then assemble a view from its results."""
    spec = config.load_experiment_spec(config_path)
    experiment = Experiment(
        base_config=spec['base_config'], ivs=spec['ivs'], name=spec['name'],
        trials=spec['trials'], global_seed=spec['global_seed'],
        results_root=spec['results_root'], ordinal_ivs=spec['ordinal_ivs'],
        save_models=spec['save_models'],
    )
    experiment.run_grid(visualize=True)
    if not experiment.results:
        raise SystemExit(f"Experiment '{spec['name']}' produced no results; nothing to view.")

    label = label or spec['name']
    view_dir = os.path.join(views_root,
                            f"{datetime.now().strftime('%Y-%m-%d')}_{safe_slug(label)}")
    if os.path.exists(view_dir):
        shutil.rmtree(view_dir)

    build_view(experiment.results_root, experiment.results, view_dir,
              experiment.ivs, experiment.ordinal_ivs)
    print(f"\n✓ View assembled at '{view_dir}' "
          f"({len({r.get('config_id') for r in experiment.results})} configs; safe to delete anytime)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_filter_args(parser)
    parser.add_argument('--config', default=None,
                        help="Path to a declarative experiment spec YAML (see configs/). "
                             "Trains any missing cells then builds the view; overrides "
                             "the filter flags above.")
    parser.add_argument('--label', default=None,
                        help="View folder label (default: derived from filters).")
    parser.add_argument('--views-root', default='views',
                        help="Where view folders are created (default: views/).")
    args = parser.parse_args()

    if args.config:
        build_view_from_config(args.config, args.views_root, args.label)
        return

    _scope, subset, experiments, _trials = filter_store_rows(args)
    if not subset:
        raise SystemExit("Filtered subset is empty; nothing to view.")

    label = args.label or (('_'.join(sorted(experiments)) if experiments else 'all'))
    view_dir = os.path.join(args.views_root,
                            f"{datetime.now().strftime('%Y-%m-%d')}_{safe_slug(label)}")
    if os.path.exists(view_dir):
        shutil.rmtree(view_dir)

    ivs, ordinal = infer_ivs(args.results_root, experiments)
    build_view(args.results_root, subset, view_dir, ivs, ordinal)
    print(f"\n✓ View assembled at '{view_dir}' "
          f"({len({r.get('config_id') for r in subset})} configs; safe to delete anytime)")


if __name__ == '__main__':
    main()
