"""Upgrade every already-run experiment in the results store, in place.

Run this after a major repo update (e.g. a new standard visualization was
added to ``Visualizer.register_defaults()``, a renderer/style change, or a
schema tweak to config.json / summary statistics). For each stored config it:

  1. Regenerates ``config.json`` from the exact pickled RunConfig.
  2. Rebuilds every registered visualization from the saved ``viz_state/``
     with the *current* code — so renderer improvements apply retroactively.
     A visualization whose per-epoch data was never collected (it didn't exist
     when the config was trained) cannot be rendered from state; it is
     **skipped and reported** by default.
  3. With ``--rerun-missing``, configs from step 2's report are retrained from
     ``config.pkl`` + the recorded seeds so the new visualization's data gets
     collected, then all figures and the viz state are regenerated. Expensive.

Afterwards the store-level artifacts (results.csv, pooled summary.md,
index.md) are rewritten with current code.

For *schema* changes to the config dataclasses (renamed fields/values, new
fields), use ``migrate_registry.py`` instead — that is what re-hashes ids and
renames the ``cfg_*`` folders.

Examples:
    python refresh_results.py                          # whole store
    python refresh_results.py --experiment grid_search # one experiment's configs
    python refresh_results.py --config-id cfg_a3f9d2c4
    python refresh_results.py --rerun-missing          # backfill new visualizations
    python refresh_results.py --rebuild                # reconstruct registry.json from folders
"""
import argparse
import os
import pickle
import sys
from dataclasses import replace

from .internal import registry
from .experiment import (write_results_csv, write_summary_md,
                        _coerce_loaded_result)


def select_config_ids(results_root, config_ids=None, experiment=None):
    """Resolve the set of config ids to refresh."""
    reg = registry.load_registry(results_root)
    if config_ids:
        return [c for c in config_ids if c in reg['configs']
                or os.path.isdir(registry.config_dir(results_root, c))]
    if experiment:
        entry = registry.get_experiment(results_root, experiment)
        if entry is None:
            raise SystemExit(f"No experiment '{experiment}' in the registry.")
        return list(entry.get('config_ids', []))
    return sorted(reg['configs'])


def refresh_config_json(results_root, cid):
    """Rewrite config.json from the exact pickled RunConfig (schema drift)."""
    cdir = registry.config_dir(results_root, cid)
    pkl = os.path.join(cdir, 'config.pkl')
    if not os.path.exists(pkl):
        return False
    with open(pkl, 'rb') as f:
        config = pickle.load(f)
    import json
    with open(os.path.join(cdir, 'config.json'), 'w') as f:
        json.dump(registry.flatten_config(config), f, indent=2)
    return True


def refresh_visualizations(results_root, cid, sampling, device):
    """Re-render a config's figures from its saved viz_state with current code.

    Returns the list of registered visualization names that have *no* saved
    state (i.e. would need a retrain to appear), or None if the config has no
    viz_state at all."""
    from .visualization import Visualizer

    cdir = registry.config_dir(results_root, cid)
    state_dir = os.path.join(cdir, 'viz_state')
    if not os.path.isdir(state_dir):
        return None

    viz = Visualizer(name=cid, output_dir=cdir, sampling=sampling, device=device)
    viz.register_defaults()

    missing = [v.name for v in viz.visualizations.values()
               if not os.path.exists(os.path.join(state_dir, f'{v.name}.state.pkl'))]
    viz.load_state()
    # Drop visualizations with no data: finalizing them would only produce
    # empty/broken figures.
    for name in missing:
        viz.visualizations.pop(name, None)

    if viz.visualizations:
        viz.finalize()
    viz.cleanup()
    return missing


def rerun_config_with_viz(results_root, cid, sampling, device):
    """Retrain every recorded trial of a config with a full fresh Visualizer so
    all current visualizations (including newly-added ones) collect their data,
    then render and persist the new viz state."""
    from .train import Processor
    from .visualization import Visualizer

    cdir = registry.config_dir(results_root, cid)
    with open(os.path.join(cdir, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    rows = sorted((r for r in registry.load_results(results_root)
                   if r.get('config_id') == cid and r.get('seed') is not None),
                  key=lambda r: (r.get('trial_idx') is None, r.get('trial_idx')))
    if not rows:
        print(f"  ⊘ {cid}: no recorded trials to rerun")
        return

    viz = Visualizer(name=cid, output_dir=cdir, sampling=sampling, device=device)
    viz.register_defaults()
    for row in rows:
        print(f"  ↻ retraining {cid} trial {row.get('trial_idx')} (seed {row['seed']})...",
              flush=True)
        trial_config = replace(config, train=replace(config.train, seed=row['seed']))
        processor = Processor.from_run_config(trial_config, visualizer=viz)
        processor.run()
        viz.next_trial()
    viz.finalize()
    viz.save_state()
    viz.cleanup()


def refresh_store_files(results_root):
    """Rewrite results.csv / pooled summary.md / index.md with current code."""
    rows = [_coerce_loaded_result(r) for r in registry.load_results(results_root)]
    if rows:
        write_results_csv(rows, registry.results_csv_path(results_root))
        ivs, ordinal = registry.union_ivs(results_root)
        write_summary_md(registry.summary_md_path(results_root), rows, ivs, ordinal,
                         output_root=results_root,
                         title="Results Store Summary (pooled)")
    # save_registry regenerates index.md from the (possibly rebuilt) registry.
    registry.save_registry(results_root, registry.load_registry(results_root))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-root', default='results',
                        help="Results store root (default: results).")
    parser.add_argument('--config-id', dest='config_ids', default=None,
                        help="Comma-separated config ids to refresh (default: all).")
    parser.add_argument('--experiment', default=None,
                        help="Refresh only this experiment's configs.")
    parser.add_argument('--no-viz', action='store_true',
                        help="Skip re-rendering visualizations (config.json/CSV/summary only).")
    parser.add_argument('--rerun-missing', action='store_true',
                        help="Retrain configs whose saved viz_state lacks data for a "
                             "currently-registered visualization (expensive).")
    parser.add_argument('--rebuild', action='store_true',
                        help="Reconstruct registry.json's config lookup from the cfg_* folders first.")
    parser.add_argument('--sampling', type=int, default=5, help="Visualization sampling rate.")
    parser.add_argument('--device', default='cpu', help="Device for visualization PCA (cpu/cuda).")
    args = parser.parse_args()

    root = args.results_root
    if not os.path.isdir(root):
        raise SystemExit(f"No results store at '{root}'.")

    if args.rebuild:
        print(f"Rebuilding registry.json from config folders under '{root}'...")
        registry.rebuild_configs(root)

    config_ids = select_config_ids(
        root, args.config_ids.split(',') if args.config_ids else None,
        args.experiment)
    print(f"Refreshing {len(config_ids)} config(s) in '{root}'...\n")

    needs_rerun = {}   # cid -> [missing viz names]
    no_state = []
    for cid in config_ids:
        cdir = registry.config_dir(root, cid)
        if not os.path.isdir(cdir):
            print(f"⊘ {cid}: folder missing, skipped")
            continue
        print(f"• {cid}")
        if refresh_config_json(root, cid):
            print("  ✓ config.json regenerated")
        if not args.no_viz:
            missing = refresh_visualizations(root, cid, args.sampling, args.device)
            if missing is None:
                no_state.append(cid)
                print("  ⊘ no viz_state saved — figures not re-renderable without retraining")
            elif missing:
                needs_rerun[cid] = missing
                print(f"  ✓ figures re-rendered; missing data for: {', '.join(missing)}")
            else:
                print("  ✓ figures re-rendered from saved state")

    to_backfill = list(needs_rerun) + no_state
    if to_backfill and args.rerun_missing:
        print(f"\n--rerun-missing: retraining {len(to_backfill)} config(s) to "
              f"collect data for new visualizations...")
        for cid in to_backfill:
            rerun_config_with_viz(root, cid, args.sampling, args.device)

    refresh_store_files(root)
    print(f"\n✓ Store files (results.csv, summary.md, index.md) rewritten.")

    if to_backfill and not args.rerun_missing:
        print(f"\n⚠ {len(to_backfill)} config(s) have visualizations that can't be "
              f"rendered from saved state:")
        for cid in needs_rerun:
            print(f"   {cid}: missing {', '.join(needs_rerun[cid])}")
        for cid in no_state:
            print(f"   {cid}: no viz_state at all")
        print("→ Re-run with --rerun-missing to retrain them (uses config.pkl + recorded seeds).")


if __name__ == '__main__':
    main()
