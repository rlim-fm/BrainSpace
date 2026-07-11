"""Flat global results store: content-hash config identity + lookup registry.

Every run ever executed lands in a single store root (default ``results/``),
with one folder per *distinct configuration* named by a global, content-hash
config id:

    results/
    ├── registry.json   # lookup table: config_id → full config fields + name,
    │                   # plus one entry per experiment (ivs, trials, config_ids)
    ├── index.md        # human-readable rendering of registry.json
    ├── runs.csv        # APPEND-ONLY log of every run ever executed
    ├── results.csv     # canonical result rows for statistics, keyed (config_id, seed)
    ├── summary.md      # pooled statistics over every row in results.csv
    ├── manifests.pkl   # {experiment_name: manifest} for Experiment.load/extend
    └── cfg_<hash8>/    # per-config dir: config.json/.pkl, viz_state/, figures

Config identity
---------------
The identity of a config is its complete, fully-resolved field set — every
field of DatasetConfig/ModelConfig/TrainConfig, defaults included, rendered
with ``format_config_value`` — minus ``train_seed`` (varies per trial) and
``train_device`` (execution environment, not science).

    config_id = 'cfg_' + sha1(canonical-json(identity))[:8]

Consequences:
  - Adding new IV values (or whole new IVs) mints new ids automatically;
    existing ids and folders are untouched.
  - Re-encountering a previously-run config — in any experiment — maps back
    to the same folder and result rows (enabling cross-experiment caching).
  - Renaming a field/architecture or adding a new config field changes the
    canonical form; run ``migrate_registry.py`` to rewrite the lookup and
    re-hash/rename folders in one pass.

``registry.json`` is only an index: it can be rebuilt from the per-config
folders (``refresh_results.py --rebuild``), and hash-named folders mean
concurrent jobs can never collide on ids.
"""
import csv
import hashlib
import json
import os
import pickle
from dataclasses import fields
from datetime import datetime

import numpy as np

SCHEMA_VERSION = 1


def format_duration(seconds):
    """Human-readable duration for progress prints: '12.3s', '3m45s', '1h23m'."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h{mins}m"

# Fields excluded from config identity: seed varies per trial, device is the
# execution environment. Both still appear in config.json / results.csv rows.
IDENTITY_EXCLUDE = ('train_seed', 'train_device')

# Result columns that must be numeric for downstream statistics. '' / 'None'
# become NaN (floats) or None (ints) rather than raising.
_FLOAT_COLS = {
    'final_train_loss', 'final_test_loss', 'best_test_loss',
    'test_loss_all', 'test_loss_in_domain', 'test_loss_domain_id', 'test_loss_len_id',
    'test_loss_domain_ood', 'test_loss_len_ood',
}
_INT_COLS = {'config_idx', 'trial_idx', 'epochs', 'best_test_epoch', 'seed'}

_DATA_VIEWS = ['all', 'in_domain', 'domain_id', 'len_id', 'domain_ood', 'len_ood']

# Fixed column order for the append-only runs.csv log.
RUNS_FIELDS = [
    'timestamp', 'experiment', 'config_id', 'config_idx', 'trial_idx', 'seed',
    'status', 'final_train_loss', 'final_test_loss', 'best_test_loss',
    'best_test_epoch', 'epochs', 'test_loss_all', 'test_loss_in_domain',
    'test_loss_domain_ood', 'test_loss_len_ood', 'error',
    # Appended after 'error' (not inserted) so a runs.csv written with the
    # older header stays column-aligned when new rows are appended: existing
    # named columns keep their positions and the extra trailing values land in
    # DictReader's restkey rather than shifting every field.
    'test_loss_domain_id', 'test_loss_len_id',
]


# ----------------------------------------------------------------------------
# Store paths
# ----------------------------------------------------------------------------
def registry_path(root) -> str:
    return os.path.join(root, 'registry.json')


def index_path(root) -> str:
    return os.path.join(root, 'index.md')


def runs_csv_path(root) -> str:
    return os.path.join(root, 'runs.csv')


def results_csv_path(root) -> str:
    return os.path.join(root, 'results.csv')


def summary_md_path(root) -> str:
    return os.path.join(root, 'summary.md')


def manifests_path(root) -> str:
    return os.path.join(root, 'manifests.pkl')


def config_dir(root, config_id) -> str:
    return os.path.join(root, config_id)


# ----------------------------------------------------------------------------
# Config formatting, flattening, identity
# ----------------------------------------------------------------------------
def format_config_value(val):
    """Human-readable, *stable* string for a config field value.

    Primitives pass through as-is. Classes render as their name. Dicts render
    with sorted keys (so insertion order can't change identity). Instances
    whose class doesn't define a custom __repr__/__str__ (i.e. would otherwise
    print as an uninformative default `object.__repr__`, e.g. GroundTruth
    subclasses) are reconstructed as ``ClassName(k=v, ...)`` from their
    attributes. Anything else uses its own repr.
    """
    if isinstance(val, (str, int, float, bool, type(None))):
        return val
    if isinstance(val, type):
        return val.__name__
    if isinstance(val, dict):
        items = ', '.join(f'{k!r}: {format_config_value(v)!r}'
                          for k, v in sorted(val.items(), key=lambda kv: str(kv[0])))
        return '{' + items + '}'
    cls = type(val)
    if cls.__repr__ is object.__repr__ and cls.__str__ is object.__str__:
        attrs = ', '.join(f'{k}={v!r}' for k, v in vars(val).items())
        return f'{cls.__name__}({attrs})'
    return repr(val)


def flatten_config(run_config) -> dict:
    """Complete flattened field set of a RunConfig: every field of the data /
    model / train sub-configs, defaults included, as ``{sub_field: value}``."""
    out = {}
    for sub_name in ('data', 'model', 'train'):
        sub = getattr(run_config, sub_name)
        for f in fields(sub):
            out[f'{sub_name}_{f.name}'] = format_config_value(getattr(sub, f.name))
    return out


def config_identity(run_config) -> dict:
    """The flattened field set that defines a config's identity (see module
    docstring): everything except IDENTITY_EXCLUDE."""
    flat = flatten_config(run_config)
    for k in IDENTITY_EXCLUDE:
        flat.pop(k, None)
    return flat


def identity_hash(identity: dict) -> str:
    """Content-hash config id for a (already-flattened) identity dict."""
    canonical = json.dumps(identity, sort_keys=True)
    return 'cfg_' + hashlib.sha1(canonical.encode()).hexdigest()[:8]


def config_id_for(run_config) -> str:
    return identity_hash(config_identity(run_config))


def default_identity() -> dict:
    """Identity of an all-defaults RunConfig (used to show non-default fields)."""
    from ..config import get_config_class
    return config_identity(get_config_class('run')())


def describe_config(identity: dict, defaults: dict = None) -> str:
    """Short 'field=value' summary of the fields that differ from defaults."""
    defaults = defaults if defaults is not None else default_identity()
    diff = {k.split('_', 1)[1]: v for k, v in identity.items()
            if str(v) != str(defaults.get(k))}
    return ', '.join(f'{k}={v}' for k, v in diff.items()) or '(all defaults)'


# ----------------------------------------------------------------------------
# registry.json load/save (atomic)
# ----------------------------------------------------------------------------
def _empty_registry() -> dict:
    return {'schema_version': SCHEMA_VERSION, 'configs': {}, 'experiments': []}


def load_registry(root) -> dict:
    path = registry_path(root)
    if not os.path.exists(path):
        return _empty_registry()
    with open(path) as f:
        try:
            reg = json.load(f)
        except json.JSONDecodeError:
            return _empty_registry()
    reg.setdefault('schema_version', SCHEMA_VERSION)
    reg.setdefault('configs', {})
    reg.setdefault('experiments', [])
    return reg


def save_registry(root, reg, write_index=True):
    """Atomically persist registry.json and refresh the human-readable index."""
    os.makedirs(root, exist_ok=True)
    path = registry_path(root)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(reg, f, indent=2)
    os.replace(tmp, path)
    if write_index:
        _write_index_md(root, reg)


# ----------------------------------------------------------------------------
# Config + experiment registration
# ----------------------------------------------------------------------------
def get_or_register_config(root, run_config, name=None) -> str:
    """Return the global config id for run_config, registering it in the
    lookup table (with its full identity field set) on first sight."""
    identity = config_identity(run_config)
    cid = identity_hash(identity)
    reg = load_registry(root)
    if cid not in reg['configs']:
        reg['configs'][cid] = {
            'name': name or cid,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'fields': identity,
        }
        save_registry(root, reg)
    return cid


def register_experiment(root, *, name, ivs, ordinal_ivs, trials, global_seed,
                        config_ids, results):
    """Upsert an experiment entry (keyed by name) and refresh index.md.

    ``results`` should be this experiment's result rows (for headline metrics).
    """
    reg = load_registry(root)
    reg['experiments'] = [e for e in reg['experiments'] if e.get('name') != name]
    entry = {
        'name': name,
        'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ivs': {k: [str(format_config_value(v)) for v in vals]
                for k, vals in (ivs or {}).items()},
        'ordinal_ivs': list(ordinal_ivs or []),
        'trials': trials,
        'global_seed': global_seed,
        'config_ids': list(config_ids or []),
        'n_configs': len(set(config_ids or [])),
        'n_rows': len(results or []),
        'metrics': _headline_metrics(results or []),
    }
    reg['experiments'].append(entry)
    reg['experiments'].sort(key=lambda e: e.get('updated', ''), reverse=True)
    save_registry(root, reg)
    return entry


def get_experiment(root, name):
    """The registry entry for an experiment name, or None."""
    for e in load_registry(root)['experiments']:
        if e.get('name') == name:
            return e
    return None


def union_ivs(root):
    """Union of IV keys/values and ordinal IVs across all experiment entries
    (used for pooled statistics over the whole store)."""
    ivs, ordinal = {}, set()
    for e in load_registry(root)['experiments']:
        for k, vals in e.get('ivs', {}).items():
            merged = ivs.setdefault(k, [])
            for v in vals:
                if v not in merged:
                    merged.append(v)
        ordinal.update(e.get('ordinal_ivs', []))
    return ivs, sorted(ordinal)


# ----------------------------------------------------------------------------
# Manifests (pickled Experiment state for load/extend)
# ----------------------------------------------------------------------------
def load_manifests(root) -> dict:
    path = manifests_path(root)
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_manifest(root, name, manifest):
    """Read-modify-write one experiment's manifest into manifests.pkl."""
    os.makedirs(root, exist_ok=True)
    manifests = load_manifests(root)
    manifests[name] = manifest
    tmp = manifests_path(root) + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(manifests, f)
    os.replace(tmp, manifests_path(root))


# ----------------------------------------------------------------------------
# Result rows: canonical results.csv + append-only runs.csv
# ----------------------------------------------------------------------------
def _coerce(row: dict) -> dict:
    """Coerce a raw CSV string row into typed values for statistics."""
    out = {}
    for k, v in row.items():
        if k in _FLOAT_COLS:
            out[k] = float(v) if v not in ('', 'None', None) else np.nan
        elif k in _INT_COLS:
            out[k] = int(float(v)) if v not in ('', 'None', None) else None
        else:
            out[k] = v
    return out


def read_results_csv(path: str) -> list:
    """Read a results.csv into typed row dicts (empty if absent)."""
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return [_coerce(r) for r in csv.DictReader(f)]


def load_results(root, filter_fn=None) -> list:
    """All canonical result rows in the store (optionally filtered)."""
    rows = read_results_csv(results_csv_path(root))
    if filter_fn is not None:
        rows = [r for r in rows if filter_fn(r)]
    return rows


def append_run(root, row: dict):
    """Append one run record to the append-only runs.csv log."""
    os.makedirs(root, exist_ok=True)
    path = runs_csv_path(root)
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_FIELDS, extrasaction='ignore')
        if new:
            writer.writeheader()
        writer.writerow({'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                         **row})


def _headline_metrics(results: list) -> dict:
    """Min / mean best- and final-test-loss over successful rows."""
    def stat(vals, fn):
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(fn(vals)) if vals else None

    return {
        'n_success': sum(1 for r in results if r.get('final_test_loss') not in (None, '')
                         and not (isinstance(r.get('final_test_loss'), float) and np.isnan(r['final_test_loss']))),
        'min_best_test_loss': stat([r.get('best_test_loss') for r in results], min),
        'mean_best_test_loss': stat([r.get('best_test_loss') for r in results], np.mean),
        'min_final_test_loss': stat([r.get('final_test_loss') for r in results], min),
        'mean_final_test_loss': stat([r.get('final_test_loss') for r in results], np.mean),
    }


# ----------------------------------------------------------------------------
# Rebuild (disaster recovery): reconstruct the lookup from config dirs
# ----------------------------------------------------------------------------
def rebuild_configs(root) -> dict:
    """Reconstruct registry.json's config lookup from the per-config folders.

    Prefers each folder's config.pkl (exact identity recomputed from the
    RunConfig); falls back to config.json minus IDENTITY_EXCLUDE. Existing
    entries' names/created stamps are preserved. Returns the new registry."""
    reg = load_registry(root)
    old = reg['configs']
    rebuilt = {}
    for entry in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        cdir = os.path.join(root, entry)
        if not (entry.startswith('cfg_') and os.path.isdir(cdir)):
            continue
        identity = None
        pkl = os.path.join(cdir, 'config.pkl')
        js = os.path.join(cdir, 'config.json')
        if os.path.exists(pkl):
            try:
                with open(pkl, 'rb') as f:
                    identity = config_identity(pickle.load(f))
            except Exception as e:
                print(f"⊘ Warning: could not unpickle {pkl}: {e}")
        if identity is None and os.path.exists(js):
            with open(js) as f:
                identity = {k: v for k, v in json.load(f).items()
                            if k not in IDENTITY_EXCLUDE}
        if identity is None:
            print(f"⊘ Warning: no config.pkl/json in {cdir}; skipped")
            continue
        cid = identity_hash(identity)
        if cid != entry:
            print(f"⚠ {entry}: recomputed id is {cid} (schema drift?); "
                  f"keeping folder name — run migrate_registry.py to reconcile")
            cid = entry
        prev = old.get(cid, {})
        rebuilt[cid] = {
            'name': prev.get('name', cid),
            'created': prev.get('created',
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            'fields': identity,
        }
    reg['configs'] = rebuilt
    save_registry(root, reg)
    return reg


# ----------------------------------------------------------------------------
# index.md
# ----------------------------------------------------------------------------
def _fmt(v, spec='.6f'):
    return format(v, spec) if isinstance(v, (int, float)) and v is not None else '—'


def _write_index_md(root, reg):
    """Render a human-readable index: configs lookup table + experiments."""
    rows = read_results_csv(results_csv_path(root))
    by_config = {}
    for r in rows:
        by_config.setdefault(r.get('config_id'), []).append(r)
    try:
        defaults = default_identity()
    except Exception:
        defaults = {}

    lines = ["# Results Store Index\n\n",
             f"_Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}; "
             f"{len(reg['configs'])} configs, {len(reg['experiments'])} experiments, "
             f"{len(rows)} result rows._\n\n"]

    lines += ["## Configs\n\n",
              "| Config ID | Name | Non-default fields | Rows | Mean Best Test | Min Best Test | Created |\n",
              "|-----------|------|--------------------|------|----------------|---------------|--------|\n"]
    for cid in sorted(reg['configs']):
        e = reg['configs'][cid]
        m = _headline_metrics(by_config.get(cid, []))
        lines.append(
            f"| `{cid}` | {e.get('name', cid)} | {describe_config(e.get('fields', {}), defaults)} | "
            f"{len(by_config.get(cid, []))} | {_fmt(m['mean_best_test_loss'])} | "
            f"{_fmt(m['min_best_test_loss'])} | {e.get('created', '')} |\n")

    lines += ["\n## Experiments\n\n",
              "| Experiment | Configs | Trials | Rows | Mean Best Test | Min Best Test | IVs | Updated |\n",
              "|------------|---------|--------|------|----------------|---------------|-----|---------|\n"]
    for e in reg['experiments']:
        m = e.get('metrics', {})
        iv_str = ", ".join(f"{k}={'/'.join(map(str, v))}" for k, v in e.get('ivs', {}).items()) or "—"
        lines.append(
            f"| `{e['name']}` | {e.get('n_configs', '?')} | {e.get('trials', '?')} | "
            f"{e.get('n_rows', '?')} | {_fmt(m.get('mean_best_test_loss'))} | "
            f"{_fmt(m.get('min_best_test_loss'))} | {iv_str} | {e.get('updated', '')} |\n")

    with open(index_path(root), 'w') as f:
        f.write("".join(lines))
