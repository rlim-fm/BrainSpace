"""Migrate the results store after a schema change, re-hashing config ids.

Config ids are content hashes of each config's full field set, so renaming a
config field, renaming an architecture/setting value, or adding a new config
field changes the canonical form. This script rewrites everything that encodes
the old schema — the registry lookup, cfg_* folder names, results.csv/runs.csv
columns and values, per-config config.json/config.pkl, and manifests.pkl — in
one pass, printing the old→new id remap.

Operations (combinable; all support --dry-run):

  --rename-field train.cot_sup=cot
      A config dataclass field was renamed in code. Rewrites lookup keys,
      CSV columns (config_train_cot_sup → config_train_cot), config.json keys,
      re-pickles config.pkl against the current schema, and re-hashes ids.
      The sub-config scope ('data.', 'model.', 'train.') is optional.

  --rename-value arch:OldClass=NewClass
      A *value* was renamed (e.g. a model class). Rewrites the stored value in
      the lookup and CSVs, re-hashes ids, and unpickles old config.pkl /
      manifests.pkl through a shim that maps the old class name to the new one.

  --fill-field model.new_field=default
      A new field was added to a config dataclass. Backfills the given value
      into every stored entry that lacks it and re-hashes ids. (config.pkl is
      re-pickled against the current schema, which fills real defaults.)

Examples:
    python migrate_registry.py --rename-field train.cot_sup=cot --dry-run
    python migrate_registry.py --rename-value arch:NaiveTRNN=NaiveTropicalRNN
    python migrate_registry.py --fill-field model.causal=True --results-root sandbox/my_test
"""
import argparse
import csv
import io
import json
import os
import pickle
from dataclasses import fields

from .internal import registry
from .experiment import _coerce_value


# ----------------------------------------------------------------------------
# Spec parsing
# ----------------------------------------------------------------------------
def parse_field_rename(spec):
    """'train.cot_sup=cot' → (scope, old, new); scope may be None."""
    lhs, _, new = spec.partition('=')
    if not new:
        raise SystemExit(f"--rename-field expects [scope.]old=new (got '{spec}')")
    scope, _, old = lhs.rpartition('.')
    return (scope or None, old.strip(), new.strip())


def parse_value_rename(spec):
    """'arch:Old=New' → (field, old, new)."""
    field, _, rest = spec.partition(':')
    old, _, new = rest.partition('=')
    if not (field and old and new):
        raise SystemExit(f"--rename-value expects field:old=new (got '{spec}')")
    return (field.strip(), old.strip(), new.strip())


def parse_fill_field(spec):
    """'model.new_field=0.5' → (flat_key, coerced_value)."""
    lhs, _, val = spec.partition('=')
    if not val or '.' not in lhs:
        raise SystemExit(f"--fill-field expects scope.field=value (got '{spec}')")
    scope, _, name = lhs.rpartition('.')
    if scope not in ('data', 'model', 'train'):
        raise SystemExit(f"--fill-field scope must be data/model/train (got '{scope}')")
    return (f'{scope}_{name.strip()}', _coerce_value(val))


# ----------------------------------------------------------------------------
# Transformations
# ----------------------------------------------------------------------------
def _flat_key_matches(flat_key, scope, name):
    """Does a flattened key like 'train_cot_sup' match a (scope, field) spec?"""
    if scope is not None:
        return flat_key == f'{scope}_{name}'
    return any(flat_key == f'{sub}_{name}' for sub in ('data', 'model', 'train'))


def transform_identity(identity, field_renames, value_renames, fill_fields):
    """Apply all operations to one flattened identity/fields dict."""
    out = {}
    for k, v in identity.items():
        nk = k
        for scope, old, new in field_renames:
            if _flat_key_matches(k, scope, old):
                # Preserve the actual sub prefix of the matched key.
                for s in ('data', 'model', 'train'):
                    if k == f'{s}_{old}':
                        nk = f'{s}_{new}'
                        break
        nv = v
        for fname, old, new in value_renames:
            if _flat_key_matches(nk, None, fname) and str(v) == old:
                nv = new
        out[nk] = nv
    for flat_key, val in fill_fields:
        out.setdefault(flat_key, registry.format_config_value(val))
    return out


def transform_csv_header(col, field_renames):
    """Rename 'config_<sub>_<field>' columns per the field renames."""
    for scope, old, new in field_renames:
        subs = (scope,) if scope else ('data', 'model', 'train')
        for s in subs:
            if col == f'config_{s}_{old}':
                return f'config_{s}_{new}'
    return col


def rewrite_csv(path, field_renames, value_renames, id_remap):
    """Rewrite one CSV: renamed columns, renamed values, remapped config ids."""
    if not os.path.exists(path):
        return
    with open(path, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return
    header = [transform_csv_header(c, field_renames) for c in rows[0]]

    # Columns affected by each value rename (post-header-rename names).
    value_cols = {}
    for fname, old, new in value_renames:
        for i, col in enumerate(header):
            if any(col == f'config_{s}_{fname}' for s in ('data', 'model', 'train')):
                value_cols.setdefault(i, {})[old] = new
    id_col = header.index('config_id') if 'config_id' in header else None

    out_rows = [header]
    for row in rows[1:]:
        row = list(row)
        for i, mapping in value_cols.items():
            if i < len(row) and row[i] in mapping:
                row[i] = mapping[row[i]]
        if id_col is not None and id_col < len(row):
            row[id_col] = id_remap.get(row[id_col], row[id_col])
        out_rows.append(row)

    buf = io.StringIO()
    csv.writer(buf).writerows(out_rows)
    with open(path, 'w', newline='') as f:
        f.write(buf.getvalue())


# ----------------------------------------------------------------------------
# Pickle handling (schema-tolerant reload of RunConfigs)
# ----------------------------------------------------------------------------
class _RenamedClassUnpickler(pickle.Unpickler):
    """Unpickler that maps renamed class names to their current classes."""

    def __init__(self, f, class_renames):
        super().__init__(f)
        self._class_renames = class_renames

    def find_class(self, module, name):
        name = self._class_renames.get(name, name)
        return super().find_class(module, name)


def load_pickle_tolerant(path, class_renames):
    with open(path, 'rb') as f:
        return _RenamedClassUnpickler(f, class_renames).load()


def rebuild_run_config(old_rc, field_renames):
    """Reconstruct a RunConfig against the *current* dataclass schema.

    Old pickles restore raw __dict__ state, so renamed attrs keep their old
    names and newly-added fields are absent. This maps old attr names to new
    ones and lets current dataclass defaults fill anything missing."""
    from .config import RunConfig, DatasetConfig, ModelConfig, TrainConfig
    subs = {}
    for sub_name, cls in (('data', DatasetConfig), ('model', ModelConfig),
                          ('train', TrainConfig)):
        old_vars = dict(vars(getattr(old_rc, sub_name)))
        for scope, old, new in field_renames:
            if scope in (None, sub_name) and old in old_vars and new not in old_vars:
                old_vars[new] = old_vars.pop(old)
        kwargs = {f.name: old_vars[f.name] for f in fields(cls) if f.name in old_vars}
        subs[sub_name] = cls(**kwargs)
    return RunConfig(**subs)


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-root', default='results',
                        help="Results store root (default: results).")
    parser.add_argument('--rename-field', action='append', default=[],
                        help="[scope.]old=new (repeatable), e.g. train.cot_sup=cot")
    parser.add_argument('--rename-value', action='append', default=[],
                        help="field:old=new (repeatable), e.g. arch:OldClass=NewClass")
    parser.add_argument('--fill-field', action='append', default=[],
                        help="scope.field=value (repeatable), e.g. model.causal=True")
    parser.add_argument('--dry-run', action='store_true',
                        help="Print the planned id remap and affected files; write nothing.")
    args = parser.parse_args()

    field_renames = [parse_field_rename(s) for s in args.rename_field]
    value_renames = [parse_value_rename(s) for s in args.rename_value]
    fill_fields = [parse_fill_field(s) for s in args.fill_field]
    if not (field_renames or value_renames or fill_fields):
        parser.error("Nothing to do: pass --rename-field / --rename-value / --fill-field.")

    root = args.results_root
    reg = registry.load_registry(root)
    if not reg['configs']:
        raise SystemExit(f"No configs registered in '{root}'.")

    # Class-name renames for unpickling (value renames on class-typed fields).
    class_renames = {old: new for _f, old, new in value_renames}

    # ------------------------------------------------------------------
    # Plan: transform every lookup entry and compute the id remap.
    # ------------------------------------------------------------------
    new_configs, id_remap = {}, {}
    for cid, entry in reg['configs'].items():
        new_fields = transform_identity(entry.get('fields', {}), field_renames,
                                        value_renames, fill_fields)
        new_cid = registry.identity_hash(new_fields)
        id_remap[cid] = new_cid
        if new_cid in new_configs:
            raise SystemExit(f"Collision: {cid} and another entry both map to {new_cid}; "
                             f"aborting (no changes written).")
        new_configs[new_cid] = {**entry, 'fields': new_fields}

    changed = {o: n for o, n in id_remap.items() if o != n}
    print(f"Store: '{root}' — {len(reg['configs'])} configs, "
          f"{len(changed)} id(s) change:\n")
    print(f"  {'OLD':<14} {'NEW':<14}")
    for old, new in sorted(id_remap.items()):
        marker = '→' if old != new else '='
        print(f"  {old:<14} {marker} {new:<14}")

    if args.dry_run:
        print("\n(dry run — nothing written; also would rewrite results.csv, runs.csv, "
              "config.json/.pkl in each folder, manifests.pkl, registry.json, index.md)")
        return

    # ------------------------------------------------------------------
    # Apply: folders → per-config files → CSVs → manifests → registry.
    # ------------------------------------------------------------------
    for old, new in changed.items():
        old_dir, new_dir = registry.config_dir(root, old), registry.config_dir(root, new)
        if os.path.isdir(old_dir):
            if os.path.exists(new_dir):
                raise SystemExit(f"Cannot rename {old_dir} → {new_dir}: target exists.")
            os.rename(old_dir, new_dir)
            print(f"✓ renamed {old} → {new}")

    for new_cid, entry in new_configs.items():
        cdir = registry.config_dir(root, new_cid)
        pkl = os.path.join(cdir, 'config.pkl')
        if os.path.exists(pkl):
            try:
                rc = rebuild_run_config(load_pickle_tolerant(pkl, class_renames),
                                        field_renames)
                with open(pkl, 'wb') as f:
                    pickle.dump(rc, f)
                with open(os.path.join(cdir, 'config.json'), 'w') as f:
                    json.dump(registry.flatten_config(rc), f, indent=2)
                recomputed = registry.config_id_for(rc)
                if recomputed != new_cid:
                    print(f"⚠ {new_cid}: identity recomputed from config.pkl is "
                          f"{recomputed}; lookup entry kept as source of truth")
            except Exception as e:
                print(f"⚠ {new_cid}: could not rewrite config.pkl/json ({e})")
        elif os.path.isdir(cdir):
            js = os.path.join(cdir, 'config.json')
            if os.path.exists(js):
                with open(js) as f:
                    flat = json.load(f)
                with open(js, 'w') as f:
                    json.dump(transform_identity(flat, field_renames, value_renames,
                                                 fill_fields), f, indent=2)

    rewrite_csv(registry.results_csv_path(root), field_renames, value_renames, id_remap)
    rewrite_csv(registry.runs_csv_path(root), field_renames, value_renames, id_remap)

    # Manifests: remap config_ids, rename IV keys, re-pickle base_configs.
    manifests = {}
    if os.path.exists(registry.manifests_path(root)):
        manifests = load_pickle_tolerant(registry.manifests_path(root), class_renames)
        for name, m in manifests.items():
            try:
                m['base_config'] = rebuild_run_config(m['base_config'], field_renames)
            except Exception as e:
                print(f"⚠ manifest '{name}': base_config not rebuilt ({e})")
            key_map = {old: new for scope, old, new in field_renames}
            m['ivs'] = {key_map.get(k, k): v for k, v in m.get('ivs', {}).items()}
            m['ordinal_ivs'] = [key_map.get(k, k) for k in m.get('ordinal_ivs', [])]
            m['config_iv_values'] = [{key_map.get(k, k): v for k, v in iv.items()}
                                     for iv in m.get('config_iv_values', [])]
            m['config_ids'] = [id_remap.get(c, c) for c in m.get('config_ids', [])]
        tmp = registry.manifests_path(root) + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(manifests, f)
        os.replace(tmp, registry.manifests_path(root))

    # Registry: new config lookup + remapped experiment entries.
    key_map = {old: new for scope, old, new in field_renames}
    reg['configs'] = new_configs
    for e in reg['experiments']:
        e['config_ids'] = [id_remap.get(c, c) for c in e.get('config_ids', [])]
        e['ivs'] = {key_map.get(k, k):
                    [dict((vr[1], vr[2]) for vr in value_renames).get(v, v) for v in vals]
                    for k, vals in e.get('ivs', {}).items()}
        e['ordinal_ivs'] = [key_map.get(k, k) for k in e.get('ordinal_ivs', [])]
    registry.save_registry(root, reg)

    # Regenerate pooled summary from the rewritten CSV.
    from .refresh_results import refresh_store_files
    refresh_store_files(root)

    print(f"\n✓ Migration complete: {len(changed)} folder(s) renamed, CSVs/registry/"
          f"manifests rewritten, index.md + summary.md refreshed.")


if __name__ == '__main__':
    main()
