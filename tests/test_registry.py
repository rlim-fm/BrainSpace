"""Invariants for the flat results store: config identity, lookup registration,
run logging, cell-level caching, and schema-migration transforms."""
import math
import os

import pytest
import torch.nn as nn
import torch.optim as optim

from brainspace.internal import registry
from brainspace.config import RunConfig, DatasetConfig, ModelConfig, TrainConfig
from brainspace.datasets import cumsum
from brainspace.experiment import Experiment, write_results_csv
from brainspace.models import LSTM, GRU


def make_config(epochs=2, hidden_dim=8, seed=42, device=None):
    return RunConfig(
        data=DatasetConfig(ground_truth=cumsum(), data_dim=(6, 1), N=32),
        model=ModelConfig(arch=LSTM, hidden_dim=hidden_dim, num_layers=1,
                          pool=True, seq2seq=False),
        train=TrainConfig(epochs=epochs, criterion=nn.MSELoss(), seed=seed,
                          device=device, optimizer_cls=optim.AdamW, lr=1e-2),
    )


# ----------------------------------------------------------------------------
# Config identity
# ----------------------------------------------------------------------------

def test_config_id_deterministic_for_equal_configs():
    assert registry.config_id_for(make_config()) == registry.config_id_for(make_config())


def test_config_id_ignores_seed_and_device():
    base = registry.config_id_for(make_config(seed=42, device=None))
    assert registry.config_id_for(make_config(seed=7, device=None)) == base
    assert registry.config_id_for(make_config(seed=42, device="cpu")) == base


def test_config_id_changes_with_any_other_field():
    base = registry.config_id_for(make_config())
    assert registry.config_id_for(make_config(hidden_dim=16)) != base
    assert registry.config_id_for(make_config(epochs=3)) != base


def test_identity_covers_every_field_except_excluded():
    identity = registry.config_identity(make_config())
    flat = registry.flatten_config(make_config())
    assert set(identity) == set(flat) - set(registry.IDENTITY_EXCLUDE)


def test_format_config_value_dict_key_order_canonical():
    a = registry.format_config_value({'step_size': 200, 'gamma': 0.5})
    b = registry.format_config_value({'gamma': 0.5, 'step_size': 200})
    assert a == b


# Golden identity of an all-defaults RunConfig. If this test fails, the
# content-hash of EVERY stored config in EVERY domain store (Tropical-RNN,
# ClusterGAR, ...) has changed: existing cfg_ folders and result rows no longer
# match their recomputed ids. Either revert the change that broke it, or — if
# the schema change is intentional — run migrate_registry.py (--dry-run first)
# in each domain store and update the pinned values here.
GOLDEN_DEFAULT_ID = 'cfg_7de1b062'
GOLDEN_DEFAULT_IDENTITY = {
    'data_N': 2048,
    'data_data_dim': '(10, 1)',
    'data_ground_truth': 'CumulativeSum()',
    'data_min_seq_len': 5,
    'data_use_padding': False,
    'data_x_range': '(-8, 8)',
    'model_arch': 'LSTM',
    'model_arch_kwargs': '{}',
    'model_causal': True,
    'model_dropout': 0.0,
    'model_hidden_dim': 64,
    'model_num_layers': 2,
    'model_output_dim': 1,
    'model_pool': False,
    'model_seq2seq': True,
    'train_batch_size': None,
    'train_cot': False,
    'train_criterion': 'MSELoss()',
    'train_dtype': 'torch.float32',
    'train_epochs': 1000,
    'train_lr': 0.001,
    'train_optimizer_cls': 'AdamW',
    'train_scheduler_cls': None,
    'train_scheduler_kwargs': '{}',
    'train_weight_decay': 0.0,
}


def test_golden_identity_of_default_config():
    assert registry.config_identity(RunConfig()) == GOLDEN_DEFAULT_IDENTITY
    assert registry.config_id_for(RunConfig()) == GOLDEN_DEFAULT_ID


def test_golden_id_of_non_default_config():
    assert registry.config_id_for(make_config()) == 'cfg_b5e31228'


# ----------------------------------------------------------------------------
# Lookup registration
# ----------------------------------------------------------------------------

def test_get_or_register_config_idempotent(tmp_path):
    root = str(tmp_path / "store")
    cid1 = registry.get_or_register_config(root, make_config(), name="first")
    cid2 = registry.get_or_register_config(root, make_config(), name="second")
    assert cid1 == cid2
    reg = registry.load_registry(root)
    assert len(reg['configs']) == 1
    assert reg['configs'][cid1]['name'] == "first"  # first name wins


def test_new_iv_value_registers_new_id_without_touching_existing(tmp_path):
    root = str(tmp_path / "store")
    cid1 = registry.get_or_register_config(root, make_config(hidden_dim=8))
    entry1 = dict(registry.load_registry(root)['configs'][cid1])
    cid2 = registry.get_or_register_config(root, make_config(hidden_dim=16))
    reg = registry.load_registry(root)
    assert cid2 != cid1
    assert reg['configs'][cid1] == entry1  # untouched
    assert set(reg['configs']) == {cid1, cid2}


def test_append_run_is_append_only(tmp_path):
    root = str(tmp_path / "store")
    registry.append_run(root, {'experiment': 'e', 'config_id': 'cfg_x',
                               'trial_idx': 0, 'seed': 1, 'status': 'ok'})
    registry.append_run(root, {'experiment': 'e', 'config_id': 'cfg_x',
                               'trial_idx': 0, 'seed': 1, 'status': 'ok'})
    with open(registry.runs_csv_path(root)) as f:
        lines = f.readlines()
    assert len(lines) == 3  # header + 2 rows, duplicates preserved


# ----------------------------------------------------------------------------
# Store-level behavior of Experiment (tiny real training)
# ----------------------------------------------------------------------------

def run_lines(root):
    with open(registry.runs_csv_path(root)) as f:
        return len(f.readlines()) - 1


@pytest.mark.slow
def test_store_caching_merge_and_isolation(tmp_path):
    root = str(tmp_path / "store")

    exp = Experiment(make_config(), ivs={"hidden_dim": [8, 16]}, name="a",
                     trials=1, global_seed=42, results_root=root)
    exp.run_grid(visualize=False)
    assert len(registry.load_results(root)) == 2
    assert run_lines(root) == 2
    assert all(r['config_id'].startswith('cfg_') for r in registry.load_results(root))

    # Same grid under another experiment name: fully cached, nothing retrained,
    # results.csv stays keyed (config_id, seed) with no duplicates.
    exp2 = Experiment(make_config(), ivs={"hidden_dim": [8, 16]}, name="b",
                      trials=1, global_seed=42, results_root=root)
    exp2.run_grid(visualize=False)
    assert run_lines(root) == 2
    assert len(registry.load_results(root)) == 2

    # force=True retrains and logs new runs but keeps results.csv idempotent.
    exp3 = Experiment(make_config(), ivs={"hidden_dim": [8, 16]}, name="a",
                      trials=1, global_seed=42, results_root=root)
    exp3.run_grid(visualize=False, force=True)
    assert run_lines(root) == 4
    assert len(registry.load_results(root)) == 2

    # Isolation: everything lives under the given root.
    assert os.path.exists(registry.registry_path(root))
    assert os.path.exists(registry.manifests_path(root))
    cfg_dirs = [d for d in os.listdir(root) if d.startswith('cfg_')]
    assert len(cfg_dirs) == 2


@pytest.mark.slow
def test_experiment_load_and_extend_reuses_store(tmp_path):
    root = str(tmp_path / "store")
    exp = Experiment(make_config(), ivs={"hidden_dim": [8]}, name="a",
                     trials=1, global_seed=42, results_root=root)
    exp.run_grid(visualize=False)

    loaded = Experiment.load("a", results_root=root)
    assert loaded.config_ids == exp.config_ids
    assert len(loaded.results) == 1

    loaded.extend(trials=2, visualize=False)
    assert len(registry.load_results(root)) == 2
    assert run_lines(root) == 2  # only the missing trial ran


# ----------------------------------------------------------------------------
# Migration transforms (pure logic; end-to-end covered by migrate_registry.py)
# ----------------------------------------------------------------------------

def test_migrate_rename_field_rehashes_to_current_schema_hash():
    """Renaming a stored old-schema field must yield exactly the hash the
    current schema would produce for the same values."""
    from brainspace.migrate_registry import transform_identity
    current = registry.config_identity(make_config())
    old = {('train_lrate' if k == 'train_lr' else k): v for k, v in current.items()}
    migrated = transform_identity(old, [('train', 'lrate', 'lr')], [], [])
    assert migrated == current
    assert registry.identity_hash(migrated) == registry.identity_hash(current)


def test_migrate_rename_value_and_fill_field():
    from brainspace.migrate_registry import transform_identity
    identity = {'model_arch': 'LSTM', 'model_hidden_dim': 8}
    out = transform_identity(identity, [], [('arch', 'LSTM', 'GRU')],
                             [('train_extra', 1)])
    assert out['model_arch'] == 'GRU'
    assert out['train_extra'] == 1
    assert out['model_hidden_dim'] == 8


def test_migrate_rebuild_run_config_maps_old_attr_names():
    from brainspace.migrate_registry import rebuild_run_config
    cfg = make_config()
    # Simulate an old pickle: the attribute still has its pre-rename name.
    vars(cfg.train)['lrate'] = vars(cfg.train).pop('lr')
    rebuilt = rebuild_run_config(cfg, [('train', 'lrate', 'lr')])
    assert rebuilt.train.lr == 1e-2
    assert registry.config_id_for(rebuilt) == registry.config_id_for(make_config())


def test_migrate_csv_header_transform():
    from brainspace.migrate_registry import transform_csv_header
    renames = [('train', 'cot_sup', 'cot'), (None, 'archaic', 'modern')]
    assert transform_csv_header('config_train_cot_sup', renames) == 'config_train_cot'
    assert transform_csv_header('config_model_archaic', renames) == 'config_model_modern'
    assert transform_csv_header('final_test_loss', renames) == 'final_test_loss'
