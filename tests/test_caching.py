"""Invariants for the cross-run caching + extension system of the flat store.

The store caches at the granularity of a *cell* — a ``(config_id, seed)`` pair.
A cell that already has a successful row (from *any* experiment) is never
retrained: its stored row is adopted verbatim. ``force=True`` bypasses this.

These tests pin the behaviors most likely to regress under refactors:
  - a cell is reused regardless of experiment name, visualization sampling, or
    anything else that is *not* part of config identity;
  - epochs *is* part of config identity, so different epoch counts are distinct
    cells that must each train;
  - ``extend`` adds trials / configs (incl. a previously-undefined IV such as a
    new architecture) while running only what is genuinely missing.

Cell-eligibility logic (which rows count as cached) is fast-unit-tested; the
end-to-end caching behavior trains tiny models and is marked ``slow``.
"""
import numpy as np
import pytest
import torch.nn as nn
import torch.optim as optim

from brainspace.internal import registry
from brainspace.config import RunConfig, DatasetConfig, ModelConfig, TrainConfig
from brainspace.datasets import cumsum
from brainspace.experiment import Experiment, write_results_csv
from brainspace.models import LSTM, GRU


def make_config(epochs=2, hidden_dim=8, arch=LSTM):
    return RunConfig(
        data=DatasetConfig(ground_truth=cumsum(), data_dim=(6, 1), N=32),
        model=ModelConfig(arch=arch, hidden_dim=hidden_dim, num_layers=1,
                          pool=True, seq2seq=False),
        train=TrainConfig(epochs=epochs, criterion=nn.MSELoss(), seed=42,
                          optimizer_cls=optim.AdamW, lr=1e-2),
    )


def run_lines(root):
    """Number of data rows in the append-only runs.csv (excludes header)."""
    with open(registry.runs_csv_path(root)) as f:
        return len(f.readlines()) - 1


def rows_by_cell(root):
    """{(config_id, seed): row} for every canonical result row in the store."""
    return {(r['config_id'], r['seed']): r for r in registry.load_results(root)}


# ============================================================================
# Cell-eligibility (fast: no training, just the cache index)
# ============================================================================

def test_only_successful_rows_are_cacheable(tmp_path):
    """_load_store_cache indexes only rows with a real (non-NaN) test loss, so
    failed / missing cells are retried rather than adopted as 'done'."""
    root = str(tmp_path / "store")
    write_results_csv([
        {'config_id': 'cfg_ok', 'seed': 1, 'trial_idx': 0, 'final_test_loss': 0.5},
        {'config_id': 'cfg_nan', 'seed': 2, 'trial_idx': 0, 'final_test_loss': ''},
    ], registry.results_csv_path(root))

    exp = Experiment(make_config(), results_root=root)
    exp._load_store_cache()

    assert ('cfg_ok', 1) in exp._store_cache
    assert ('cfg_nan', 2) not in exp._store_cache  # NaN loss => not cached


def test_cache_key_is_config_id_and_seed(tmp_path):
    """The same config at a different seed is a distinct, independently-cached
    cell (seed is deliberately excluded from config identity)."""
    root = str(tmp_path / "store")
    write_results_csv([
        {'config_id': 'cfg_a', 'seed': 1, 'trial_idx': 0, 'final_test_loss': 0.5},
        {'config_id': 'cfg_a', 'seed': 2, 'trial_idx': 1, 'final_test_loss': 0.7},
    ], registry.results_csv_path(root))

    exp = Experiment(make_config(), results_root=root)
    exp._load_store_cache()
    assert set(exp._store_cache) == {('cfg_a', 1), ('cfg_a', 2)}


# ============================================================================
# Cross-run cache reuse (slow: tiny real training)
# ============================================================================

@pytest.mark.slow
def test_rerun_reuses_cache_and_preserves_values(tmp_path):
    """Re-running the same grid under a different experiment name retrains
    nothing and adopts the stored loss values byte-for-byte."""
    root = str(tmp_path / "store")

    exp1 = Experiment(make_config(), ivs={"hidden_dim": [8, 16]}, name="first",
                      trials=1, global_seed=42, results_root=root)
    exp1.run_grid(visualize=False)
    baseline = {k: v['final_test_loss'] for k, v in rows_by_cell(root).items()}
    assert run_lines(root) == 2

    exp2 = Experiment(make_config(), ivs={"hidden_dim": [8, 16]}, name="second",
                      trials=1, global_seed=42, results_root=root)
    exp2.run_grid(visualize=False)

    # No new training, no duplicate canonical rows, identical loss values.
    assert run_lines(root) == 2
    assert len(registry.load_results(root)) == 2
    assert {k: v['final_test_loss'] for k, v in rows_by_cell(root).items()} == baseline
    # The reused rows are relabeled to the new experiment in-memory.
    assert all(r['experiment'] == "second" for r in exp2.results)


@pytest.mark.slow
def test_force_bypasses_cache_but_keeps_results_idempotent(tmp_path):
    """force=True retrains cached cells (new runs logged) yet results.csv stays
    keyed by cell, so no duplicate canonical rows accumulate."""
    root = str(tmp_path / "store")
    common = dict(ivs={"hidden_dim": [8]}, trials=1, global_seed=42, results_root=root)

    Experiment(make_config(), name="a", **common).run_grid(visualize=False)
    assert run_lines(root) == 1

    Experiment(make_config(), name="a", **common).run_grid(visualize=False, force=True)
    assert run_lines(root) == 2                        # a second run was logged
    assert len(registry.load_results(root)) == 1       # still one canonical row


# ============================================================================
# Sampling levels: a visualization concern, orthogonal to caching
# ============================================================================

@pytest.mark.slow
def test_cache_independent_of_sampling_level(tmp_path):
    """Sampling controls how many animation frames are kept — it is not part of
    config identity, so re-running the identical grid at a *different* sampling
    still hits the cache and retrains nothing."""
    root = str(tmp_path / "store")

    exp1 = Experiment(make_config(), name="s1", trials=1, global_seed=42,
                      results_root=root)
    exp1.run_grid(visualize=True, sampling=1)
    assert run_lines(root) == 1

    exp2 = Experiment(make_config(), name="s2", trials=1, global_seed=42,
                      results_root=root)
    exp2.run_grid(visualize=True, sampling=5)  # coarser sampling, same config

    assert run_lines(root) == 1                     # nothing retrained
    assert len(registry.load_results(root)) == 1


# ============================================================================
# Epoch counts: epochs IS part of config identity => distinct cells
# ============================================================================

@pytest.mark.slow
def test_different_epochs_are_distinct_cells(tmp_path):
    """Two epoch counts mint two config ids; caching a run at one epoch count
    never satisfies the other, so only the genuinely-new epoch value trains."""
    root = str(tmp_path / "store")

    exp1 = Experiment(make_config(), ivs={"epochs": [2]}, name="e1",
                      trials=1, global_seed=42, results_root=root)
    exp1.run_grid(visualize=False)
    assert run_lines(root) == 1

    # epochs=2 is reused from the store; epochs=4 is a brand-new cell.
    exp2 = Experiment(make_config(), ivs={"epochs": [2, 4]}, name="e2",
                      trials=1, global_seed=42, results_root=root)
    exp2.run_grid(visualize=False)

    assert run_lines(root) == 2                       # exactly one new training
    results = registry.load_results(root)
    assert len({r['config_id'] for r in results}) == 2
    assert {r['epochs'] for r in results} == {2, 4}


# ============================================================================
# Extension: add trials / configs, running only what is missing
# ============================================================================

@pytest.mark.slow
def test_extend_trials_runs_only_missing_and_reuses_prior(tmp_path):
    """Extending trials 1 -> 3 trains only the two new trial rounds; the
    original trial-0 row is reused unchanged."""
    root = str(tmp_path / "store")
    exp = Experiment(make_config(), ivs={"hidden_dim": [8]}, name="a",
                     trials=1, global_seed=42, results_root=root)
    exp.run_grid(visualize=False)
    trial0 = rows_by_cell(root)
    (cid, seed0), = trial0.keys()
    loss0 = trial0[(cid, seed0)]['final_test_loss']

    Experiment.load("a", results_root=root).extend(trials=3, visualize=False)

    assert run_lines(root) == 3                       # 1 original + 2 new rounds
    results = registry.load_results(root)
    assert len(results) == 3
    assert all(r['config_id'] == cid for r in results)
    assert {r['seed'] for r in results} == {42, 43, 44}  # global_seed + trial_idx
    # Original cell survives extension with its value intact.
    assert rows_by_cell(root)[(cid, seed0)]['final_test_loss'] == loss0


@pytest.mark.slow
def test_extend_new_architecture_adds_config_without_retraining_existing(tmp_path):
    """Adding a previously-undefined IV value (a new architecture) via extend
    creates and trains only the new config; the existing architecture's rows
    are untouched."""
    root = str(tmp_path / "store")
    exp = Experiment(make_config(arch=LSTM), ivs={"arch": [LSTM]}, name="arch",
                     trials=1, global_seed=42, results_root=root)
    exp.run_grid(visualize=False)
    lstm_cell = rows_by_cell(root)
    assert run_lines(root) == 1

    loaded = Experiment.load("arch", results_root=root)
    loaded.extend(add_ivs={"arch": [GRU]}, visualize=False)

    assert run_lines(root) == 2                        # only GRU trained
    assert len(loaded.config_ids) == 2                 # LSTM + GRU configs
    results = registry.load_results(root)
    assert {r['config_model_arch'] for r in results} == {"LSTM", "GRU"}
    # The LSTM cell is byte-for-byte preserved across the extension.
    for cell, row in lstm_cell.items():
        assert rows_by_cell(root)[cell]['final_test_loss'] == row['final_test_loss']


@pytest.mark.slow
def test_extend_new_iv_value_adds_single_config(tmp_path):
    """Adding one new value to an existing IV runs exactly one new config."""
    root = str(tmp_path / "store")
    exp = Experiment(make_config(), ivs={"hidden_dim": [8]}, name="h",
                     trials=1, global_seed=42, results_root=root)
    exp.run_grid(visualize=False)

    Experiment.load("h", results_root=root).extend(
        add_ivs={"hidden_dim": [16]}, visualize=False)

    assert run_lines(root) == 2
    results = registry.load_results(root)
    assert {int(r['config_model_hidden_dim']) for r in results} == {8, 16}
    assert len({r['config_id'] for r in results}) == 2
