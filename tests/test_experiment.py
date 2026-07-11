"""Pure-logic invariants for the grid-search Experiment framework."""
import math

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from brainspace.internal import stats
from brainspace.config import RunConfig, DatasetConfig, ModelConfig, TrainConfig
from brainspace.datasets import cumsum
from brainspace.experiment import Experiment
from brainspace.models import LSTM


def base_config(epochs=2):
    return RunConfig(
        data=DatasetConfig(ground_truth=cumsum(), data_dim=(6, 1), N=32),
        model=ModelConfig(arch=LSTM, hidden_dim=8, num_layers=1, pool=True, seq2seq=False),
        train=TrainConfig(epochs=epochs, criterion=nn.MSELoss(), seed=42,
                          optimizer_cls=optim.AdamW, lr=1e-2),
    )


# ----------------------------------------------------------------------------
# _get_seed precedence
# ----------------------------------------------------------------------------

def test_get_seed_dict_mapping():
    exp = Experiment(base_config=base_config(), seed_mapping={(0, 1): 99}, trials=3)
    assert exp._get_seed(0, 1) == 99
    assert exp._get_seed(0, 2) == 2  # falls back to trial_idx default


def test_get_seed_callable():
    exp = Experiment(base_config=base_config(),
                     seed_mapping=lambda c, t: 1000 + c * 10 + t, trials=3)
    assert exp._get_seed(2, 1) == 1021


def test_get_seed_global_increment_formula():
    exp = Experiment(base_config=base_config(), global_seed=100, trials=5)
    # global_seed + trial_idx (config_idx does not affect seed, so a given
    # trial number shares its seed across all configs)
    assert exp._get_seed(0, 0) == 100
    assert exp._get_seed(1, 2) == 100 + 2
    assert exp._get_seed(1, 2) == exp._get_seed(0, 2)


def test_get_seed_default_fallback():
    exp = Experiment(base_config=base_config(), trials=3)
    assert exp._get_seed(4, 2) == 2


# ----------------------------------------------------------------------------
# _apply_iv
# ----------------------------------------------------------------------------

def test_apply_iv_simple_key_routes_to_subconfig():
    exp = Experiment(base_config=base_config())
    cfg = exp._apply_iv(base_config(), "hidden_dim", 128)
    assert cfg.model.hidden_dim == 128


def test_apply_iv_dot_notation():
    exp = Experiment(base_config=base_config())
    cfg = exp._apply_iv(base_config(), "train.epochs", 999)
    assert cfg.train.epochs == 999


def test_apply_iv_arch_kwargs_merge():
    exp = Experiment(base_config=base_config())
    start = base_config()
    start.model.arch_kwargs = {"d_model": 8}
    cfg = exp._apply_iv(start, "arch_kwargs", {"n_heads": 2})
    assert cfg.model.arch_kwargs == {"d_model": 8, "n_heads": 2}


def test_apply_iv_does_not_mutate_original():
    exp = Experiment(base_config=base_config())
    original = base_config()
    exp._apply_iv(original, "hidden_dim", 256)
    assert original.model.hidden_dim == 8  # unchanged


def test_apply_iv_unknown_key_raises():
    exp = Experiment(base_config=base_config())
    with pytest.raises(KeyError):
        exp._apply_iv(base_config(), "nonexistent_param", 1)


# ----------------------------------------------------------------------------
# _create_configs
# ----------------------------------------------------------------------------

def test_create_configs_cartesian_product_count():
    exp = Experiment(base_config=base_config(),
                     ivs={"hidden_dim": [16, 32, 64], "num_layers": [1, 2]})
    assert len(exp.configs) == 6


def test_create_configs_no_ivs_single_config():
    exp = Experiment(base_config=base_config())
    assert len(exp.configs) == 1


# ----------------------------------------------------------------------------
# End-to-end tiny grid
# ----------------------------------------------------------------------------

@pytest.mark.slow
def test_run_grid_collects_results(tmp_path):
    from brainspace.internal import registry

    root = str(tmp_path / "results")
    exp = Experiment(
        base_config=base_config(epochs=2),
        ivs={"hidden_dim": [8, 16]},
        name="grid_test",
        trials=1,
        global_seed=42,
        results_root=root,
    )
    exp.run_grid(visualize=False)
    assert len(exp.results) == 2  # 2 configs x 1 trial
    assert all("final_test_loss" in r for r in exp.results)
    # Flat-store contract: rows carry global config ids + experiment name, and
    # the store artifacts exist at the root.
    assert all(r["config_id"].startswith("cfg_") for r in exp.results)
    assert all(r["experiment"] == "grid_test" for r in exp.results)
    for artifact in ("registry.json", "index.md", "runs.csv", "results.csv",
                     "summary.md", "manifests.pkl"):
        assert (tmp_path / "results" / artifact).exists()
    assert registry.get_experiment(root, "grid_test") is not None


# ----------------------------------------------------------------------------
# compute_subset_test_losses: in_domain / domain_ood / len_ood partitioning
# ----------------------------------------------------------------------------

@pytest.mark.slow
def test_compute_subset_test_losses_partition_invariants():
    """in_domain must be exactly the complement of (domain_ood | len_ood) on
    the random-portion test set, 'all' must match the existing aggregate
    test_loss, and len_ood must be NaN for non-padded (fixed-length) data --
    regardless of the specific loss values, which is what makes this a
    persistent regression check rather than a one-off numeric assertion.
    """
    config = RunConfig(
        data=DatasetConfig(ground_truth=cumsum(), data_dim=(6, 1), N=32,
                            use_padding=True, min_seq_len=3, x_range=(-1, 1)),
        model=ModelConfig(arch=LSTM, hidden_dim=8, num_layers=1, pool=True, seq2seq=False),
        train=TrainConfig(epochs=2, criterion=nn.MSELoss(), seed=42,
                          optimizer_cls=optim.AdamW, lr=1e-2),
    )
    from brainspace.train import Processor
    processor = Processor.from_run_config(config)
    processor.run()

    n = processor._x_test_random_n
    lo, hi = processor.x_range
    mask_n = processor.test_mask[:n].bool()
    x_n = processor.x_test[:n]
    oob = ((x_n < lo) | (x_n > hi)) & mask_n.unsqueeze(-1)
    domain_ood = oob.any(dim=(1, 2))
    seq_lens = mask_n.long().sum(dim=1)
    len_ood = seq_lens > max(processor.train_sequence_lengths)
    in_domain = ~(domain_ood | len_ood)

    # Partition invariant: in_domain is exactly the complement of the union.
    assert torch.equal(in_domain, ~(domain_ood | len_ood))
    assert (in_domain & domain_ood).sum().item() == 0
    assert (in_domain & len_ood).sum().item() == 0
    assert in_domain.sum().item() + (domain_ood | len_ood).sum().item() == n

    # Per-axis in-distribution views are the single-axis complements, and
    # in_domain is exactly their intersection.
    domain_id = ~domain_ood
    len_id = ~len_ood
    assert torch.equal(in_domain, domain_id & len_id)

    result = stats.compute_subset_test_losses(processor)
    assert set(result) == {'all', 'in_domain', 'domain_id', 'len_id',
                           'domain_ood', 'len_ood'}
    assert result['all'] == pytest.approx(float(processor.logs['test_loss'][-1]))
    for key, mask in [('in_domain', in_domain), ('domain_id', domain_id),
                      ('len_id', len_id), ('domain_ood', domain_ood),
                      ('len_ood', len_ood)]:
        if mask.sum().item() == 0:
            assert math.isnan(result[key])
        else:
            assert math.isfinite(result[key])


@pytest.mark.slow
def test_compute_subset_test_losses_len_views_nan_when_not_padded():
    """The length axis has no meaning for fixed-length (non-padded) data --
    there is no length variation -- so both length views (len_ood, len_id)
    must always be NaN there, while the domain views stay defined."""
    config = RunConfig(
        data=DatasetConfig(ground_truth=cumsum(), data_dim=(6, 1), N=32, use_padding=False),
        model=ModelConfig(arch=LSTM, hidden_dim=8, num_layers=1, pool=True, seq2seq=False),
        train=TrainConfig(epochs=2, criterion=nn.MSELoss(), seed=42,
                          optimizer_cls=optim.AdamW, lr=1e-2),
    )
    from brainspace.train import Processor
    processor = Processor.from_run_config(config)
    processor.run()

    result = stats.compute_subset_test_losses(processor)
    assert math.isnan(result['len_ood'])
    assert math.isnan(result['len_id'])
