"""Invariants for the declarative configuration dataclasses."""
import pytest
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from brainspace.config import RunConfig, DatasetConfig, ModelConfig, TrainConfig
from brainspace.datasets import Dataset
from brainspace.models import LSTM, GRU, SimpleTransformerModel


def test_dataclass_defaults_and_composition():
    cfg = RunConfig()
    assert isinstance(cfg.data, DatasetConfig)
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.train, TrainConfig)
    assert cfg.model.arch is LSTM


def test_dataset_config_build():
    ds = DatasetConfig(data_dim=(5, 3), N=20).build()
    assert isinstance(ds, Dataset)
    assert ds.feature_dim == 3
    assert ds.x_train.shape == (20, 5, 3)


def test_model_config_build_returns_model_defaults():
    model = ModelConfig(arch=LSTM, hidden_dim=16).build(input_dim=1)
    assert isinstance(model, LSTM)
    assert model.hidden_dim == 16


def test_model_config_build_gru():
    model = ModelConfig(arch=GRU).build(input_dim=2)
    assert isinstance(model, GRU)


def test_train_config_build_optimizer_and_scheduler():
    model = LSTM(input_dim=1, output_dim=1, hidden_dim=16)
    optimizer, scheduler = TrainConfig(
        optimizer_cls=optim.Adam, lr=5e-3, weight_decay=1e-4
    ).build_optimizer_and_scheduler(model)
    assert isinstance(optimizer, optim.Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-4)
    assert scheduler is None


def test_train_config_build_optimizer_with_scheduler():
    model = LSTM(input_dim=1, output_dim=1, hidden_dim=16)
    optimizer, scheduler = TrainConfig(
        scheduler_cls=StepLR, scheduler_kwargs={"step_size": 10}
    ).build_optimizer_and_scheduler(model)
    assert isinstance(optimizer, optim.AdamW)
    assert isinstance(scheduler, StepLR)


def test_model_config_build_passes_subclass_fields_by_signature():
    """Fields added by domain config subclasses reach the arch constructor
    when (and only when) its __init__ signature names them."""
    import dataclasses

    @dataclasses.dataclass
    class ExtendedModelConfig(ModelConfig):
        extra_flag: bool = True

    model = ExtendedModelConfig(arch=LSTM, hidden_dim=8).build(input_dim=1)
    assert isinstance(model, LSTM)  # LSTM has no extra_flag param: ignored

    class FlaggedLSTM(LSTM):
        def __init__(self, *args, extra_flag=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.extra_flag = extra_flag

    model = ExtendedModelConfig(arch=FlaggedLSTM, hidden_dim=8).build(input_dim=1)
    assert model.extra_flag is True


def test_model_config_arch_kwargs_merged():
    # SimpleTransformerModel accepts n_heads / d_model via arch_kwargs.
    model = ModelConfig(
        arch=SimpleTransformerModel,
        arch_kwargs={"d_model": 16, "n_heads": 4},
    ).build(input_dim=1)
    assert isinstance(model, SimpleTransformerModel)


def test_model_config_num_classes_remap_for_transformer():
    # SimpleTransformerModel takes `num_classes`, not `output_dim`; build() remaps.
    model = ModelConfig(
        arch=SimpleTransformerModel, output_dim=3,
        arch_kwargs={"d_model": 8, "n_heads": 2},
    ).build(input_dim=1)
    assert model.output_layer.out_features == 3
