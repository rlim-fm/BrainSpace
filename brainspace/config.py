import dataclasses
from dataclasses import dataclass, field, replace
from typing import Optional, Callable, Any, Union
import inspect
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from .models import DynamicNN, StaticNN, LSTM, GRU, MHA
from .datasets import cumsum, sin, x2, Dataset


@dataclass
class DatasetConfig:
    x_range: tuple = (-8, 8)
    data_dim: tuple = (10, 1)
    N: int = 2048
    ground_truth: Callable = field(default_factory=lambda: cumsum())
    use_padding: bool = False
    min_seq_len: int = 5

    def build(self, rng=None, device=None, seed=None, cot=False) -> Dataset:
        """Create a Dataset from this config.

        Args:
            rng: accepted for backwards compatibility (unused).
            device: torch device for the generated tensors.
            seed: data-generation seed. Pass the run seed here so that
                ``from_run_config`` produces identical datasets across calls.
            cot: Chain-of-Thought supervision density; see
                ``Dataset.__init__`` for semantics. Pass ``TrainConfig.cot``
                here so the supervision mask is fixed at dataset-construction
                time.
        """
        return Dataset(
            ground_truth=self.ground_truth,
            x_range=self.x_range,
            data_dim=self.data_dim,
            N=self.N,
            use_padding=self.use_padding,
            min_seq_len=self.min_seq_len,
            cot=cot,
            seed=seed,
            device=device,
        )


@dataclass
class ModelConfig:
    arch: type = LSTM
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    pool: bool = False
    seq2seq: bool = True
    # Causal (autoregressive) attention masking for transformer archs; ignored
    # by archs whose __init__ has no 'causal' parameter.
    causal: bool = True
    output_dim: int = 1
    arch_kwargs: dict = field(default_factory=dict)

    def build(self, input_dim: int):
        """
        Instantiate model.

        Every dataclass field whose name appears in the arch ``__init__``
        signature is passed through (so subclass fields added by domain
        packages plug in with no changes here). ``arch``/``arch_kwargs`` are
        structural, and ``output_dim`` maps to ``num_classes`` for archs that
        use that name instead.

        Args:
            input_dim: feature dimension (derived from Dataset.feature_dim)

        Returns:
            model instance
        """
        sig = inspect.signature(self.arch.__init__).parameters

        base_kwargs = {
            'input_dim': input_dim,
            'output_dim': self.output_dim,
        }

        for f in dataclasses.fields(self):
            if f.name in ('arch', 'arch_kwargs', 'output_dim'):
                continue
            if f.name in sig:
                base_kwargs[f.name] = getattr(self, f.name)

        if 'num_classes' in sig:
            base_kwargs.pop('output_dim', None)
            base_kwargs['num_classes'] = self.output_dim

        base_kwargs.update(self.arch_kwargs)
        model = self.arch(**base_kwargs)
        model.metadata = self.metadata()
        return model

    def metadata(self) -> dict:
        """Descriptive config metadata (architecture + hyperparameters), independent of any built model instance."""
        meta = {"arch": self.arch.__name__}
        meta.update({
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ('arch', 'arch_kwargs')
        })
        return meta


@dataclass
class TrainConfig:
    epochs: int = 1000
    criterion: nn.Module = field(default_factory=nn.MSELoss)
    batch_size: Optional[int] = None
    seed: Optional[int] = None
    dtype: torch.dtype = torch.float32
    device: Optional[str] = None
    # bool: enable/disable dense per-timestep (Chain-of-Thought) supervision. float in (0, 1]:
    # enable with that fraction of otherwise-valid intermediate steps randomly
    # (Bernoulli) excluded from supervision, except each sample's own final
    # valid timestep, which is always kept. The dropped set is fixed once for
    # the whole run (not resampled per epoch) — see Processor.train_epoch.
    cot: Union[bool, float] = False

    optimizer_cls: type = field(default_factory=lambda: optim.AdamW)
    lr: float = 0.001
    weight_decay: float = 0.0
    scheduler_cls: Optional[type] = None
    scheduler_kwargs: dict = field(default_factory=dict)

    def build_optimizer_and_scheduler(self, model):
        """
        Instantiate optimizer and (optionally) scheduler for the given model.

        Args:
            model: the model to optimize

        Returns:
            (optimizer, scheduler) tuple
        """
        optimizer = self.optimizer_cls(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = (
            self.scheduler_cls(optimizer, **self.scheduler_kwargs)
            if self.scheduler_cls else None
        )
        return optimizer, scheduler

    def build_epoch_hooks(self) -> list:
        """Per-epoch callbacks ``hook(processor, epoch)`` run at the start of
        every training epoch. The base config has none; domain subclasses
        override this to wire schedules (e.g. a temperature/tau annealing
        schedule) into the training loop."""
        return []

    def build_batch_sampler(self, dataset):
        """Optional training batch sampler for the Processor.

        Return an iterable yielding index batches (LongTensor / sequence of
        ints) or ``(indices, weights)`` pairs; return ``None`` (the default)
        for the standard contiguous-slice gradient-accumulation path. Domain
        subclasses override this to wire k-NN / cluster samplers built from
        their own config fields."""
        return None


@dataclass
class RunConfig:
    data: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ---------------------------------------------------------------------------
# Domain configuration classes
#
# Domain packages that subclass the config dataclasses register them here so
# that YAML loading, IV resolution, and the results-store identity all build
# the domain's classes. Field names shared with the base classes keep the
# same identity hashes (see internal/registry.py).
# ---------------------------------------------------------------------------

_CONFIG_CLASSES = {
    'run': RunConfig,
    'data': DatasetConfig,
    'model': ModelConfig,
    'train': TrainConfig,
}


def use_config_classes(run=None, data=None, model=None, train=None):
    """Install domain subclasses of the config dataclasses (any subset)."""
    for kind, cls in (('run', run), ('data', data), ('model', model), ('train', train)):
        if cls is not None:
            _CONFIG_CLASSES[kind] = cls


def get_config_class(kind: str) -> type:
    return _CONFIG_CLASSES[kind]


# ---------------------------------------------------------------------------
# Declarative YAML loading
#
# Config fields that hold Python objects (architectures, ground truths,
# optimizers, criteria, schedulers) are written in YAML as either a bare
# registry name (``arch: GRU``) or a dict with ``type`` plus ``args``/
# ``kwargs`` (``ground_truth: {type: cumsum}``). Domain packages extend the
# registries via the ``register_*`` helpers below. See ``configs/`` for
# worked examples.
# ---------------------------------------------------------------------------

ARCH_REGISTRY = {
    'LSTM': LSTM, 'GRU': GRU, 'MHA': MHA,
}
GROUND_TRUTH_REGISTRY = {
    'cumsum': cumsum, 'x2': x2, 'sin': sin,
}
OPTIMIZER_REGISTRY = {'AdamW': optim.AdamW, 'Adam': optim.Adam, 'SGD': optim.SGD}
CRITERION_REGISTRY = {'MSELoss': nn.MSELoss, 'L1Loss': nn.L1Loss}
SCHEDULER_REGISTRY = {
    'StepLR': optim.lr_scheduler.StepLR,
    'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR,
}

# Which registry each config field name resolves through (used for both
# sub-config building and IV value resolution). Extensible via
# register_field_registry for fields added by domain config subclasses.
_FIELD_REGISTRIES = {
    'arch': ARCH_REGISTRY,
    'ground_truth': GROUND_TRUTH_REGISTRY,
    'optimizer_cls': OPTIMIZER_REGISTRY,
    'criterion': CRITERION_REGISTRY,
    'scheduler_cls': SCHEDULER_REGISTRY,
}

# Field names whose registry entries are classes to pass through as-is (not
# instantiated) when written as a bare YAML name.
_CLASS_VALUED_FIELDS = {'arch', 'optimizer_cls', 'scheduler_cls'}


def register_arch(name: str, cls: type):
    ARCH_REGISTRY[name] = cls


def register_ground_truth(name: str, factory: Callable):
    GROUND_TRUTH_REGISTRY[name] = factory


def register_criterion(name: str, cls: type):
    CRITERION_REGISTRY[name] = cls


def register_optimizer(name: str, cls: type):
    OPTIMIZER_REGISTRY[name] = cls


def register_scheduler(name: str, cls: type):
    SCHEDULER_REGISTRY[name] = cls


def register_field_registry(field_name: str, registry: dict, class_valued: bool = False):
    """Route a (domain-added) config field through its own name registry when
    resolving YAML/IV values (e.g. ``tau_schedule`` in a tropical package)."""
    _FIELD_REGISTRIES[field_name] = registry
    if class_valued:
        _CLASS_VALUED_FIELDS.add(field_name)


def _resolve(spec, registry, field_name):
    """Resolve one YAML value through `registry`.

    A bare string is looked up directly (and called with no args, for
    fields expecting an instance like `ground_truth`/`criterion`; class-valued
    fields like `arch`/`optimizer_cls`/`scheduler_cls` are left as the class
    itself). A dict `{type: Name, args: [...], kwargs: {...}}` looks up `type`
    and calls it with args/kwargs. Anything else (already-scalar values)
    passes through unchanged.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec not in registry:
            raise ValueError(f"Unknown {field_name} '{spec}'; "
                             f"available: {sorted(registry)}")
        obj = registry[spec]
        if field_name in _CLASS_VALUED_FIELDS:
            return obj
        return obj()
    if isinstance(spec, dict):
        type_name = spec.get('type')
        if type_name not in registry:
            raise ValueError(f"Unknown {field_name} type '{type_name}'; "
                             f"available: {sorted(registry)}")
        obj = registry[type_name]
        args = spec.get('args', [])
        kwargs = spec.get('kwargs', {})
        if field_name in _CLASS_VALUED_FIELDS and not args and not kwargs:
            return obj
        return obj(*args, **kwargs)
    return spec


def _build_subconfig(cls, d: dict):
    """Build a dataclass config from a YAML dict, resolving any field named
    in `_FIELD_REGISTRIES` through its registry and passing the rest
    straight through to the dataclass constructor."""
    d = dict(d or {})
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - valid_fields
    if unknown:
        raise ValueError(f"Unknown field(s) {sorted(unknown)} for {cls.__name__}; "
                         f"valid fields: {sorted(valid_fields)}")
    kwargs = {}
    for key, value in d.items():
        if key in _FIELD_REGISTRIES:
            kwargs[key] = _resolve(value, _FIELD_REGISTRIES[key], key)
        elif key in ('x_range', 'data_dim') and isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _build_dataset_config(d: dict) -> DatasetConfig:
    return _build_subconfig(get_config_class('data'), d)


def _build_model_config(d: dict) -> ModelConfig:
    return _build_subconfig(get_config_class('model'), d)


def _build_train_config(d: dict) -> TrainConfig:
    return _build_subconfig(get_config_class('train'), d)


def build_run_config(d: dict) -> RunConfig:
    """Build a RunConfig from a YAML-parsed dict with 'data'/'model'/'train' keys."""
    d = d or {}
    return get_config_class('run')(
        data=_build_dataset_config(d.get('data', {})),
        model=_build_model_config(d.get('model', {})),
        train=_build_train_config(d.get('train', {})),
    )


def load_run_config(path: str) -> RunConfig:
    """Load a single (non-grid) RunConfig from a YAML file with 'data'/'model'/'train' keys."""
    with open(path) as f:
        d = yaml.safe_load(f)
    return build_run_config(d)


def _sub_config_search_order():
    return ('data', 'model', 'train')


def _resolve_iv_key_field(iv_key: str):
    """Map an IV key (bare or dotted, e.g. 'arch' or 'model.arch') to the
    bare field name that decides which registry (if any) applies, mirroring
    Experiment._apply_iv's sub-config search order."""
    if '.' in iv_key:
        _sub, field_name = iv_key.split('.', 1)
        return field_name
    return iv_key


def _resolve_iv_values(iv_key: str, values: list) -> list:
    """Resolve each value in an IV's value list through the registry that
    applies to its field name (if any); otherwise return values unchanged."""
    field_name = _resolve_iv_key_field(iv_key)
    registry = _FIELD_REGISTRIES.get(field_name)
    if registry is None:
        return list(values)
    return [_resolve(v, registry, field_name) for v in values]


def load_experiment_spec(path: str) -> dict:
    """Load a full experiment spec (base RunConfig + IVs + metadata) from a
    YAML file.

    Expected top-level keys: `name`, `base` (RunConfig fields, as in
    `load_run_config`), `ivs` (dict of key -> list of values), `ordinal_ivs`
    (list, optional), `trials` (default 5), `global_seed` (optional),
    `results_root` (optional, default 'results'), `save_models` (default False).
    """
    with open(path) as f:
        spec = yaml.safe_load(f) or {}

    ivs_raw = spec.get('ivs', {}) or {}
    ivs = {key: _resolve_iv_values(key, values) for key, values in ivs_raw.items()}

    return {
        'name': spec.get('name', 'experiment'),
        'base_config': build_run_config(spec.get('base', {})),
        'ivs': ivs,
        'ordinal_ivs': list(spec.get('ordinal_ivs', []) or []),
        'trials': spec.get('trials', 5),
        'global_seed': spec.get('global_seed'),
        'results_root': spec.get('results_root', 'results'),
        'save_models': spec.get('save_models', False),
    }
