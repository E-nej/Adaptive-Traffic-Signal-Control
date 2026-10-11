"""Experiment configuration: one TOML file per experiment.

A config file only needs the values that differ from the defaults below; the
fully resolved config is saved next to every run's results.  Relative paths
are relative to the directory the `tsc` command is started from (the
repository root in the docs).

    name = "full"

    [observation]
    mode = "detector"
    noise = 0.05

Unknown keys are errors, so a typo cannot silently fall back to a default.
"""

import dataclasses
import tomllib
from dataclasses import dataclass, field

import tomli_w


@dataclass
class NetworkConfig:
    path: str = 'networks/test5/test.net.xml'
    # Rebuild every signal plan with netconvert (conflict-free stages + intergreens).
    # Steps 3+ (real signal plans) will set this to false.
    rebuild_tls: bool = True
    tls_layout: str = 'opposites'
    yellow: int = 3          # s
    all_red: int = 2         # s
    min_green: int = 10      # s
    max_green: int = 60      # s, SUMO's actuated baseline only (the RL shield is [signals])


@dataclass
class DemandConfig:
    generator: str = 'random_trips'   # Poisson trips between random O/D edges
    profile: str = 'peak'             # see demand.PROFILES
    duration: int = 3600              # s, episode length
    min_distance: float = 150.0       # m, shortest trip


@dataclass
class SignalConfig:
    """Python-controlled junctions (max-pressure, RL)."""
    decision_interval: int = 5        # s between decisions after min green
    max_green: int | None = None      # s, safety shield (None = off)
    max_red: int | None = None        # s, safety shield (None = off)


@dataclass
class ObservationConfig:
    mode: str = 'grid'                # 'grid' (perfect information) or 'detector'
    comm: bool = True                 # append neighbours' messages
    noise: float = 0.0                # detector noise (std, normalised units)
    dropout: float = 0.0              # probability a detector reading fails
    e2_length: float = 50.0           # m, area detector length before the stop line


@dataclass
class RewardConfig:
    kind: str = 'multi'               # 'wait' | 'multi' | 'queue' | 'det_queue'


@dataclass
class AgentConfig:
    double: bool = True
    target_update: str = 'soft'       # 'soft' (Polyak, every gradient step) | 'hard'
    tau: float = 0.005
    hard_every: int = 1000            # gradient steps between hard updates
    gamma: float = 0.95
    lr: float = 2.5e-4
    batch_size: int = 100
    buffer_size: int = 50000
    learn_start: int = 500
    hidden: list[int] = field(default_factory=lambda: [400, 400])


@dataclass
class TrainConfig:
    seeds: list[int] = field(default_factory=lambda: [0])
    episodes: int = 150
    level_min: float = 600.0          # veh/h, demand level sampled uniformly per episode
    level_max: float = 3000.0
    eps_min: float = 0.05
    eps_fraction: float = 0.6         # fraction of episodes over which epsilon decays
    val_every: int = 10               # episodes between validations (model selection)
    val_levels: list[float] = field(default_factory=lambda: [1200.0, 2400.0])


@dataclass
class EvaluateConfig:
    baselines: list[str] = field(
        default_factory=lambda: ['fixed', 'webster', 'actuated', 'maxpressure'])
    runs: list[str] = field(default_factory=list)   # trained run dirs, globs allowed
    which: str = 'best'               # 'best' (validation) or 'final' weights
    levels: list[float] = field(
        default_factory=lambda: [600.0, 1200.0, 1800.0, 2400.0, 3000.0, 3600.0])
    n_seeds: int = 10                 # evaluation scenarios per level
    n_calibration: int = 3            # scenarios per level for the Webster plan
    tag: str = ''                     # suffix for controller names (e.g. '_shield')


@dataclass
class Config:
    name: str = 'experiment'
    output_dir: str = 'outputs'
    workers: int = 4                  # parallel processes (seeds or evaluation tasks)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)


# ----------------------------------------------------------------------
# Loading / saving
# ----------------------------------------------------------------------

def _build(cls, data, where):
    if not isinstance(data, dict):
        raise ValueError(f'[{where}] must be a table, got {data!r}')
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f'Unknown key(s) in [{where or "top level"}]: {sorted(unknown)}')
    kwargs = {}
    for name, value in data.items():
        sub = fields[name].default_factory if fields[name].default_factory is not dataclasses.MISSING else None
        if sub is not None and dataclasses.is_dataclass(sub):
            value = _build(sub, value, name)
        kwargs[name] = value
    return cls(**kwargs)


def from_dict(data):
    return _build(Config, data, '')


def load(path, overrides=()):
    """Read a TOML config; `overrides` are 'section.key=value' strings (value in TOML syntax)."""
    with open(path, 'rb') as f:
        data = tomllib.load(f)
    for item in overrides:
        apply_override(data, item)
    return from_dict(data)


def apply_override(data, item):
    key, sep, raw = item.partition('=')
    if not sep:
        raise ValueError(f"Override must look like 'section.key=value', got {item!r}")
    try:
        value = tomllib.loads(f'v = {raw}')['v']
    except tomllib.TOMLDecodeError:
        value = raw                    # bare strings: --set observation.mode=detector
    *path, last = key.strip().split('.')
    node = data
    for part in path:
        node = node.setdefault(part, {})
    node[last] = value


def to_dict(cfg):
    """Plain dict without None values (TOML has no null; None means 'use the default')."""
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items() if v is not None}
        return x
    return clean(dataclasses.asdict(cfg))


def save(cfg, path):
    with open(path, 'wb') as f:
        tomli_w.dump(to_dict(cfg), f)


def replace(cfg, **sections):
    """Copy of cfg with some fields of some sections replaced: replace(cfg, train={'seeds': [1]})."""
    changes = {}
    for section, values in sections.items():
        current = getattr(cfg, section)
        changes[section] = dataclasses.replace(current, **values) if isinstance(values, dict) else values
    return dataclasses.replace(cfg, **changes)
