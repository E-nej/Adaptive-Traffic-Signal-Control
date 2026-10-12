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
    generator: str = 'random_trips'   # 'random_trips' (closed test network) | 'od'
                                      # | 'file' (a published scenario's route files)
    profile: str = 'peak'             # see demand.PROFILES
    duration: int = 3600              # s, episode length
    begin: int = 0                    # s, simulation start (time of day of 'file' demand)
    min_distance: float = 150.0       # m, shortest trip
    # generator 'od' (networks with open boundaries, see demand.py)
    major_share: float = 0.6          # share of trips between ends of the main road
    tidal_share: float = 0.5          # share of directional trips going along tidal_axis
    tidal_axis: list[float] = field(default_factory=lambda: [1.0, 0.0])
    tidal_random: bool = True         # draw the tidal direction per scenario
    # generator 'file'
    route_files: list[str] = field(default_factory=list)   # globs, relative to the repo
    eval_files: int = 0               # with several files: the last ones, for evaluation only
    count_undeparted: bool = False    # vehicles still waiting to enter at the end count in
                                      # the delay (their departDelay so far), as in RESCO


@dataclass
class SignalConfig:
    """Python-controlled junctions (max-pressure, RL)."""
    decision_interval: int = 5        # s between decisions after min green
    control: str = 'stages'           # 'stages' (pick the next stage) | 'plan' (adjust a
                                      # coordinated plan: FOLLOW / HOLD / EARLY, step 2F)
                                      # | 'hierarchy' (a manager picks the experts, 2H)
    max_extension: int = 20           # s, plan control: longest green extension past the plan
    hold_guard: bool = False          # plan control (and plan experts under 'hierarchy'):
                                      # extend only if the next stage keeps its
                                      # minimum green; catch up without skipping stages
    experts: list[str] = field(       # 'hierarchy': controllers to choose from
        default_factory=lambda: ['greenwave', 'coord_actuated', 'maxpressure'])
    actuated_loops: str = 'stopline'  # 'actuated' expert / actuated_py: 'stopline' (2G) or
                                      # 'upstream' (where SUMO's actuated control has them)
    max_green: int | None = None      # s, safety shield (None = off)
    max_red: int | None = None        # s, safety shield (None = off)
    red_call: bool = False            # max_red only for stages a detector reports a vehicle
                                      # waiting for (else every stage is served in time)
    amp_weight: float = 1.0           # advanced_mp: keep the green while weight x moving
    amp_range: float = 100.0          # vehicles within amp_range m >= the highest pressure
    rl_experts: dict[str, str] = field(default_factory=dict)   # experts 'rl' / 'rlf':
                                      # trained stage-control runs ('{seed}' = the hierarchy
                                      # run's training seed); an RL expert without one is
                                      # listed (same input size) but unavailable


@dataclass
class HierarchyConfig:
    """[signals] control = 'hierarchy' (controllers.hierarchy)."""
    scope: str = 'network'            # one expert per 'network' | 'group' | 'junction'
    group_range: float = 400.0        # m, scope 'group': signals this close form a group
    interval: int = 300               # s between manager decisions
    manager: str = 'learned'          # 'learned' (DQN) | 'fixed' (always fixed_expert)
                                      # | 'table' (best expert for the measured demand)
    table_levels: list[float] = field(default_factory=list)   # 'table': demand grid (empty:
                                      # 9 levels from train.level_min to level_max)
    table_scenarios: int = 3          # 'table': scenarios per level and expert
    hysteresis: int = 2               # 'table': decisions in a row before a switch
    table_alone: bool = False         # 'table' + worker 'expert': also measure every expert
                                      # without its worker; the table picks per level
    fixed_expert: int = 0             # index into [signals] experts
    worker: str = 'off'               # 'off': junctions run the chosen expert unmodified;
                                      # 'residual': per-junction RL fine-tunes it
    reward: str = 'queue'             # manager reward (reward.KINDS), mean over the group
    reward_scale: float = 1.0
    default_expert: str = ''          # learned manager: '' (none) | an expert's name | 'auto'
                                      # (the best single expert on the validation scenarios,
                                      # measured before training).  The manager runs it
                                      # unless another expert's Q-value is higher by
                                      # `margin`·|Q(default)|, and a run whose manager never
                                      # beats it on validation deploys it (best checkpoint)
    margin: float = 0.05
    accept_scenarios: int = 0         # with a default expert (or workers under a fixed
                                      # expert): after training, the best checkpoint must
                                      # beat that expert alone on this many fresh scenarios
                                      # per validation level (paired, one-sided Wilcoxon
                                      # p < 0.05), else the run deploys the expert alone
    worker_runs: dict[str, str] = field(default_factory=dict)  # worker = 'expert': trained
                                      # worker run per expert name ('{seed}' = this run's
                                      # training seed); experts without one run alone
    gamma: float = 0.8
    lr: float = 1e-3
    tau: float = 0.01
    batch_size: int = 32
    buffer_size: int = 20000
    updates: int = 20                 # gradient steps per manager decision
    hidden: list[int] = field(default_factory=lambda: [64, 64])


@dataclass
class ObservationConfig:
    mode: str = 'grid'                # 'grid' (perfect information) or 'detector'
    comm: bool = True                 # append neighbours' messages
    message: str = 'aggregate'        # 'aggregate' | 'directional' (see observation.py)
    noise: float = 0.0                # detector noise (std, normalised units)
    dropout: float = 0.0              # probability a detector reading fails
    e2_length: float = 50.0           # m, area detector length before the stop line
    approach_length: float = 0.0      # m, detector mode: a second area detector this long;
                                      # adds the vehicles moving toward the stop line per
                                      # lane (0 = off; ~ speed limit x decision interval)
    layout: str = 'flat'              # 'flat' (lanes in order) | 'phase' (one feature row per
                                      # green stage, for agent.phase_net; see observation.py)


@dataclass
class RewardConfig:
    kind: str = 'multi'               # see reward.KINDS
    neighbour_weight: float = 0.0     # α: r_i + α·Σ r_neighbours (cooperative reward)
    scale: float = 1.0                # rewards are multiplied by this before learning
    pressure_weight: float = 0.25     # kind 'queue_pressure': weight of |pressure|


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
    shared: bool = False              # one network for all junctions (+ junction one-hot)
    smdp: bool = False                # time-based discounting: the reward is the discounted
                                      # sum over the decision's seconds, the next value is
                                      # discounted by gamma ** seconds (gamma per second)
    dueling: bool = False             # dueling head: V(s) + A(s, a) - mean A
    n_step: int = 1                   # n-step returns (decisions per stored transition)
    ref_action: int = -1              # >= 0: Q(s, a) = V(s) + A(s, a) - A(s, ref_action), A
                                      # starting at zero: the untrained greedy policy takes
                                      # ref_action (workers: AGREE, i.e. start as the expert)
                                      # and only learned advantages move it away (residual RL)
    phase_net: bool = False           # FRAP / MPLight-style phase-wise Q-network, shared by
                                      # all junctions (needs observation.layout = 'phase')


@dataclass
class TrainConfig:
    seeds: list[int] = field(default_factory=lambda: [0])
    episodes: int = 150
    level_min: float = 600.0          # veh/h, demand level sampled uniformly per episode
    level_max: float = 3000.0
    eps_start: float = 1.0            # epsilon of the first episode (a worker that starts as
                                      # its expert needs little exploration)
    eps_min: float = 0.05
    eps_fraction: float = 0.6         # fraction of episodes over which epsilon decays
    val_every: int = 10               # episodes between validations (model selection)
    val_levels: list[float] = field(default_factory=lambda: [1200.0, 2400.0])
    val_per_level: int = 1            # validation scenarios per level (more: less noisy
                                      # model selection)
    # plan control: green-wave plans calibrated at these levels (per direction); an episode
    # runs the plan matching its demand with probability plan_match, else a random one
    plan_levels: list[float] = field(
        default_factory=lambda: [1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
    plan_match: float = 0.5
    # 'cpu' (the bit-exact reference) or 'gpu' (CUDA / ROCm; reproducible only on the same
    # GPU and software stack, see docs/GPU.md); gpu_index picks one of several GPUs.
    device: str = 'cpu'
    gpu_index: int | None = None


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
    # fixed-time plans (and plan-controlled policies) use the plan of this demand level
    # instead of each level's own (None): a stale plan
    plan_level: float | None = None
    plan_direction: str = 'match'     # 'match' | 'opposite' (e.g. the AM plan in the PM peak)
    assignment: str = ''              # assignment.json of `tsc assign` (baseline select_assigned)
    # Trained policies are evaluated on the CPU, whatever they were trained on, so
    # evaluation results do not depend on the GPU model (reproducibility policy, docs/GPU.md).
    device: str = 'cpu'


@dataclass
class Config:
    name: str = 'experiment'
    output_dir: str = 'outputs'
    workers: int = 4                  # parallel processes (seeds or evaluation tasks)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)


# ----------------------------------------------------------------------
# Loading / saving
# ----------------------------------------------------------------------

# Keys of removed features: still in the config.toml of earlier runs, dropped when loading.
RETIRED = {
    'signals': {'select_interval'},     # RL controller selection (2G), removed
}


def _build(cls, data, where):
    if not isinstance(data, dict):
        raise ValueError(f'[{where}] must be a table, got {data!r}')
    fields = {f.name: f for f in dataclasses.fields(cls)}
    data = {k: v for k, v in data.items() if k not in RETIRED.get(where, ())}
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
