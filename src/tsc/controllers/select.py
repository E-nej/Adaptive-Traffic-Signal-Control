"""Expert controllers and switching between them (used by the hierarchy, step 2H, and the
best fixed assignment per junction, `tsc assign`).

Experts (all Python-controlled, all on the same detectors):

    greenwave      : follow the coordinated plan (PlanController)
    coord_actuated : coordinated-actuated control on the plan (PlanActuated)
    maxpressure    : max-pressure on the E2 detectors (acyclic, no coordination)
    actuated       : isolated gap-based actuated control, as SUMO's 'actuated' (GapActuated)
    rl, rlf        : trained stage-control RL policies ([signals] rl_experts, RLExpert;
                     e.g. a per-junction DQN and a phase-wise network)

An ExpertPool holds a subset of them ([signals] experts, in this order of indices) and
which one runs each junction.  Inactive experts keep their state up to date, and a plan
expert that takes over again rejoins the plan's timing (PlanActions.rejoin), so switching
is always safe (the SignalHead enforces minimum green and intergreens).

(Step 2G learned the choice per junction with DQN; it failed on the corridor and on Maribor
and was removed. See PLAN_REALISM.md.)
"""

from tsc.controllers.base import Controller
from tsc.controllers.maxpressure import DetectorMaxPressure
from tsc.controllers.plan import PlanActuated, PlanController
from tsc.detectors import e1_id, e1u_id

RL_EXPERTS = ('rl', 'rlf')
EXPERTS = ('greenwave', 'coord_actuated', 'maxpressure', 'actuated') + RL_EXPERTS
DEFAULT_EXPERTS = ('greenwave', 'coord_actuated', 'maxpressure')


class GapActuated(Controller):
    """Isolated actuated control (the logic of SUMO's 'actuated'): keep the green while
    vehicles keep arriving on the served lanes (gap < `gap` s), up to `max_green` s, then
    switch to the next stage in order.  Needs a SignalHead that is ready every second.

    loops = 'stopline' : the stop-line loops (step 2G).  A vehicle standing on the loop
                         (e.g. a left turner waiting for a gap) holds the green.
    loops = 'upstream' : loops 2 s of travel before the stop line, where SUMO puts them.
    loops = 'sumo'     : upstream loops, and like SUMO a lane that keeps its green in the
                         next stage does not extend the current one.
    """

    LOOPS = {'stopline': e1_id, 'upstream': e1u_id, 'sumo': e1u_id}

    def __init__(self, network, gap=3, max_green=60, loops='stopline'):
        if loops not in self.LOOPS:
            raise ValueError(f'Unknown loops {loops!r}; choose from {sorted(self.LOOPS)}')
        self.network = network
        self.gap = gap
        self.max_green = max_green
        self.loop_id = self.LOOPS[loops]
        self.served = {
            tl_id: [sorted({lane for i, lane, _ in network.links[tl_id] if g[i] in 'Gg'})
                    for g in network.greens[tl_id]]
            for tl_id in network.tl_ids
        }
        if loops == 'sumo':
            self.served = {
                tl_id: [sorted(set(lanes) - set(stages[(k + 1) % len(stages)]))
                        if len(stages) > 1 else lanes
                        for k, lanes in enumerate(stages)]
                for tl_id, stages in self.served.items()
            }

    def reset(self, sim):
        self.since = {l: 99 for tl in self.network.tl_ids for l in self.network.in_lanes[tl]}

    def tick(self, sim):
        for lane in self.since:
            if sim.api.inductionloop.getLastStepVehicleNumber(self.loop_id(lane)) > 0:
                self.since[lane] = 0
            else:
                self.since[lane] += 1

    def act(self, sim, tl_id):
        head = sim.heads[tl_id]
        lanes = self.served[tl_id][head.stage]
        if head.timer < self.max_green and any(self.since[l] < self.gap for l in lanes):
            return head.stage
        return (head.stage + 1) % len(self.network.greens[tl_id])


class RLExpert(Controller):
    """A trained stage-control RL run as an expert: its own observer, detectors (the run's
    approach detector must be in the simulation) and greedy policy.  Like max-pressure it is
    asked every `interval` s of green (the pool's periodic experts)."""

    def __init__(self, network, run_dir):
        from tsc.experiments.runs import load_policy
        if not run_dir:
            raise ValueError("expert 'rl' needs [signals] rl_expert (a trained run directory)")
        self.policy, run_cfg = load_policy(run_dir, network, 'best')
        if run_cfg.signals.control != 'stages':
            raise ValueError(f"{run_dir}: expert 'rl' needs a stage-control run, "
                             f"not {run_cfg.signals.control!r}")
        self.approach_length = run_cfg.observation.approach_length

    def reset(self, sim):
        self.policy.reset(sim)

    def tick(self, sim):
        self.policy.tick(sim)

    def act(self, sim, tl_id):
        return self.policy.act(sim, tl_id)


class Unavailable(Controller):
    """Placeholder for an expert listed without its trained run (e.g. 'rl' in a worker run
    that must have the same expert list, hence input size, as the table it will serve).
    It must never be activated."""

    def __init__(self, name):
        self.name = name

    def act(self, sim, tl_id):
        raise RuntimeError(f'expert {self.name!r} is not available in this run')


class ExpertPool:
    """The experts and which one runs each junction (index into `experts`)."""

    needs_plan = True

    def __init__(self, network, interval=5, max_extension=20, noise=0.0, dropout=0.0,
                 e2_length=50.0, seed=0, experts=DEFAULT_EXPERTS, actuated_loops='stopline',
                 hold_guard=False, rl_runs=None):
        unknown = set(experts) - set(EXPERTS)
        if unknown:
            raise ValueError(f'Unknown experts {sorted(unknown)}; choose from {EXPERTS}')
        self.network = network
        self.interval = interval
        self.names = list(experts)
        make = {
            'greenwave': lambda: PlanController(network, interval, max_extension,
                                                guard=hold_guard),
            'coord_actuated': lambda: PlanActuated(network, interval, max_extension,
                                                   guard=hold_guard),
            'maxpressure': lambda: DetectorMaxPressure(noise=noise, dropout=dropout,
                                                       e2_length=e2_length, seed=seed),
            'actuated': lambda: GapActuated(network, loops=actuated_loops),
        }
        rl_runs = rl_runs or {}
        for name in RL_EXPERTS:
            make[name] = (lambda run=rl_runs.get(name), name=name:
                          RLExpert(network, run.format(seed=seed)) if run else Unavailable(name))
        self.experts = [make[name]() for name in self.names]
        self.available = [i for i, e in enumerate(self.experts) if not isinstance(e, Unavailable)]
        self.plan_experts = [i for i, e in enumerate(self.experts) if isinstance(e, PlanController)]
        self.periodic = [i for i, n in enumerate(self.names) if n in ('maxpressure', *RL_EXPERTS)]

    def set_plans(self, plans):
        for i in self.plan_experts:
            self.experts[i].set_plans(plans)

    def reseed(self, seed):
        for e in self.experts:
            if hasattr(e, 'reseed'):
                e.reseed(seed)

    def reset(self, sim):
        for e in self.experts:
            e.reset(sim)
        self.active = {tl_id: 0 for tl_id in self.network.tl_ids}
        self.mp_stage = {}

    def tick(self, sim):
        for e in self.experts:
            e.tick(sim)

    def stage(self, sim, tl_id, action):
        self.activate(sim, tl_id, action)
        return self.propose(sim, tl_id)

    def activate(self, sim, tl_id, action):
        """Hand the junction to expert `action` (a plan expert rejoins its plan)."""
        if action != self.active[tl_id]:
            if action in self.plan_experts:
                self.experts[action].actions.rejoin(sim, tl_id)
            self.active[tl_id] = action
            self.mp_stage.pop(tl_id, None)

    def propose(self, sim, tl_id):
        """Stage the active expert asks for now (call once per second and junction)."""
        action = self.active[tl_id]
        if action not in self.periodic:
            return self.experts[action].act(sim, tl_id)
        # max-pressure (and RL) re-decide every `interval` s of green, as their baselines do
        head = sim.heads[tl_id]
        if tl_id not in self.mp_stage or (head.timer - head.min_green) % self.interval == 0:
            self.mp_stage[tl_id] = self.experts[action].act(sim, tl_id)
        return self.mp_stage[tl_id]


class FixedSelection(PlanController):
    """Non-learning selector: a fixed expert per junction ({tl_id: expert index}, default
    `default`).  With one expert everywhere it equals that expert; with a per-junction
    assignment it is the 'best fixed choice per junction' baseline."""

    needs_plan = True

    def __init__(self, network, assignment=None, default=0, **kwargs):
        self.actions = ExpertPool(network, **kwargs)
        self.assignment = dict(assignment or {})
        self.default = default

    def set_plans(self, plans):
        self.actions.set_plans(plans)

    def reseed(self, seed):
        self.actions.reseed(seed)

    def reset(self, sim):
        self.actions.reset(sim)

    def tick(self, sim):
        self.actions.tick(sim)

    def act(self, sim, tl_id):
        return self.actions.stage(sim, tl_id, self.assignment.get(tl_id, self.default))


__all__ = ['DEFAULT_EXPERTS', 'EXPERTS', 'ExpertPool', 'FixedSelection', 'GapActuated',
           'RLExpert']
