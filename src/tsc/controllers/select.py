"""Controller selection (step 2G): each junction picks which controller runs it.

Experts (all Python-controlled, all on the same detectors):

    greenwave      : follow the coordinated plan (PlanController)
    coord_actuated : coordinated-actuated control on the plan (PlanActuated)
    maxpressure    : max-pressure on the E2 detectors (acyclic, no coordination)
    actuated       : isolated gap-based actuated control, as SUMO's 'actuated' (GapActuated)

A selector uses a subset of them ([signals] experts, in this order of indices).
Every `select_interval` seconds (about one cycle) the policy picks an expert per
junction; between choices the chosen expert controls the junction.  Inactive
experts keep their state up to date, and a plan expert that takes over again
rejoins the plan's timing (PlanActions.rejoin), so switching is always safe (the
SignalHead enforces minimum green and intergreens).

The policy's extra features are the junction's own expert and, with
communication, the share of its neighbours running each expert: a green wave
only helps if the neighbours run it too.
"""

import numpy as np

from tsc.controllers.base import Controller
from tsc.controllers.maxpressure import DetectorMaxPressure
from tsc.controllers.plan import PlanActuated, PlanController
from tsc.detectors import e1_id, e1u_id

EXPERTS = ('greenwave', 'coord_actuated', 'maxpressure', 'actuated')
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


class SelectorActions:
    """Action = index (into `experts`) of the expert that runs the junction until the next
    choice."""

    needs_plan = True

    def __init__(self, network, select_interval=90, interval=5, max_extension=20,
                 noise=0.0, dropout=0.0, e2_length=50.0, comm=True, seed=0,
                 experts=DEFAULT_EXPERTS, actuated_loops='stopline'):
        unknown = set(experts) - set(EXPERTS)
        if unknown:
            raise ValueError(f'Unknown experts {sorted(unknown)}; choose from {EXPERTS}')
        self.network = network
        self.select_interval = select_interval
        self.interval = interval
        self.comm = comm
        self.names = list(experts)
        make = {
            'greenwave': lambda: PlanController(network, interval, max_extension),
            'coord_actuated': lambda: PlanActuated(network, interval, max_extension),
            'maxpressure': lambda: DetectorMaxPressure(noise=noise, dropout=dropout,
                                                       e2_length=e2_length, seed=seed),
            'actuated': lambda: GapActuated(network, loops=actuated_loops),
        }
        self.experts = [make[name]() for name in self.names]
        self.n_actions = len(self.experts)
        self.extra_dim = 2 * self.n_actions
        self.plan_experts = [i for i, e in enumerate(self.experts) if isinstance(e, PlanController)]
        self.periodic = [i for i, n in enumerate(self.names) if n == 'maxpressure']

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
        self.next_choice = {tl_id: 0 for tl_id in self.network.tl_ids}
        self.mp_stage = {}

    def tick(self, sim):
        for e in self.experts:
            e.tick(sim)

    # ------------------------------------------------------------------

    def decides(self, sim, tl_id):
        if sim.t >= self.next_choice[tl_id]:
            self.next_choice[tl_id] = sim.t + self.select_interval
            return True
        return False

    def mask(self, sim, tl_id):
        return np.ones(self.n_actions, dtype=bool)

    def stage(self, sim, tl_id, action):
        if action != self.active[tl_id]:
            if action in self.plan_experts:
                self.experts[action].actions.rejoin(sim, tl_id)
            self.active[tl_id] = action
            self.mp_stage.pop(tl_id, None)
        if action not in self.periodic:
            return self.experts[action].act(sim, tl_id)
        # max-pressure re-decides every `interval` s of green, as its baseline does
        head = sim.heads[tl_id]
        if tl_id not in self.mp_stage or (head.timer - head.min_green) % self.interval == 0:
            self.mp_stage[tl_id] = self.experts[action].act(sim, tl_id)
        return self.mp_stage[tl_id]

    def features(self, sim, tl_id):
        own = np.zeros(self.n_actions, dtype=np.float32)
        own[self.active[tl_id]] = 1.0
        nb = np.zeros(self.n_actions, dtype=np.float32)
        neighbours = self.network.neighbours[tl_id]
        if self.comm and neighbours:
            for n in neighbours:
                nb[self.active[n]] += 1.0 / len(neighbours)
        return np.concatenate([own, nb])


class FixedSelection(PlanController):
    """Non-learning selector: a fixed expert per junction ({tl_id: expert index}, default
    `default`).  With one expert everywhere it equals that expert; with a per-junction
    assignment it is the 'best fixed choice per junction' baseline."""

    needs_plan = True

    def __init__(self, network, assignment=None, default=0, **kwargs):
        self.actions = SelectorActions(network, **kwargs)
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


__all__ = ['DEFAULT_EXPERTS', 'EXPERTS', 'FixedSelection', 'GapActuated', 'SelectorActions']
