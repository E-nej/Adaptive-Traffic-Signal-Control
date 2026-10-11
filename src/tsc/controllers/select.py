"""Controller selection (step 2G): each junction picks which controller runs it.

Experts (all Python-controlled, all on the same detectors):

    0 greenwave : follow the coordinated plan (PlanController)
    1 actuated  : coordinated-actuated control on the plan (PlanActuated)
    2 maxpressure : max-pressure on the E2 detectors (acyclic, no coordination)

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

from tsc.controllers.maxpressure import DetectorMaxPressure
from tsc.controllers.plan import PlanActuated, PlanController

EXPERTS = ('greenwave', 'actuated', 'maxpressure')


class SelectorActions:
    """Action = index of the expert (EXPERTS) that runs the junction until the next choice."""

    n_actions = len(EXPERTS)
    extra_dim = 2 * len(EXPERTS)
    needs_plan = True

    def __init__(self, network, select_interval=90, interval=5, max_extension=20,
                 noise=0.0, dropout=0.0, e2_length=50.0, comm=True, seed=0):
        self.network = network
        self.select_interval = select_interval
        self.interval = interval
        self.comm = comm
        self.experts = [
            PlanController(network, interval, max_extension),
            PlanActuated(network, interval, max_extension),
            DetectorMaxPressure(noise=noise, dropout=dropout, e2_length=e2_length, seed=seed),
        ]

    def set_plans(self, plans):
        for e in self.experts[:2]:
            e.set_plans(plans)

    def reseed(self, seed):
        self.experts[2].reseed(seed)

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
            if action < 2:
                self.experts[action].actions.rejoin(sim, tl_id)
            self.active[tl_id] = action
            self.mp_stage.pop(tl_id, None)
        if action < 2:
            return self.experts[action].act(sim, tl_id)
        # max-pressure re-decides every `interval` s of green, as its baseline does
        head = sim.heads[tl_id]
        if tl_id not in self.mp_stage or (head.timer - head.min_green) % self.interval == 0:
            self.mp_stage[tl_id] = self.experts[2].act(sim, tl_id)
        return self.mp_stage[tl_id]

    def features(self, sim, tl_id):
        own = np.zeros(len(EXPERTS), dtype=np.float32)
        own[self.active[tl_id]] = 1.0
        nb = np.zeros(len(EXPERTS), dtype=np.float32)
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


__all__ = ['EXPERTS', 'FixedSelection', 'SelectorActions']
