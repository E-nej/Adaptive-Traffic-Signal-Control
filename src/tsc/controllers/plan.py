"""Control on top of a coordinated fixed-time plan (green wave), step 2F.

The plan keeps the junctions coordinated; a policy only adjusts it.  Every
`interval` seconds of green (after min green) the policy picks one of

    FOLLOW : switch exactly when the plan does
    HOLD   : keep the current stage for another interval although the plan has
             moved on (green extension, at most `max_extension` s past the plan)
    EARLY  : end the current stage now; the next stage keeps it until the plan's
             own end of that stage (early return of unused green)

Whatever the policy does, the plan's absolute timing stays the reference: an
extension shortens the next stage, an early end lengthens it, and a junction that
falls two windows behind (e.g. after a shield override) jumps back to the plan.
So the junctions stay coordinated, as in a coordinated-actuated controller.

The SignalHead must be ready every second (decision_interval = 1) so the plan's
switching times are kept to the second; the policy is consulted only every
`interval` s.
"""

import numpy as np

from tsc.controllers.base import Controller
from tsc.detectors import e1_id, e2_id

FOLLOW, HOLD, EARLY = 0, 1, 2
N_ACTIONS = 3


class PlanActions:
    """Plan bookkeeping and the FOLLOW / HOLD / EARLY action semantics for all junctions."""

    n_actions = N_ACTIONS
    extra_dim = 6           # plan features appended to the observation
    needs_plan = True

    def __init__(self, network, interval=5, max_extension=20, guard=False):
        self.network = network
        self.interval = interval
        self.max_extension = max_extension
        # guard: a green may only be held if the next stage still gets its minimum green
        # within its planned window, and a junction behind the plan catches up one stage at a
        # time instead of skipping stages (plans with short stage windows, e.g. Maribor)
        self.guard = guard
        self.plans = None

    def set_plans(self, plans):
        """{tl_id: StagePlan} for the next episode (call before reset)."""
        self.plans = plans

    def reset(self, sim):
        if self.plans is None:
            raise RuntimeError('PlanActions needs set_plans() before an episode')
        self.count = {}
        for tl_id, plan in self.plans.items():
            n = plan.counter(0)
            # the head starts in greens[0]: the latest window of that stage at or before n
            back = next(d for d in range(plan.k) if plan.stage(n - d) == sim.heads[tl_id].stage) \
                if sim.heads[tl_id].stage in plan.stages else 0
            self.count[tl_id] = n - back
        self.last = {tl_id: FOLLOW for tl_id in self.plans}

    def rejoin(self, sim, tl_id):
        """Take over a junction another controller has been running: continue in the plan's
        current window, switching to its stage at once if the head shows another one."""
        plan = self.plans[tl_id]
        n = plan.counter(sim.t)
        self.count[tl_id] = n if sim.heads[tl_id].stage == plan.stage(n) else n - 1
        self.last[tl_id] = FOLLOW

    def tick(self, sim):
        pass

    # ------------------------------------------------------------------

    def decides(self, sim, tl_id):
        """True at the policy's decision points (every `interval` s of green after min green)."""
        head = sim.heads[tl_id]
        return (head.timer - head.min_green) % self.interval == 0

    def mask(self, sim, tl_id):
        plan, c, t = self.plans[tl_id], self.count[tl_id], sim.t
        n = plan.counter(t)
        late = t - plan.window_end(c)                 # > 0: past the plan's switch time
        mask = np.ones(N_ACTIONS, dtype=bool)
        # holding only matters when the planned switch is due within the next interval,
        # and only within the extension budget
        mask[HOLD] = -self.interval < late < self.max_extension and             self._room(sim, tl_id, c, t + self.interval)
        mask[EARLY] = c <= n and plan.k > 1           # not when already ahead of the plan
        return mask

    def stage(self, sim, tl_id, action):
        """Stage to request now, given the policy's latest action."""
        plan, t = self.plans[tl_id], sim.t
        n = plan.counter(t)
        c = self.count[tl_id]
        if action == EARLY and c <= n and plan.k > 1:
            c += 1
        elif c < n:
            late = t - plan.window_end(c)
            if action == HOLD and late < self.max_extension and self._room(sim, tl_id, c, t):
                pass
            elif n - c >= 2 and not self.guard:
                c = n                                 # fell behind: rejoin the plan
            else:
                c += 1
        self.count[tl_id] = c
        return plan.stage(c)

    def _room(self, sim, tl_id, c, t):
        """With the guard: would the stage after window c still get its minimum green within
        its planned window if the current green ends at time t?"""
        if not self.guard:
            return True
        head = sim.heads[tl_id]
        return t + head.yellow + head.all_red + head.min_green <= self.plans[tl_id].window_end(c + 1)

    def features(self, sim, tl_id):
        """[planned stage of now (scaled index), s to the planned switch, ahead/behind,
        s past the planned switch, planned split of the current stage, cycle position]."""
        plan, t = self.plans[tl_id], sim.t
        n = plan.counter(t)
        c = self.count[tl_id]
        k = max(1, len(self.network.greens[tl_id]) - 1)
        to_switch = plan.window_end(c) - t
        cycle_pos = ((t - plan.origin) % plan.cycle) / plan.cycle
        return np.array([
            plan.stage(n) / k,
            np.clip(to_switch / 60.0, -1.0, 1.0),
            float(np.clip(c - n, -1, 1)),
            np.clip(-to_switch / max(1, self.max_extension), 0.0, 1.0),
            plan.split(c) / 60.0,
            cycle_pos,
        ], dtype=np.float32)


class PlanController(Controller):
    """Python replay of a fixed-time plan (policy = always FOLLOW).  With the same plan it
    reproduces SUMO's own fixed-time run up to the intergreen details (sanity check), and
    it is the reference the plan-based RL is compared with."""

    needs_plan = True

    def __init__(self, network, interval=5, max_extension=20, guard=False):
        self.actions = PlanActions(network, interval, max_extension, guard=guard)

    def set_plans(self, plans):
        self.actions.set_plans(plans)

    def reset(self, sim):
        self.actions.reset(sim)

    def policy(self, sim, tl_id):
        return FOLLOW

    def act(self, sim, tl_id):
        a = self.actions
        if a.decides(sim, tl_id):
            choice = self.policy(sim, tl_id)
            a.last[tl_id] = choice if a.mask(sim, tl_id)[choice] else FOLLOW
        return a.stage(sim, tl_id, a.last[tl_id])


class PlanActuated(PlanController):
    """Coordinated-actuated control (classic, no learning): extend the green while vehicles
    keep arriving on the served lanes (gap < `gap` s at the stop-line loops) and end it
    early when the served lanes' detectors are empty and another stage has a queue."""

    def __init__(self, network, interval=5, max_extension=20, gap=3, guard=False):
        super().__init__(network, interval, max_extension, guard=guard)
        self.network = network
        self.gap = gap

    def reset(self, sim):
        super().reset(sim)
        self.since = {l: 99 for tl in self.network.tl_ids for l in self.network.in_lanes[tl]}
        self.served = {}
        for tl_id in self.network.tl_ids:
            links = self.network.links[tl_id]
            self.served[tl_id] = [sorted({lane for i, lane, _ in links if g[i] in 'Gg'})
                                  for g in self.network.greens[tl_id]]

    def tick(self, sim):
        for lane in self.since:
            if sim.api.inductionloop.getLastStepVehicleNumber(e1_id(lane)) > 0:
                self.since[lane] = 0
            else:
                self.since[lane] += 1

    def policy(self, sim, tl_id):
        stage = sim.heads[tl_id].stage
        lanes = self.served[tl_id][stage]
        if any(self.since[l] < self.gap for l in lanes):
            return HOLD
        api = sim.api
        if not any(api.lanearea.getLastStepVehicleNumber(e2_id(l)) for l in lanes):
            others = {l for s, ls in enumerate(self.served[tl_id]) if s != stage for l in ls}
            if any(api.lanearea.getJamLengthVehicle(e2_id(l)) > 0 for l in others):
                return EARLY
        return FOLLOW
