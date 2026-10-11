"""Fixed-time signal plans as stage schedules, for Python control on top of a plan.

A plan file is a SUMO additional file with one <tlLogic> per junction (Webster,
green wave).  Its green phases are the junction's green stages (network.greens);
the phases between them are intergreens.  SUMO runs a program with cycle C and
offset o so that cycle time 0 falls on t = o (mod C).

For Python control (SignalHead) the schedule is expressed as *windows*: stage
k+1's window starts when stage k's green ends (its intergreen is when the
signal head switches), so "follow the plan" means "request the stage whose
window contains the current time".  Windows are numbered with an absolute
counter n = cycle·K + k, so a controller can tell whether it is ahead of or
behind the plan.
"""

import math
import xml.etree.ElementTree as ET


class StagePlan:
    """One junction's fixed-time plan.

    cycle, offset : s
    stages        : [(stage index, green start, green end)] in cycle time, in order
    """

    def __init__(self, cycle, offset, stages):
        if not stages:
            raise ValueError('A plan needs at least one green stage')
        self.cycle = float(cycle)
        self.offset = float(offset)
        self.stages = [s for s, _, _ in stages]
        self.green = [(start, end) for _, start, end in stages]
        self.k = len(stages)
        last_end = stages[-1][2]
        # window k starts at the end of stage k-1's green; window 0 starts at origin
        self.origin = self.offset + last_end
        self.bounds = [((end - last_end) % self.cycle) for _, _, end in stages[:-1]]

    def counter(self, t):
        """Absolute window number n = cycle·K + k of time t."""
        u = t - self.origin
        cyc = math.floor(u / self.cycle)
        r = u - cyc * self.cycle
        k = sum(1 for b in self.bounds if b <= r)
        return cyc * self.k + k

    def stage(self, n):
        """Green stage (index into network.greens) of window n."""
        return self.stages[n % self.k]

    def window_end(self, n):
        """Absolute time at which window n ends (its stage's green ends)."""
        cyc, k = divmod(n, self.k)
        end = self.bounds[k] if k < self.k - 1 else self.cycle
        return self.origin + cyc * self.cycle + end

    def split(self, n):
        """Planned green time of window n's stage (s)."""
        start, end = self.green[n % self.k]
        return (end - start) % self.cycle


def load_plans(path, network):
    """{tl_id: StagePlan} of every junction of `network` in the plan file."""
    plans = {}
    for tl in ET.parse(path).getroot().iter('tlLogic'):
        tl_id = tl.get('id')
        if tl_id not in network.greens:
            continue
        greens = network.greens[tl_id]
        t = 0.0
        stages = []
        for ph in tl.iter('phase'):
            dur = float(ph.get('duration'))
            state = ph.get('state')
            if dur <= 0:
                raise ValueError(f'{path}: phase without duration in {tl_id}')
            if 'y' not in state and any(c in 'Gg' for c in state):
                if state not in greens:
                    raise ValueError(f'{path}: phase {state!r} of {tl_id} is not a green stage '
                                     f'of the network')
                if stages and stages[-1][0] == greens.index(state) and stages[-1][2] == t:
                    stages[-1] = (stages[-1][0], stages[-1][1], t + dur)   # split phase
                else:
                    stages.append((greens.index(state), t, t + dur))
            t += dur
        plans[tl_id] = StagePlan(t, float(tl.get('offset', 0.0)), stages)
    missing = set(network.tl_ids) - set(plans)
    if missing:
        raise ValueError(f'{path} has no plan for {sorted(missing)}')
    return plans
