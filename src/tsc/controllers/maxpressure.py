"""Max-pressure control (Varaiya 2013), lane-level variant."""

import numpy as np

from tsc.controllers.base import Controller
from tsc.detectors import e2_id
from tsc.observation import VEH_SPACE


class MaxPressure(Controller):
    """Pressure of a stage = vehicles on the incoming lanes it serves minus
    vehicles on the outgoing lanes it feeds.  At every decision point the
    stage with the highest pressure gets (or keeps) the green.

    Reads whole-lane vehicle counts, i.e. perfect information.
    """

    def reset(self, sim):
        net = sim.network
        self.served = {}
        for tl_id in net.tl_ids:
            per_stage = []
            for state in net.greens[tl_id]:
                ins, outs = set(), set()
                for idx, in_lane, out_lane in net.links[tl_id]:
                    if state[idx] in 'Gg':
                        ins.add(in_lane)
                        outs.add(out_lane)
                per_stage.append((sorted(ins), sorted(outs)))
            self.served[tl_id] = per_stage

    def count(self, sim, lane):
        return sim.api.lane.getLastStepVehicleNumber(lane)

    def act(self, sim, tl_id):
        pressures = [
            sum(self.count(sim, l) for l in ins) - sum(self.count(sim, l) for l in outs)
            for ins, outs in self.served[tl_id]
        ]
        current = sim.heads[tl_id].stage
        best = max(range(len(pressures)), key=lambda i: pressures[i])
        # keep the current stage on ties to avoid needless switching
        return current if pressures[current] >= pressures[best] else best


class DetectorMaxPressure(MaxPressure):
    """Max-pressure that only sees the E2 area detectors (same sensors as RL in step 2).

    Upstream count = vehicles in the E2 zone of a served incoming lane.
    Downstream count = vehicles in the E2 zone of the outgoing lane if that lane
    is an approach of another signalised junction (so it has a detector), else 0.
    Noise / dropout follow the RL detector model: Gaussian noise of
    `noise` × E2 capacity (vehicles), and a failed detector reads 0.
    """

    def __init__(self, noise=0.0, dropout=0.0, e2_length=50.0, seed=0):
        self.noise = noise
        self.dropout = dropout
        self.e2_length = e2_length
        self.seed = seed

    def reseed(self, seed):
        """Independent sensor noise per evaluation scenario."""
        self.seed = seed

    def reset(self, sim):
        super().reset(sim)
        net = sim.network
        self.rng = np.random.default_rng(self.seed)
        self.detected = {l for tl_id in net.tl_ids for l in net.in_lanes[tl_id]}
        self.e2_cap = {l: max(1.0, min(self.e2_length, net.lane_length[l]) / VEH_SPACE)
                       for l in self.detected}

    def count(self, sim, lane):
        if lane not in self.detected:
            return 0.0
        if self.dropout and self.rng.random() < self.dropout:
            return 0.0
        n = sim.api.lanearea.getLastStepVehicleNumber(e2_id(lane))
        if self.noise:
            n = max(0.0, n + self.rng.normal(0.0, self.noise * self.e2_cap[lane]))
        return n
