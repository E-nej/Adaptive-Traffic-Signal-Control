"""Max-pressure control (Varaiya 2013), lane-level variant."""

import numpy as np

from tsc.controllers.base import Controller
from tsc.detectors import e2_id
from tsc.network import lane_edge
from tsc.observation import VEH_SPACE


class MaxPressure(Controller):
    """Pressure of a stage = vehicles on the incoming lanes it serves minus
    vehicles on the outgoing lanes it feeds.  At every decision point the
    stage with the highest pressure gets (or keeps) the green.

    Reads vehicle counts over whole approaches, i.e. perfect information.  Where an
    approach continues before the controlled edge (a lane split such as a left-turn
    bay), vehicles there count for the stage that serves their next movement, and the
    edge chain after an outgoing edge counts as downstream queue.
    """

    def reset(self, sim):
        net = sim.network
        self.served = {}
        self.movements = {}
        for tl_id in net.tl_ids:
            per_stage, moves = [], []
            for state in net.greens[tl_id]:
                ins, outs, mv = set(), set(), set()
                for idx, in_lane, out_lane in net.links[tl_id]:
                    if state[idx] in 'Gg':
                        ins.add(in_lane)
                        outs.add(out_lane)
                        mv.add((lane_edge(in_lane), lane_edge(out_lane)))
                per_stage.append((sorted(ins), sorted(outs)))
                moves.append(mv)
            self.served[tl_id] = per_stage
            self.movements[tl_id] = moves

    def count(self, sim, lane):
        return sim.api.lane.getLastStepVehicleNumber(lane)

    def upstream_counts(self, sim, tl_id):
        """Vehicles on the approach chains of tl_id, per stage serving their next movement."""
        net, api = sim.network, sim.api
        counts = [0] * len(self.served[tl_id])
        for in_edge, chain in self._chains(net, tl_id):
            for edge in chain:
                for veh in api.edge.getLastStepVehicleIDs(edge):
                    route = list(api.vehicle.getRoute(veh))
                    try:
                        k = route.index(in_edge, api.vehicle.getRouteIndex(veh))
                    except ValueError:
                        continue
                    if k + 1 >= len(route):
                        continue
                    move = (in_edge, route[k + 1])
                    for s, moves in enumerate(self.movements[tl_id]):
                        if move in moves:
                            counts[s] += 1
        return counts

    def downstream_count(self, sim, out_lanes):
        net, api = sim.network, sim.api
        edges = sorted({lane_edge(l) for l in out_lanes})
        return sum(api.edge.getLastStepVehicleNumber(e)
                   for out in edges for e in net.downstream_edges.get(out, []))

    @staticmethod
    def _chains(net, tl_id):
        edges = sorted({lane_edge(l) for l in net.in_lanes[tl_id]})
        return [(e, net.upstream_edges[e]) for e in edges if net.upstream_edges.get(e)]

    def act(self, sim, tl_id):
        up = self.upstream_counts(sim, tl_id)
        pressures = [
            sum(self.count(sim, l) for l in ins) + up[s]
            - sum(self.count(sim, l) for l in outs) - self.downstream_count(sim, outs)
            for s, (ins, outs) in enumerate(self.served[tl_id])
        ]
        current = sim.heads[tl_id].stage
        best = max(range(len(pressures)), key=lambda i: pressures[i])
        # keep the current stage on ties to avoid needless switching
        return current if pressures[current] >= pressures[best] else best


class DetectorMaxPressure(MaxPressure):
    """Max-pressure that only sees the E2 area detectors (same sensors as RL in step 2).

    Upstream count = vehicles in the E2 zone of a served incoming lane.
    Downstream count = vehicles in the E2 zones of the next junction's approach: the
    outgoing lane itself if it is a controlled approach lane there, else the lanes of
    the controlled edge at the end of the outgoing edge's chain (e.g. the next bay);
    0 if the road leaves the signalised network.
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

    def upstream_counts(self, sim, tl_id):
        return [0] * len(self.served[tl_id])      # detectors only cover the controlled lanes

    def downstream_count(self, sim, out_lanes):
        net = sim.network
        total = 0.0
        for out in sorted({lane_edge(l) for l in out_lanes}):
            chain = net.downstream_edges.get(out, [])
            if chain:
                last = net.sumo_net.getEdge(chain[-1])
                total += sum(self.count(sim, lane.getID()) for lane in last.getLanes())
        return total

    def count(self, sim, lane):
        if lane not in self.detected:
            return 0.0
        if self.dropout and self.rng.random() < self.dropout:
            return 0.0
        n = sim.api.lanearea.getLastStepVehicleNumber(e2_id(lane))
        if self.noise:
            n = max(0.0, n + self.rng.normal(0.0, self.noise * self.e2_cap[lane]))
        return n
