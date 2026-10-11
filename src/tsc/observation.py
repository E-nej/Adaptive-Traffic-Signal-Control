"""Agent observations.

mode='grid'     : vehicle-position grid per incoming lane (perfect information,
                  like the paper but with the real lane length — fixes F1)
mode='detector' : only what roadside detectors measure: per incoming lane an
                  E2 area detector on the last 50 m (jam length, occupancy, mean
                  speed) and an E1 loop at the stop line (vehicles passed since
                  the last decision), optionally with noise and detector failures

Normalisation uses fixed physical constants (lane capacity, speed limit),
never the number of vehicles in the episode (fixes F5).
"""

import numpy as np

from tsc.detectors import e1_id, e2_id

VEH_SPACE = 7.5            # m per queued vehicle (length + gap)
CELL_BOUNDS = [7, 14, 21, 28, 40, 55, 70, 90, 110]   # m from the stop line → 10 cells
N_CELLS = len(CELL_BOUNDS) + 1
MSG_SIZE = 4


DET_FEATS = 4             # jam ratio, occupancy, speed ratio, passed vehicles
PASSED_NORM = 10.0        # vehicles per decision interval


class Observer:
    """Observation vector of every junction (padded to the largest junction).

    Layout: per incoming lane (grid: N_CELLS occupancy cells + halting ratio;
    detector: DET_FEATS), one-hot green stage, time in green, then — with
    comm — MSG_SIZE values per neighbour.

    message_mode is a test-time diagnostic (`tsc diagnose messages`):
    'real' (default), 'zero' (all messages 0), or 'wrong' (each neighbour slot
    gets the message of a random *other* junction).  If 'wrong' performs as
    well as 'real', the policy does not use neighbour-specific information.
    """

    MESSAGE_MODES = ('real', 'zero', 'wrong')

    def __init__(self, network, mode='grid', comm=True, noise=0.0, dropout=0.0, seed=0,
                 e2_length=50.0):
        if mode not in ('grid', 'detector'):
            raise ValueError(mode)
        self.network = network
        self.mode = mode
        self.comm = comm
        self.noise = noise
        self.dropout = dropout
        self.rng = np.random.default_rng(seed)
        self.cap = {l: max(1.0, network.lane_length[l] / VEH_SPACE) for l in network.lane_length}
        self.e2_cap = {l: max(1.0, min(e2_length, network.lane_length[l]) / VEH_SPACE)
                       for l in network.lane_length}
        per_lane = N_CELLS + 1 if mode == 'grid' else DET_FEATS
        self.local_dim = network.max_in_lanes * per_lane + network.max_greens + 1
        self.dim = self.local_dim + (network.max_neighbours * MSG_SIZE if comm else 0)
        self.passed = {l: 0 for tl_id in network.tl_ids for l in network.in_lanes[tl_id]}
        self.set_message_mode('real')

    def set_message_mode(self, mode, seed=0):
        if mode not in self.MESSAGE_MODES:
            raise ValueError(f'Unknown message mode {mode!r}; choose from {self.MESSAGE_MODES}')
        self.message_mode = mode
        self.message_rng = np.random.default_rng(seed)

    def reset(self, sim=None):
        self.passed = {l: 0 for l in self.passed}

    def tick(self, sim):
        """Call once per simulation second: accumulates the stop-line loop counts."""
        if self.mode != 'detector':
            return
        for lane in self.passed:
            self.passed[lane] += sim.api.inductionloop.getLastStepVehicleNumber(e1_id(lane))

    # ------------------------------------------------------------------

    def _phase_feats(self, sim, tl_id):
        head = sim.heads[tl_id]
        stage = np.zeros(self.network.max_greens)
        stage[head.stage] = 1.0
        return np.concatenate([stage, [min(head.green_time, 120) / 60.0]])

    def _lane_detectors(self, sim, lane):
        """[jam ratio, occupancy, speed ratio, passed] for one lane, with noise / failure."""
        if self.dropout and self.rng.random() < self.dropout:
            self.passed[lane] = 0
            return np.zeros(DET_FEATS)          # failed detector reports nothing
        api = sim.api
        det = e2_id(lane)
        jam = api.lanearea.getJamLengthVehicle(det) / self.e2_cap[lane]
        occ = api.lanearea.getLastStepOccupancy(det) / 100.0
        speed = api.lanearea.getLastStepMeanSpeed(det)
        speed = 1.0 if speed < 0 else min(1.0, speed / self.network.lane_speed[lane])
        passed = self.passed[lane] / PASSED_NORM
        self.passed[lane] = 0
        feats = np.array([jam, occ, speed, passed])
        if self.noise:
            feats = np.clip(feats + self.rng.normal(0.0, self.noise, DET_FEATS), 0.0, None)
        return feats

    def local(self, sim, tl_id):
        if self.mode == 'detector':
            lane_feats = np.zeros((self.network.max_in_lanes, DET_FEATS))
            for i, lane in enumerate(self.network.in_lanes[tl_id]):
                lane_feats[i] = self._lane_detectors(sim, lane)
            return np.concatenate([lane_feats.ravel(), self._phase_feats(sim, tl_id)])
        return self._grid_local(sim, tl_id)

    def _grid_local(self, sim, tl_id):
        api, network = sim.api, self.network
        lanes = network.in_lanes[tl_id]
        lane_feats = np.zeros((network.max_in_lanes, N_CELLS + 1))
        for i, lane in enumerate(lanes):
            length = network.lane_length[lane]
            for veh in api.lane.getLastStepVehicleIDs(lane):
                dist = length - api.vehicle.getLanePosition(veh)
                lane_feats[i, np.searchsorted(CELL_BOUNDS, dist, side='right')] = 1.0
            lane_feats[i, N_CELLS] = api.lane.getLastStepHaltingNumber(lane) / self.cap[lane]
        return np.concatenate([lane_feats.ravel(), self._phase_feats(sim, tl_id)])

    def message(self, sim, tl_id):
        """[queue ratio, occupancy ratio, mean speed ratio, stage] of a junction's approaches."""
        api, network = sim.api, self.network
        lanes = network.in_lanes[tl_id]
        n_greens = len(network.greens[tl_id])
        stage = sim.heads[tl_id].stage / max(1, n_greens - 1)
        if self.mode == 'detector':
            jam = sum(api.lanearea.getJamLengthVehicle(e2_id(l)) for l in lanes)
            occ = np.mean([api.lanearea.getLastStepOccupancy(e2_id(l)) / 100.0 for l in lanes])
            speeds = [api.lanearea.getLastStepMeanSpeed(e2_id(l)) / network.lane_speed[l] for l in lanes]
            speed = np.mean([1.0 if v < 0 else min(1.0, v) for v in speeds])
            return [jam / sum(self.e2_cap[l] for l in lanes), occ, speed, stage]
        cap = sum(self.cap[l] for l in lanes)
        halting = sum(api.lane.getLastStepHaltingNumber(l) for l in lanes)
        count = sum(api.lane.getLastStepVehicleNumber(l) for l in lanes)
        speed = np.mean([api.lane.getLastStepMeanSpeed(l) / network.lane_speed[l] for l in lanes])
        return [halting / cap, count / cap, min(1.0, speed), stage]

    def observe(self, sim, tl_id):
        parts = [self.local(sim, tl_id)]
        if self.comm:
            neighbours = self.network.neighbours[tl_id]
            if self.message_mode == 'zero':
                msgs = [[0.0] * MSG_SIZE] * len(neighbours)
            elif self.message_mode == 'wrong':
                others = [t for t in self.network.tl_ids if t != tl_id]
                msgs = [self.message(sim, self.message_rng.choice(others)) for _ in neighbours]
            else:
                msgs = [self.message(sim, n) for n in neighbours]
            msgs += [[0.0] * MSG_SIZE] * (self.network.max_neighbours - len(msgs))
            parts.append(np.ravel(msgs))
        return np.concatenate(parts).astype(np.float32)

    def action_mask(self, tl_id):
        mask = np.zeros(self.network.max_greens, dtype=bool)
        mask[:len(self.network.greens[tl_id])] = True
        return mask
