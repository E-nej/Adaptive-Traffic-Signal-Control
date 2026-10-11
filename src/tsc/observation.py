"""Agent observations and rewards for the realism experiments.

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

VEH_SPACE = 7.5            # m per queued vehicle (length + gap)
CELL_BOUNDS = [7, 14, 21, 28, 40, 55, 70, 90, 110]   # m from the stop line → 10 cells
N_CELLS = len(CELL_BOUNDS) + 1
MSG_SIZE = 4


DET_FEATS = 4             # jam ratio, occupancy, speed ratio, passed vehicles
E2_LENGTH = 50.0
PASSED_NORM = 10.0        # vehicles per decision interval


class Observer:
    def __init__(self, scn, mode='grid', comm=True, noise=0.0, dropout=0.0, seed=0):
        if mode not in ('grid', 'detector'):
            raise ValueError(mode)
        self.scn = scn
        self.mode = mode
        self.comm = comm
        self.noise = noise
        self.dropout = dropout
        self.rng = np.random.default_rng(seed)
        self.cap = {l: max(1.0, scn.lane_length[l] / VEH_SPACE) for l in scn.lane_length}
        self.e2_cap = {l: max(1.0, min(E2_LENGTH, scn.lane_length[l]) / VEH_SPACE)
                       for l in scn.lane_length}
        per_lane = N_CELLS + 1 if mode == 'grid' else DET_FEATS
        self.local_dim = scn.max_in_lanes * per_lane + scn.max_greens + 1
        self.dim = self.local_dim + (scn.max_neighbours * MSG_SIZE if comm else 0)
        self.passed = {l: 0 for tl_id in scn.tl_ids for l in scn.in_lanes[tl_id]}

    def reset(self, sim=None):
        self.passed = {l: 0 for l in self.passed}

    def tick(self, sim):
        """Call once per simulation second: accumulates the stop-line loop counts."""
        if self.mode != 'detector':
            return
        for lane in self.passed:
            self.passed[lane] += sim.api.inductionloop.getLastStepVehicleNumber(f'e1_{lane}')

    # ------------------------------------------------------------------

    def _phase_feats(self, sim, tl_id):
        head = sim.heads[tl_id]
        stage = np.zeros(self.scn.max_greens)
        stage[head.stage] = 1.0
        return np.concatenate([stage, [min(head.green_time, 120) / 60.0]])

    def _lane_detectors(self, sim, lane):
        """[jam ratio, occupancy, speed ratio, passed] for one lane, with noise / failure."""
        if self.dropout and self.rng.random() < self.dropout:
            self.passed[lane] = 0
            return np.zeros(DET_FEATS)          # failed detector reports nothing
        api = sim.api
        det = f'e2_{lane}'
        jam = api.lanearea.getJamLengthVehicle(det) / self.e2_cap[lane]
        occ = api.lanearea.getLastStepOccupancy(det) / 100.0
        speed = api.lanearea.getLastStepMeanSpeed(det)
        speed = 1.0 if speed < 0 else min(1.0, speed / self.scn.lane_speed[lane])
        passed = self.passed[lane] / PASSED_NORM
        self.passed[lane] = 0
        feats = np.array([jam, occ, speed, passed])
        if self.noise:
            feats = np.clip(feats + self.rng.normal(0.0, self.noise, DET_FEATS), 0.0, None)
        return feats

    def local(self, sim, tl_id):
        if self.mode == 'detector':
            lane_feats = np.zeros((self.scn.max_in_lanes, DET_FEATS))
            for i, lane in enumerate(self.scn.in_lanes[tl_id]):
                lane_feats[i] = self._lane_detectors(sim, lane)
            return np.concatenate([lane_feats.ravel(), self._phase_feats(sim, tl_id)])
        return self._grid_local(sim, tl_id)

    def _grid_local(self, sim, tl_id):
        api, scn = sim.api, self.scn
        lanes = scn.in_lanes[tl_id]
        lane_feats = np.zeros((scn.max_in_lanes, N_CELLS + 1))
        for i, lane in enumerate(lanes):
            length = scn.lane_length[lane]
            for veh in api.lane.getLastStepVehicleIDs(lane):
                dist = length - api.vehicle.getLanePosition(veh)
                lane_feats[i, np.searchsorted(CELL_BOUNDS, dist, side='right')] = 1.0
            lane_feats[i, N_CELLS] = api.lane.getLastStepHaltingNumber(lane) / self.cap[lane]
        return np.concatenate([lane_feats.ravel(), self._phase_feats(sim, tl_id)])

    def message(self, sim, tl_id):
        """[queue ratio, occupancy ratio, mean speed ratio, stage] of a junction's approaches."""
        api, scn = sim.api, self.scn
        lanes = scn.in_lanes[tl_id]
        n_greens = len(scn.greens[tl_id])
        stage = sim.heads[tl_id].stage / max(1, n_greens - 1)
        if self.mode == 'detector':
            jam = sum(api.lanearea.getJamLengthVehicle(f'e2_{l}') for l in lanes)
            occ = np.mean([api.lanearea.getLastStepOccupancy(f'e2_{l}') / 100.0 for l in lanes])
            speeds = [api.lanearea.getLastStepMeanSpeed(f'e2_{l}') / scn.lane_speed[l] for l in lanes]
            speed = np.mean([1.0 if v < 0 else min(1.0, v) for v in speeds])
            return [jam / sum(self.e2_cap[l] for l in lanes), occ, speed, stage]
        cap = sum(self.cap[l] for l in lanes)
        halting = sum(api.lane.getLastStepHaltingNumber(l) for l in lanes)
        count = sum(api.lane.getLastStepVehicleNumber(l) for l in lanes)
        speed = np.mean([api.lane.getLastStepMeanSpeed(l) / scn.lane_speed[l] for l in lanes])
        return [halting / cap, count / cap, min(1.0, speed), stage]

    def observe(self, sim, tl_id):
        parts = [self.local(sim, tl_id)]
        if self.comm:
            msgs = [self.message(sim, n) for n in self.scn.neighbours[tl_id]]
            msgs += [[0.0] * MSG_SIZE] * (self.scn.max_neighbours - len(msgs))
            parts.append(np.ravel(msgs))
        return np.concatenate(parts).astype(np.float32)

    def action_mask(self, tl_id):
        mask = np.zeros(self.scn.max_greens, dtype=bool)
        mask[:len(self.scn.greens[tl_id])] = True
        return mask


class Reward:
    """Per-second reward of one junction, averaged over a decision interval by the caller.

    kind='wait'  : W(t-1) - W(t), change of accumulated waiting time on the
                   approaches (the paper's base reward)
    kind='multi' : -(0.4*ΔW + 0.2*stops + 0.2*queue + 1e-4*CO2) (the paper's extended
                   reward; stops and queue are both the halting count, as in the paper)
    kind='queue' : -halting vehicles on the approaches
    kind='det_queue' : -jam length (vehicles) seen by the E2 detectors, i.e. a reward
                   a real installation can compute
    All are divided by the number of incoming lanes.
    """

    def __init__(self, scn, kind):
        self.scn = scn
        self.kind = kind
        self.prev_wait = {tl_id: 0.0 for tl_id in scn.tl_ids}

    def reset(self):
        self.prev_wait = {tl_id: 0.0 for tl_id in self.scn.tl_ids}

    def __call__(self, sim, tl_id):
        api = sim.api
        lanes = self.scn.in_lanes[tl_id]
        if self.kind == 'det_queue':
            return -sum(api.lanearea.getJamLengthVehicle(f'e2_{l}') for l in lanes) / len(lanes)
        halting = sum(api.lane.getLastStepHaltingNumber(l) for l in lanes)
        if self.kind == 'queue':
            r = -halting
        else:
            wait = sum(api.lane.getWaitingTime(l) for l in lanes)
            d_wait = wait - self.prev_wait[tl_id]
            self.prev_wait[tl_id] = wait
            if self.kind == 'wait':
                r = -d_wait
            elif self.kind == 'multi':
                co2 = sum(api.lane.getCO2Emission(l) for l in lanes)
                r = -(0.4 * d_wait + 0.2 * halting + 0.2 * halting + 1e-4 * co2)
            else:
                raise ValueError(self.kind)
        return r / len(lanes)
