"""Agent observations.

mode='grid'     : vehicle-position grid per incoming lane (perfect information,
                  like the paper but with the real lane length — fixes F1)
mode='detector' : only what roadside detectors measure: per incoming lane an
                  E2 area detector on the last 50 m (jam length, occupancy, mean
                  speed) and an E1 loop at the stop line (vehicles passed since
                  the last decision), optionally with noise and detector failures;
                  with approach_length > 0 also the vehicles moving toward the stop
                  line within approach_length m (a second, longer area detector)

layout='phase' (for the phase-wise network, agent.phase_net): one row per green stage,
padded to the largest junction: the summed lane features of the incoming lanes the stage
serves, the queue on the outgoing lanes it feeds ([queue ratio, measured]: detector mode
only where the outgoing lane is an approach of another signal), and the stage's signal
state ([current, green time, time since its last green]).  No messages.

Normalisation uses fixed physical constants (lane capacity, speed limit),
never the number of vehicles in the episode (fixes F5).

Messages (with comm) from each neighbour n to the receiving junction i:

message='aggregate'   (MSG_SIZE values, the original format) — n's state, the same
                      for every receiver: [queue ratio, occupancy, speed ratio, stage]
message='directional' (DIR_MSG_SIZE values, step 2E.2) — what n knows about i:
                      [queue on n's lanes whose movements lead to i,
                       vehicles n recently released toward i (stop-line loops, ~10 s),
                       share of n's links toward i that have green now,
                       how long n has been serving i,
                       queue on n's approach from i (spillback warning for i)]
Directional messages use n's detectors (E2 jam, E1 loops); in grid mode the
queues are lane halting counts.
"""

import numpy as np

from tsc.detectors import e1_id, e2_capacity, e2_id, e2a_id

VEH_SPACE = 7.5            # m per queued vehicle (length + gap)
CELL_BOUNDS = [7, 14, 21, 28, 40, 55, 70, 90, 110]   # m from the stop line → 10 cells
N_CELLS = len(CELL_BOUNDS) + 1
MSG_SIZE = 4
DIR_MSG_SIZE = 5
MESSAGE_FORMATS = ('aggregate', 'directional')
RELEASE_DECAY = 0.9       # per second: exponential memory of ~10 s for released vehicles


DET_FEATS = 4             # jam ratio, occupancy, speed ratio, passed vehicles
PASSED_NORM = 10.0        # vehicles per decision interval
LAYOUTS = ('flat', 'phase')
DOWN_FEATS = 2            # phase layout: downstream queue ratio, measured
STAGE_FEATS = 3           # phase layout: current, green time, time since last green


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
                 e2_length=50.0, message='aggregate', approach_length=0.0, layout='flat'):
        if mode not in ('grid', 'detector'):
            raise ValueError(mode)
        if layout not in LAYOUTS:
            raise ValueError(f'Unknown layout {layout!r}; choose from {LAYOUTS}')
        self.layout = layout
        if layout == 'phase':
            comm = False
        if message not in MESSAGE_FORMATS:
            raise ValueError(f'Unknown message format {message!r}; choose from {MESSAGE_FORMATS}')
        self.network = network
        self.mode = mode
        self.comm = comm
        self.message_format = message
        self.msg_size = DIR_MSG_SIZE if message == 'directional' else MSG_SIZE
        self.noise = noise
        self.dropout = dropout
        self.rng = np.random.default_rng(seed)
        self.cap = {l: max(1.0, network.lane_length[l] / VEH_SPACE) for l in network.lane_length}
        self.e2_cap = {l: e2_capacity(network, l, e2_length)
                       for tl_id in network.tl_ids for l in network.in_lanes[tl_id]}
        self.approach = mode == 'detector' and approach_length > 0
        self.det_feats = DET_FEATS + (1 if self.approach else 0)
        if self.approach:
            self.a_cap = {l: e2_capacity(network, l, approach_length)
                          for tl_id in network.tl_ids for l in network.in_lanes[tl_id]}
        per_lane = N_CELLS + 1 if mode == 'grid' else self.det_feats
        self.lane_dim = per_lane
        self.local_dim = network.max_in_lanes * per_lane + network.max_greens + 1
        self.dim = self.local_dim + (network.max_neighbours * self.msg_size if comm else 0)
        if layout == 'phase':
            self.dim = (network.max_greens, per_lane + DOWN_FEATS + STAGE_FEATS)
            self.served = {}       # tl -> [(incoming lanes, outgoing lanes) per stage]
            for tl_id in network.tl_ids:
                self.served[tl_id] = []
                for g in network.greens[tl_id]:
                    links = [(i, o) for idx, i, o in network.links[tl_id] if g[idx] in 'Gg']
                    self.served[tl_id].append((sorted({i for i, _ in links}),
                                               sorted({o for _, o in links})))
        self.passed = {l: 0 for tl_id in network.tl_ids for l in network.in_lanes[tl_id]}
        self.released = {l: 0.0 for l in self.passed}
        # lanes of src whose movements lead to dst (directional messages); toward holds
        # SUMO link indices, which skip numbers where a link is not modelled
        in_lane = {(t, i): lane for t in network.tl_ids for i, lane, _ in network.links[t]}
        self.toward_lanes = {
            (src, dst): sorted({in_lane[src, i] for i in idx})
            for (src, dst), idx in network.toward.items()
        }
        self.set_message_mode('real')

    def set_message_mode(self, mode, seed=0):
        if mode not in self.MESSAGE_MODES:
            raise ValueError(f'Unknown message mode {mode!r}; choose from {self.MESSAGE_MODES}')
        self.message_mode = mode
        self.message_rng = np.random.default_rng(seed)

    def reset(self, sim=None):
        self.passed = {l: 0 for l in self.passed}
        self.released = {l: 0.0 for l in self.released}

    def tick(self, sim):
        """Call once per simulation second: accumulates the stop-line loop counts."""
        directional = self.comm and self.message_format == 'directional'
        if self.mode != 'detector' and not directional:
            return
        for lane in self.passed:
            n = sim.api.inductionloop.getLastStepVehicleNumber(e1_id(lane))
            if self.mode == 'detector':
                self.passed[lane] += n
            if directional:
                self.released[lane] = RELEASE_DECAY * self.released[lane] + n

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
            return np.zeros(self.det_feats)     # failed detector reports nothing
        api = sim.api
        det = e2_id(lane)
        jam = api.lanearea.getJamLengthVehicle(det) / self.e2_cap[lane]
        occ = api.lanearea.getLastStepOccupancy(det) / 100.0
        speed = api.lanearea.getLastStepMeanSpeed(det)
        speed = 1.0 if speed < 0 else min(1.0, speed / self.network.lane_speed[lane])
        passed = self.passed[lane] / PASSED_NORM
        self.passed[lane] = 0
        feats = [jam, occ, speed, passed]
        if self.approach:                       # vehicles still moving toward the stop line
            a = e2a_id(lane)
            moving = (api.lanearea.getLastStepVehicleNumber(a)
                      - api.lanearea.getJamLengthVehicle(a))
            feats.append(max(0.0, moving) / self.a_cap[lane])
        feats = np.array(feats)
        if self.noise:
            feats = np.clip(feats + self.rng.normal(0.0, self.noise, len(feats)), 0.0, None)
        return feats

    def _lane(self, sim, lane):
        """Feature vector of one incoming lane (detector or grid)."""
        if self.mode == 'detector':
            return self._lane_detectors(sim, lane)
        api, length = sim.api, self.network.lane_length[lane]
        feats = np.zeros(N_CELLS + 1)
        for veh in api.lane.getLastStepVehicleIDs(lane):
            dist = length - api.vehicle.getLanePosition(veh)
            feats[np.searchsorted(CELL_BOUNDS, dist, side='right')] = 1.0
        feats[N_CELLS] = api.lane.getLastStepHaltingNumber(lane) / self.cap[lane]
        return feats

    def _downstream(self, sim, lane):
        """[queue ratio, measured] of an outgoing lane.  Detector mode: only lanes that are
        approaches of a signal have a detector (the neighbour's E2)."""
        if self.mode == 'grid':
            cap = self.cap.get(lane) or 1.0
            return [sim.api.lane.getLastStepHaltingNumber(lane) / cap, 1.0]
        if lane not in self.e2_cap or (self.dropout and self.rng.random() < self.dropout):
            return [0.0, 0.0]
        jam = sim.api.lanearea.getJamLengthVehicle(e2_id(lane)) / self.e2_cap[lane]
        if self.noise:
            jam = max(0.0, jam + self.rng.normal(0.0, self.noise))
        return [jam, 1.0]

    def phase_obs(self, sim, tl_id):
        head = sim.heads[tl_id]
        lanes = {l: self._lane(sim, l) for l in self.network.in_lanes[tl_id]}
        out = np.zeros(self.dim, dtype=np.float32)
        for k, (ins, outs) in enumerate(self.served[tl_id]):
            if outs:
                down = np.mean([self._downstream(sim, l) for l in outs], axis=0)
            else:
                down = np.zeros(DOWN_FEATS)
            current = k == head.stage
            signal = [float(current), min(head.green_time, 120) / 60.0 if current else 0.0,
                      0.0 if current else min(head.clock - head.green_end[k], 180) / 60.0]
            if ins:
                served = np.sum([lanes[l] for l in ins], axis=0)
            else:
                served = np.zeros(self.lane_dim)
            out[k] = np.concatenate([served, down, signal])
        return out

    def local(self, sim, tl_id):
        if self.mode == 'detector':
            lane_feats = np.zeros((self.network.max_in_lanes, self.det_feats))
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

    def _jam(self, sim, lanes):
        """Queued vehicles on `lanes` relative to what fits (E2 zone or whole lane)."""
        if not lanes:
            return 0.0
        api = sim.api
        if self.mode == 'detector':
            jam = sum(api.lanearea.getJamLengthVehicle(e2_id(l)) for l in lanes)
            return jam / sum(self.e2_cap[l] for l in lanes)
        return (sum(api.lane.getLastStepHaltingNumber(l) for l in lanes)
                / sum(self.cap[l] for l in lanes))

    def directional_message(self, sim, src, dst):
        """What neighbour `src` tells `dst` (see the module docstring)."""
        net = self.network
        lanes = self.toward_lanes.get((src, dst), [])
        links = net.toward.get((src, dst), [])
        head = sim.heads[src]
        serving = 0.0
        if links and head.mode == 'green':
            state = net.greens[src][head.stage]
            serving = sum(state[i] in 'Gg' for i in links) / len(links)
        released = sum(self.released[l] for l in lanes) / PASSED_NORM
        return [self._jam(sim, lanes), released, serving,
                min(head.green_time, 60) / 60.0 if serving else 0.0,
                self._jam(sim, net.arrival.get((dst, src), []))]

    def _message_to(self, sim, src, dst):
        if self.message_format == 'directional':
            return self.directional_message(sim, src, dst)
        return self.message(sim, src)

    def observe(self, sim, tl_id):
        if self.layout == 'phase':
            return self.phase_obs(sim, tl_id)
        parts = [self.local(sim, tl_id)]
        if self.comm:
            neighbours = self.network.neighbours[tl_id]
            size = self.msg_size
            if self.message_mode == 'zero':
                msgs = [[0.0] * size] * len(neighbours)
            elif self.message_mode == 'wrong':
                # a random other junction's message to one of *its* neighbours
                others = [t for t in self.network.tl_ids if t != tl_id]
                msgs = []
                for _ in neighbours:
                    src = self.message_rng.choice(others)
                    if self.message_format == 'directional':
                        dst = self.message_rng.choice(self.network.neighbours[src])
                        msgs.append(self.directional_message(sim, src, dst))
                    else:
                        msgs.append(self.message(sim, src))
            else:
                msgs = [self._message_to(sim, n, tl_id) for n in neighbours]
            msgs += [[0.0] * size] * (self.network.max_neighbours - len(msgs))
            parts.append(np.ravel(msgs))
        return np.concatenate(parts).astype(np.float32)

    def action_mask(self, tl_id):
        mask = np.zeros(self.network.max_greens, dtype=bool)
        mask[:len(self.network.greens[tl_id])] = True
        return mask
