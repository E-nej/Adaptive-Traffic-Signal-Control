"""Per-second rewards of one junction (averaged over a decision interval by the trainer)."""

from tsc.detectors import e2_id

KINDS = ('wait', 'multi', 'queue', 'det_queue', 'delay', 'queue_pressure')


class Reward:
    """Per-second reward of one junction, averaged over a decision interval by the caller.

    kind='wait'  : W(t-1) - W(t), change of accumulated waiting time on the
                   approaches (the paper's base reward)
    kind='multi' : -(0.4*ΔW + 0.2*stops + 0.2*queue + 1e-4*CO2) (the paper's extended
                   reward; stops and queue are both the halting count, as in the paper)
    kind='queue' : -halting vehicles on the approaches
    kind='det_queue' : -jam length (vehicles) seen by the E2 detectors, i.e. a reward
                   a real installation can compute
    kind='delay' : -time lost on the approaches this second, Σ n·(1 - v̄/v_max) per lane
                   (the delay the evaluation measures: it also counts slowing down, which
                   'queue' and 'wait' miss, e.g. the stops a green wave saves)
    kind='queue_pressure' : 'queue' - pressure_weight·|halting on the controlled lanes -
                   halting on the exit lanes| (pressure: helps near saturation, when
                   releasing a queue into a full exit only moves it)
    All are divided by the number of incoming lanes.  Ground-truth rewards cover the whole
    approach: the controlled lanes plus the edge chain before them (network.approach_edges,
    e.g. the road before a left-turn bay).
    """

    def __init__(self, network, kind, neighbour_weight=0.0, pressure_weight=0.25):
        if kind not in KINDS:
            raise ValueError(f'Unknown reward {kind!r}; choose from {KINDS}')
        self.network = network
        self.kind = kind
        self.neighbour_weight = neighbour_weight
        self.pressure_weight = pressure_weight
        self.prev_wait = {tl_id: 0.0 for tl_id in network.tl_ids}
        self.v_max = {}                   # lane -> speed limit (kind 'delay')

    def reset(self):
        self.prev_wait = {tl_id: 0.0 for tl_id in self.network.tl_ids}

    def all(self, sim):
        """Per-second reward of every junction: r_i + α·Σ_{n ∈ neighbours(i)} r_n.

        With α > 0 (spatial discount, Chu et al. 2019) an agent is also rewarded for its
        neighbours' traffic, e.g. for not sending traffic into a queue that spills back.
        """
        raw = {tl_id: self(sim, tl_id) for tl_id in self.network.tl_ids}
        if not self.neighbour_weight:
            return raw
        return {tl_id: r + self.neighbour_weight * sum(raw[n] for n in self.network.neighbours[tl_id])
                for tl_id, r in raw.items()}

    def __call__(self, sim, tl_id):
        """Own (local) per-second reward of one junction."""
        api = sim.api
        lanes = self.network.in_lanes[tl_id]
        if self.kind == 'det_queue':
            return -sum(api.lanearea.getJamLengthVehicle(e2_id(l)) for l in lanes) / len(lanes)
        chain = self.network.approach_edges[tl_id]
        if self.kind == 'delay':
            lost = (sum(self._lost(api.lane, l, self._speed(api, l)) for l in lanes)
                    + sum(self._lost(api.edge, e, self._speed(api, f'{e}_0')) for e in chain))
            return -lost / len(lanes)
        halting = sim.halting()[tl_id]          # computed once per second by the simulation
        if self.kind == 'queue':
            r = -halting
        elif self.kind == 'queue_pressure':
            inside = sum(api.lane.getLastStepHaltingNumber(l) for l in lanes)
            out = sum(api.lane.getLastStepHaltingNumber(l)
                      for l in self.network.out_lanes[tl_id])
            r = -halting - self.pressure_weight * abs(inside - out)
        else:
            wait = (sum(api.lane.getWaitingTime(l) for l in lanes)
                    + sum(api.edge.getWaitingTime(e) for e in chain))
            d_wait = wait - self.prev_wait[tl_id]
            self.prev_wait[tl_id] = wait
            if self.kind == 'wait':
                r = -d_wait
            elif self.kind == 'multi':
                co2 = (sum(api.lane.getCO2Emission(l) for l in lanes)
                       + sum(api.edge.getCO2Emission(e) for e in chain))
                r = -(0.4 * d_wait + 0.2 * halting + 0.2 * halting + 1e-4 * co2)
            else:
                raise ValueError(self.kind)
        return r / len(lanes)

    def _speed(self, api, lane):
        if lane not in self.v_max:
            self.v_max[lane] = max(1.0, api.lane.getMaxSpeed(lane))
        return self.v_max[lane]

    @staticmethod
    def _lost(domain, item, v_max):
        """Time lost this second by the vehicles on a lane / edge: n·(1 - v̄/v_max)."""
        n = domain.getLastStepVehicleNumber(item)
        if not n:
            return 0.0
        return n * max(0.0, 1.0 - domain.getLastStepMeanSpeed(item) / v_max)
