"""Per-second rewards of one junction (averaged over a decision interval by the trainer)."""

from tsc.detectors import e2_id

KINDS = ('wait', 'multi', 'queue', 'det_queue')


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

    def __init__(self, network, kind):
        if kind not in KINDS:
            raise ValueError(f'Unknown reward {kind!r}; choose from {KINDS}')
        self.network = network
        self.kind = kind
        self.prev_wait = {tl_id: 0.0 for tl_id in network.tl_ids}

    def reset(self):
        self.prev_wait = {tl_id: 0.0 for tl_id in self.network.tl_ids}

    def __call__(self, sim, tl_id):
        api = sim.api
        lanes = self.network.in_lanes[tl_id]
        if self.kind == 'det_queue':
            return -sum(api.lanearea.getJamLengthVehicle(e2_id(l)) for l in lanes) / len(lanes)
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
