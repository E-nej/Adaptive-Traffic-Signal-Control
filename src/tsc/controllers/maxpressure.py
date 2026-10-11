"""Max-pressure control (Varaiya 2013), lane-level variant."""

from tsc.controllers.base import Controller


class MaxPressure(Controller):
    """Pressure of a stage = vehicles on the incoming lanes it serves minus
    vehicles on the outgoing lanes it feeds.  At every decision point the
    stage with the highest pressure gets (or keeps) the green.
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

    def act(self, sim, tl_id):
        count = sim.api.lane.getLastStepVehicleNumber
        pressures = [
            sum(count(l) for l in ins) - sum(count(l) for l in outs)
            for ins, outs in self.served[tl_id]
        ]
        current = sim.heads[tl_id].stage
        best = max(range(len(pressures)), key=lambda i: pressures[i])
        # keep the current stage on ties to avoid needless switching
        return current if pressures[current] >= pressures[best] else best
