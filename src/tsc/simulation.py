"""One SUMO run of a scenario, with either SUMO-native or Python-controlled signals.

tls modes
---------
'fixed'    : netconvert's default fixed-time plan (40 s greens), run by SUMO
'webster'  : Webster-optimised fixed-time plan (additional file), run by SUMO
'actuated' : SUMO's gap-based actuated control (detectors placed by SUMO)
'python'   : a Python controller picks green stages through SignalHead
"""

import os

import helpers  # noqa: F401  (SUMO_HOME / PATH setup)
import metrics
from signals import SignalHead


def _api(gui):
    if gui or os.environ.get('RTSC_TRACI') == '1':
        import traci  # type: ignore
        return traci
    import libsumo  # type: ignore
    return libsumo


class Simulation:
    def __init__(self, scn, route_file, duration, out_dir, seed=0, tls='python',
                 additional=None, gui=False, decision_interval=5, max_green=None, max_red=None):
        self.scn = scn
        self.duration = duration
        self.out_dir = out_dir
        self.tls = tls
        self.api = _api(gui)
        os.makedirs(out_dir, exist_ok=True)
        self.tripinfo = os.path.join(out_dir, 'tripinfo.xml')
        self.statistics = os.path.join(out_dir, 'statistics.xml')

        net = scn.net_actuated if tls == 'actuated' else scn.net_file
        self.cmd = [
            'sumo-gui' if gui else 'sumo',
            '-n', net, '-r', route_file,
            '--begin', '0', '--end', str(duration),
            '--seed', str(int(seed)),
            '--no-step-log', 'true', '--no-warnings', 'true',
            '--duration-log.disable', 'true',
            '--time-to-teleport', '300',
            '--tripinfo-output', self.tripinfo,
            '--tripinfo-output.write-unfinished', 'true',
            '--device.emissions.probability', '1',
            '--statistic-output', self.statistics,
        ]
        additional = list(additional or [])
        if tls == 'python':
            additional.append(scn.detector_file())
        if additional:
            self.cmd += ['-a', ','.join(additional)]
        self._head_args = dict(decision_interval=decision_interval, max_green=max_green,
                               max_red=max_red)

    # ------------------------------------------------------------------

    def start(self):
        self.api.start(self.cmd)
        self.t = 0
        self._queue_sum = 0.0
        self.heads = {}
        if self.tls == 'python':
            self.heads = {
                tl_id: SignalHead(self.api, tl_id, self.scn.greens[tl_id], **self._head_args)
                for tl_id in self.scn.tl_ids
            }
        return self

    def decision_points(self):
        return [tl_id for tl_id, h in self.heads.items() if h.ready()]

    def step(self):
        self.api.simulationStep()
        self.t += 1
        for h in self.heads.values():
            h.tick()
        self._queue_sum += sum(
            self.api.lane.getLastStepHaltingNumber(l)
            for tl_id in self.scn.tl_ids for l in self.scn.in_lanes[tl_id]
        ) / len(self.scn.tl_ids)

    @property
    def done(self):
        return self.t >= self.duration

    def close(self):
        """Close SUMO and return the episode metrics."""
        forced = sum(h.forced for h in self.heads.values())
        self.api.close()
        out = metrics.summarise(self.tripinfo, self.statistics,
                                mean_queue=self._queue_sum / max(1, self.t))
        out['max_green_overrides'] = forced
        return out


def run_controller(scn, controller, route_file, duration, out_dir, seed=0, tls='python',
                   additional=None, gui=False, **sim_kwargs):
    """Run one episode with a non-learning controller (or SUMO-native control)."""
    sim = Simulation(scn, route_file, duration, out_dir, seed=seed, tls=tls,
                     additional=additional, gui=gui, **sim_kwargs).start()
    if controller is not None:
        controller.reset(sim)
    while not sim.done:
        if controller is not None:
            for tl_id in sim.decision_points():
                sim.heads[tl_id].request(controller.act(sim, tl_id))
        sim.step()
        if hasattr(controller, 'tick'):
            controller.tick(sim)
    return sim.close()
