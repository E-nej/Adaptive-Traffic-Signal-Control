"""One SUMO run of a scenario, with either SUMO-native or Python-controlled signals.

tls modes
---------
'fixed'    : netconvert's default fixed-time plan (40 s greens), run by SUMO
'webster'  : Webster-optimised fixed-time plan (additional file), run by SUMO
'actuated' : SUMO's gap-based actuated control (detectors placed by SUMO)
'python'   : a Python controller picks green stages through SignalHead
"""

import os

from tsc import metrics, sumo_env
from tsc.detectors import detector_file
from tsc.signals import SignalHead

TLS_MODES = ('fixed', 'webster', 'actuated', 'python')


class Simulation:
    def __init__(self, network, route_file, duration, out_dir, seed=0, tls='python',
                 additional=None, gui=False, decision_interval=5, max_green=None, max_red=None,
                 e2_length=50.0):
        if tls not in TLS_MODES:
            raise ValueError(f'Unknown tls mode: {tls!r}')
        self.network = network
        self.duration = duration
        self.out_dir = out_dir
        self.tls = tls
        self.api = sumo_env.api(gui)
        os.makedirs(out_dir, exist_ok=True)
        self.tripinfo = os.path.join(out_dir, 'tripinfo.xml')
        self.statistics = os.path.join(out_dir, 'statistics.xml')

        if tls == 'actuated':
            if network.net_actuated is None:
                raise ValueError('The actuated baseline needs network.rebuild_tls = true')
            net = network.net_actuated
        else:
            net = network.net_file
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
            additional.append(detector_file(network, e2_length))
        if additional:
            self.cmd += ['-a', ','.join(additional)]
        self._head_args = dict(yellow=network.yellow, all_red=network.all_red,
                               min_green=network.min_green, decision_interval=decision_interval,
                               max_green=max_green, max_red=max_red)

    # ------------------------------------------------------------------

    def start(self):
        self.api.start(self.cmd)
        self.t = 0
        self._queue_sum = 0.0
        self.heads = {}
        if self.tls == 'python':
            set_state = self.api.trafficlight.setRedYellowGreenState
            self.heads = {
                tl_id: SignalHead(self.network.greens[tl_id],
                                  lambda state, tl_id=tl_id: set_state(tl_id, state),
                                  **self._head_args)
                for tl_id in self.network.tl_ids
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
            for tl_id in self.network.tl_ids for l in self.network.in_lanes[tl_id]
        ) / len(self.network.tl_ids)

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


def run_episode(network, controller, route_file, duration, out_dir, seed=0, tls='python',
                additional=None, gui=False, **sim_kwargs):
    """Run one episode with a non-learning controller (None for SUMO-native control)."""
    sim = Simulation(network, route_file, duration, out_dir, seed=seed, tls=tls,
                     additional=additional, gui=gui, **sim_kwargs).start()
    if controller is not None:
        controller.reset(sim)
    while not sim.done:
        if controller is not None:
            for tl_id in sim.decision_points():
                sim.heads[tl_id].request(controller.act(sim, tl_id))
        sim.step()
        if controller is not None:
            controller.tick(sim)
    return sim.close()
