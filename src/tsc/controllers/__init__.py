"""Signal controllers.

Baselines
---------
fixed           : netconvert's fixed-time plan, run by SUMO
webster         : Webster-optimised fixed-time plan, run by SUMO
greenwave       : coordinated fixed time (common cycle + offsets from tlsCoordinator)
actuated        : SUMO's gap-based actuated control
maxpressure     : max-pressure on whole-lane counts (perfect information), Python-controlled
maxpressure_det : max-pressure on the E2 detectors only, with the noise / dropout of the
                  experiment's [observation] section (same sensors as RL in step 2)
advanced_mp     : Advanced-MP (Zhang et al. 2022): efficient pressure, keeps the green while
                  enough vehicles are still moving toward the stop line ([signals] amp_weight,
                  amp_range); perfect information
greenwave_py    : the green-wave plan replayed by Python (SignalHead), the reference for
                  plan-controlled RL (same switching mechanics)
plan_actuated   : coordinated-actuated control on the green-wave plan (gap-based extension,
                  early end of empty greens; controllers.plan)
select_assigned : best fixed controller per junction and demand level (`tsc assign`,
                  controllers.select.FixedSelection)
actuated_py     : the Python copy of SUMO's actuated control that the selector uses as its
                  'actuated' expert (controllers.select.GapActuated), to check it against
                  the SUMO original

RL policies (controllers.rl) are Python-controlled as well.
"""

from tsc.controllers.base import Controller
from tsc.controllers.maxpressure import AdvancedMaxPressure, DetectorMaxPressure, MaxPressure
from tsc.controllers.plan import PlanActuated, PlanController


def _detector_maxpressure(cfg, network):
    o = cfg.observation
    return DetectorMaxPressure(noise=o.noise, dropout=o.dropout, e2_length=o.e2_length)


def _gap_actuated(cfg, network):
    from tsc.controllers.select import GapActuated
    return GapActuated(network, loops=cfg.signals.actuated_loops)


def _plan(cls):
    def factory(cfg, network):
        s = cfg.signals
        return cls(network, interval=s.decision_interval, max_extension=s.max_extension)
    return factory


# name -> (tls mode of the simulation, factory(cfg, network) of the Python controller, or None)
BASELINES = {
    'fixed': ('fixed', None),
    'webster': ('webster', None),
    'greenwave': ('webster', None),      # also a fixed plan from an additional file
    'actuated': ('actuated', None),
    'maxpressure': ('python', lambda cfg, network: MaxPressure()),
    'maxpressure_det': ('python', _detector_maxpressure),
    'advanced_mp': ('python', lambda cfg, network: AdvancedMaxPressure(
        weight=cfg.signals.amp_weight, effective_range=cfg.signals.amp_range)),
    'greenwave_py': ('python', _plan(PlanController)),
    'plan_actuated': ('python', _plan(PlanActuated)),
    'select_assigned': ('python', None),       # built by evaluate (needs the assignment)
    'actuated_py': ('python', lambda cfg, network: _gap_actuated(cfg, network)),
}

# Python baselines that run on the green-wave plan (signals.control = 'plan' timing)
PLAN_BASELINES = ('greenwave_py', 'plan_actuated', 'select_assigned')
# Python baselines that need a SignalHead ready every second, without the shield
SECOND_BASELINES = PLAN_BASELINES + ('actuated_py',)


def baseline(name, cfg=None, network=None):
    """(tls mode, controller instance or None) of a baseline; cfg defaults to Config()."""
    try:
        tls, factory = BASELINES[name]
    except KeyError:
        raise ValueError(f'Unknown baseline {name!r}; choose from {sorted(BASELINES)}') from None
    if factory is None:
        return tls, None
    if cfg is None:
        from tsc.config import Config
        cfg = Config()
    return tls, factory(cfg, network)


__all__ = ['BASELINES', 'Controller', 'DetectorMaxPressure', 'MaxPressure', 'PLAN_BASELINES',
           'PlanActuated', 'PlanController', 'SECOND_BASELINES', 'baseline']
