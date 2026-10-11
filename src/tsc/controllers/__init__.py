"""Signal controllers.

Baselines
---------
fixed           : netconvert's fixed-time plan, run by SUMO
webster         : Webster-optimised fixed-time plan, run by SUMO
actuated        : SUMO's gap-based actuated control
maxpressure     : max-pressure on whole-lane counts (perfect information), Python-controlled
maxpressure_det : max-pressure on the E2 detectors only, with the noise / dropout of the
                  experiment's [observation] section (same sensors as RL in step 2)

RL policies (controllers.rl) are Python-controlled as well.
"""

from tsc.controllers.base import Controller
from tsc.controllers.maxpressure import DetectorMaxPressure, MaxPressure


def _detector_maxpressure(cfg):
    o = cfg.observation
    return DetectorMaxPressure(noise=o.noise, dropout=o.dropout, e2_length=o.e2_length)


# name -> (tls mode of the simulation, factory(cfg) of the Python controller, or None)
BASELINES = {
    'fixed': ('fixed', None),
    'webster': ('webster', None),
    'actuated': ('actuated', None),
    'maxpressure': ('python', lambda cfg: MaxPressure()),
    'maxpressure_det': ('python', _detector_maxpressure),
}


def baseline(name, cfg=None):
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
    return tls, factory(cfg)


__all__ = ['BASELINES', 'Controller', 'DetectorMaxPressure', 'MaxPressure', 'baseline']
