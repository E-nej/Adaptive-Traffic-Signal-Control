"""Signal controllers.

Baselines
---------
fixed       : netconvert's fixed-time plan, run by SUMO
webster     : Webster-optimised fixed-time plan, run by SUMO
actuated    : SUMO's gap-based actuated control
maxpressure : max-pressure, Python-controlled through SignalHead

RL policies (controllers.rl) are Python-controlled as well.
"""

from tsc.controllers.base import Controller
from tsc.controllers.maxpressure import MaxPressure

# name -> (tls mode of the simulation, Python controller factory or None)
BASELINES = {
    'fixed': ('fixed', None),
    'webster': ('webster', None),
    'actuated': ('actuated', None),
    'maxpressure': ('python', MaxPressure),
}


def baseline(name):
    """(tls mode, controller instance or None) of a baseline."""
    try:
        tls, factory = BASELINES[name]
    except KeyError:
        raise ValueError(f'Unknown baseline {name!r}; choose from {sorted(BASELINES)}') from None
    return tls, (factory() if factory else None)


__all__ = ['BASELINES', 'Controller', 'MaxPressure', 'baseline']
