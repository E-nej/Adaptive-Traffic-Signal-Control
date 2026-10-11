"""Interface of Python signal controllers.

A controller only picks the next green stage of a junction; SignalHead
(signals.py) turns that into a safe signal sequence.  Called by the episode
loop as: reset(sim) once, act(sim, tl_id) at every decision point of a
junction, tick(sim) after every simulation second.
"""


class Controller:
    def reset(self, sim):
        pass

    def act(self, sim, tl_id):
        """Index of the green stage the junction should show next."""
        raise NotImplementedError

    def tick(self, sim):
        pass
