"""Signal timing for TraCI-controlled junctions.

Controllers only choose the next green stage.  SignalHead enforces what a
real signal controller enforces: minimum green, yellow and all-red
intergreens, and optionally a safety shield (maximum green, maximum red).

SignalHead is pure timing logic: the signal state is written through the
`apply(state)` callback, so it can be tested without SUMO.
"""

GREEN, YELLOW, ALL_RED = 'green', 'yellow', 'all_red'


class SignalHead:
    def __init__(self, greens, apply, yellow=3, all_red=2,
                 min_green=10, decision_interval=5, max_green=None, max_red=None, calls=None):
        self.greens = greens
        self.apply = apply
        self.yellow = yellow
        self.all_red = all_red
        self.min_green = min_green
        self.decision_interval = decision_interval
        self.max_green = max_green
        self.max_red = max_red
        self.calls = calls       # calls(link): a vehicle waits at signal link `link` (or None)
        self.servable = {j for g in greens for j, c in enumerate(g) if c in 'Gg'}
        self.reset()

    def reset(self):
        self.stage = 0           # index of the current (or upcoming) green stage
        self.next_stage = 0
        self.mode = GREEN
        self.timer = 0           # seconds spent in the current mode
        self.clock = 0
        self.green_end = [0] * len(self.greens)   # when each stage last lost its green
        self.link_end = [0] * len(self.greens[0])  # ... each signal link (with calls)
        self.forced = 0          # number of shield overrides (max green / max red)
        self._apply(self.greens[0])

    # ------------------------------------------------------------------

    def ready(self):
        """True when the controller may decide (end of min green, then every decision_interval)."""
        if self.mode != GREEN or self.timer < self.min_green:
            return False
        return (self.timer - self.min_green) % self.decision_interval == 0

    def request(self, stage):
        """Hold the current green (same stage) or start switching to another stage.

        The shield overrides the request when a stage has been red longer than
        max_red, or the current green would exceed max_green.  With `calls` the max-red
        rule is per signal link: a link no stage has served for max_red while a vehicle
        waits at it is starved, and the stage serving the most starved links runs.
        """
        chosen = stage
        if self.max_red is not None and self.calls is not None:
            now = self.greens[self.stage]
            starved = [j for j, c in enumerate(now) if c not in 'Gg' and j in self.servable
                       and self.clock - self.link_end[j] >= self.max_red and self.calls(j)]
            if starved:
                stage = max(range(len(self.greens)),
                            key=lambda i: sum(self.greens[i][j] in 'Gg' for j in starved))
        elif self.max_red is not None:
            starved = [i for i in range(len(self.greens))
                       if i != self.stage and self.clock - self.green_end[i] >= self.max_red]
            if starved:
                stage = max(starved, key=lambda i: self.clock - self.green_end[i])
        if self.max_green is not None and stage == self.stage and self.timer >= self.max_green \
                and len(self.greens) > 1:
            stage = (self.stage + 1) % len(self.greens)
        if stage != chosen:
            self.forced += 1
        if stage == self.stage:
            return
        self.green_end[self.stage] = self.clock
        for j, c in enumerate(self.greens[self.stage]):
            if c in 'Gg':
                self.link_end[j] = self.clock
        self.next_stage = stage
        self.mode = YELLOW
        self.timer = 0
        self._apply(self._intergreen('y'))

    def tick(self):
        """Advance one simulation second (call after simulationStep)."""
        self.timer += 1
        self.clock += 1
        if self.mode == YELLOW and self.timer >= self.yellow:
            self.mode, self.timer = ALL_RED, 0
            self._apply(self._intergreen('r'))
            if self.all_red == 0:
                self.tick_to_green()
        elif self.mode == ALL_RED and self.timer >= self.all_red:
            self.tick_to_green()

    def tick_to_green(self):
        self.stage = self.next_stage
        self.mode, self.timer = GREEN, 0
        self._apply(self.greens[self.stage])

    @property
    def green_time(self):
        return self.timer if self.mode == GREEN else 0

    # ------------------------------------------------------------------

    def _intergreen(self, ending):
        """Links green in both stages stay green, the others get `ending` ('y' or 'r')."""
        cur, nxt = self.greens[self.stage], self.greens[self.next_stage]
        return ''.join(
            c if (c in 'Gg' and n in 'Gg') else (ending if c in 'Gg' else 'r')
            for c, n in zip(cur, nxt)
        )

    def _apply(self, state):
        self.state = state
        self.apply(state)
