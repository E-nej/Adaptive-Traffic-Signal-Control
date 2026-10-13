from tsc.signals import ALL_RED, GREEN, YELLOW, SignalHead

# link 0: green in both stages; links 1-2 only in stage 0; link 3 only in stage 1
GREENS = ['GGgr', 'Grrg']


def make(**kw):
    states = []
    head = SignalHead(GREENS, states.append, **kw)
    return head, states


def run(head, seconds):
    for _ in range(seconds):
        head.tick()


def test_starts_in_first_green():
    head, states = make()
    assert head.mode == GREEN and head.stage == 0 and states == ['GGgr']


def test_decisions_after_min_green_then_every_interval():
    head, _ = make(min_green=10, decision_interval=5)
    ready = []
    for t in range(1, 31):
        head.tick()
        if head.ready():
            ready.append(t)
    assert ready == [10, 15, 20, 25, 30]


def test_switch_yellow_then_all_red_then_green():
    head, states = make(yellow=3, all_red=2)
    run(head, 10)
    head.request(1)
    assert head.mode == YELLOW
    # link 0 is green in both stages and stays green; the others turn yellow / stay red
    assert states[-1] == 'Gyyr'
    run(head, 3)
    assert head.mode == ALL_RED and states[-1] == 'Grrr'
    run(head, 2)
    assert head.mode == GREEN and head.stage == 1 and states[-1] == 'Grrg'
    assert not head.ready()                       # min green starts again


def test_no_all_red():
    head, states = make(yellow=3, all_red=0)
    run(head, 10)
    head.request(1)
    run(head, 3)
    assert head.mode == GREEN and head.stage == 1 and states[-1] == 'Grrg'


def test_holding_the_current_stage_changes_nothing():
    head, states = make()
    run(head, 10)
    head.request(0)
    assert head.mode == GREEN and head.timer == 10 and len(states) == 1


def test_max_green_forces_a_switch():
    head, _ = make(max_green=20)
    run(head, 20)
    head.request(0)
    assert head.mode == YELLOW and head.next_stage == 1 and head.forced == 1


def test_max_red_serves_the_longest_starved_stage():
    greens = ['Grr', 'rGr', 'rrG']
    head = SignalHead(greens, lambda s: None, yellow=3, all_red=2, max_red=30)
    run(head, 30)
    head.request(0)          # the agent wants to keep stage 0, but stages 1 and 2 waited 30 s
    assert head.mode == YELLOW and head.next_stage in (1, 2) and head.forced == 1


def test_max_red_with_calls_serves_waiting_links_only():
    greens = ['GGrr', 'rGGr', 'rrrG']     # link 1 is green in stages 0 and 1
    head = SignalHead(greens, lambda s: None, yellow=3, all_red=2, max_red=30,
                      calls=lambda link: link in (1, 3))
    run(head, 30)
    head.request(0)          # link 1 is served by stage 0 itself; link 3 waits for stage 2
    assert head.mode == YELLOW and head.next_stage == 2 and head.forced == 1
    quiet = SignalHead(greens, lambda s: None, max_red=30, calls=lambda link: False)
    run(quiet, 30)
    quiet.request(0)         # no calls: the agent keeps its green
    assert quiet.mode == GREEN and quiet.forced == 0


def test_green_time():
    head, _ = make()
    run(head, 12)
    assert head.green_time == 12
    head.request(1)
    assert head.green_time == 0
