"""Plan-based control (step 2F)."""

import os

import pytest

from tsc.controllers.plan import EARLY, FOLLOW, HOLD, PlanActions, PlanController
from tsc.plans import StagePlan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORRIDOR = os.path.join(ROOT, 'networks', 'corridor', 'corridor.net.xml')


def toy_plan():
    # cycle 100: stage 0 green 0-50, intergreen, stage 1 green 55-75, stage 2 green 80-95
    return StagePlan(100, offset=10, stages=[(0, 0, 50), (1, 55, 75), (2, 80, 95)])


def test_plan_windows():
    p = toy_plan()
    # windows start where the previous stage's green ends: stage 0's window is [95, 150)
    # in cycle time, i.e. absolute [105, 160) for offset 10
    assert p.stage(p.counter(105)) == 0 and p.stage(p.counter(159)) == 0
    assert p.stage(p.counter(160)) == 1 and p.stage(p.counter(185)) == 2
    n = p.counter(120)
    assert p.window_end(n) == 160 and p.counter(160) == n + 1
    assert p.split(n) == 50 and p.split(n + 1) == 20
    assert p.counter(105 + 100) == p.counter(105) + 3       # one cycle = K windows


class FakeHead:
    def __init__(self, stage=0, timer=10, min_green=10):
        self.stage, self.timer, self.min_green = stage, timer, min_green


class FakeSim:
    def __init__(self, t, head):
        self.t, self.heads = t, {'J': head}


class FakeNet:
    greens = {'J': ['G', 'g', 'r']}


def test_plan_actions():
    acts = PlanActions(FakeNet(), interval=5, max_extension=20)
    acts.set_plans({'J': toy_plan()})
    acts.reset(FakeSim(105, FakeHead()))                     # in stage 0's window
    assert acts.stage(FakeSim(150, FakeHead()), 'J', FOLLOW) == 0
    # the plan switches to stage 1 at 160: FOLLOW switches, HOLD keeps stage 0 ...
    assert acts.stage(FakeSim(161, FakeHead()), 'J', HOLD) == 0
    assert acts.mask(FakeSim(161, FakeHead()), 'J')[HOLD]
    # ... but not past the extension budget
    assert not acts.mask(FakeSim(181, FakeHead()), 'J')[HOLD]
    assert acts.stage(FakeSim(181, FakeHead()), 'J', HOLD) == 1
    # EARLY ends stage 1 before its planned end (180 + 20) and keeps stage 2 until the plan
    # catches up, without switching back
    assert acts.stage(FakeSim(182, FakeHead(stage=1)), 'J', EARLY) == 2
    assert not acts.mask(FakeSim(183, FakeHead(stage=2)), 'J')[EARLY]   # already ahead
    assert acts.stage(FakeSim(190, FakeHead(stage=2)), 'J', FOLLOW) == 2
    # falling two windows behind rejoins the plan
    acts.count['J'] -= 2
    assert acts.stage(FakeSim(300, FakeHead()), 'J', FOLLOW) == toy_plan().stage(
        toy_plan().counter(300))


@pytest.fixture(scope='module')
def corridor(work_root):
    from tests.conftest import make_network
    return make_network(CORRIDOR, work_root)


def test_plan_replay_matches_sumo(corridor, work_root):
    """Python replay of a green wave keeps SUMO's timing (cycle starts at t = offset)."""
    from tsc.config import DemandConfig
    from tsc.controllers.webster import green_wave_programs
    from tsc.demand import generate_routes
    from tsc.plans import load_plans
    from tsc.simulation import run_episode
    demand = DemandConfig(generator='od', major_share=0.6, tidal_share=0.7, tidal_random=False,
                          duration=3600)
    route = generate_routes(corridor, demand, os.path.join(work_root, 'plan.rou.xml'), 3, 1200)
    plan = green_wave_programs(corridor, [route], os.path.join(work_root, 'gw.add.xml'))
    plans = load_plans(plan, corridor)
    assert {p.stage(p.counter(p.origin)) for p in plans.values()} == {0}
    sumo = run_episode(corridor, None, route, 600, os.path.join(work_root, 'gw_sumo'),
                       seed=3, tls='webster', additional=[plan])
    ctrl = PlanController(corridor)
    ctrl.set_plans(plans)
    py = run_episode(corridor, ctrl, route, 600, os.path.join(work_root, 'gw_py'), seed=3,
                     decision_interval=1)
    assert py['delay'] == pytest.approx(sumo['delay'], rel=0.15)


def test_low_demand_plan_has_no_negative_green(corridor, work_root):
    """Too little demand made tlsCycleAdaptation give a stage a negative green."""
    import xml.etree.ElementTree as ET
    from tsc.config import DemandConfig
    from tsc.controllers.webster import green_wave_programs, required_cycle
    from tsc.demand import generate_routes
    demand = DemandConfig(generator='od', major_share=0.6, tidal_share=0.7, tidal_random=False,
                          duration=600)
    route = generate_routes(corridor, demand, os.path.join(work_root, 'low.rou.xml'), 3, 2500)
    plan = green_wave_programs(corridor, [route], os.path.join(work_root, 'low.add.xml'))
    phases = list(ET.parse(plan).getroot().iter('phase'))
    assert min(float(p.get('duration')) for p in phases) > 0
    assert required_cycle(corridor) == 43


def test_fixed_selection_matches_experts(corridor, work_root):
    """Selecting the green wave everywhere is the plan replay itself; switching experts
    mid-run is safe (the plan expert rejoins the plan)."""
    from tsc.config import DemandConfig
    from tsc.controllers.select import FixedSelection
    from tsc.controllers.webster import green_wave_programs
    from tsc.demand import generate_routes
    from tsc.plans import load_plans
    from tsc.simulation import run_episode
    demand = DemandConfig(generator='od', major_share=0.6, tidal_share=0.7, tidal_random=False,
                          duration=900)
    route = generate_routes(corridor, demand, os.path.join(work_root, 'sel.rou.xml'), 5, 3000)
    plans = load_plans(green_wave_programs(corridor, [route], os.path.join(work_root, 'sel.add.xml')),
                       corridor)
    ref = PlanController(corridor)
    ref.set_plans(plans)
    a = run_episode(corridor, ref, route, 900, os.path.join(work_root, 'sel_ref'), seed=5,
                    decision_interval=1)
    sel = FixedSelection(corridor, default=0)
    sel.set_plans(plans)
    b = run_episode(corridor, sel, route, 900, os.path.join(work_root, 'sel_gw'), seed=5,
                    decision_interval=1)
    assert a['delay'] == pytest.approx(b['delay'])

    class Rotating(FixedSelection):      # changes every junction's expert every 120 s
        def act(self, sim, tl_id):
            return self.actions.stage(sim, tl_id, (sim.t // 120) % 3)

    rot = Rotating(corridor)
    rot.set_plans(plans)
    m = run_episode(corridor, rot, route, 900, os.path.join(work_root, 'sel_rot'), seed=5,
                    decision_interval=1, max_red=150)
    assert m['throughput'] > 0.7 and m['teleports'] == 0
