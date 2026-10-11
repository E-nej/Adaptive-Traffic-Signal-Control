"""Short SUMO runs (60 s of simulation) of every controller type."""

import os

import pytest

from tsc.config import Config, DemandConfig
from tsc.controllers import baseline
from tsc.controllers.webster import webster_programs
from tsc.demand import generate_routes
from tsc.simulation import run_episode

pytestmark = pytest.mark.sumo

DURATION = 60
DEMAND = DemandConfig(profile='flat', duration=DURATION, min_distance=50.0)


@pytest.fixture(scope='module')
def routes(test5, single_4l, work_root):
    out = {}
    for name, network in (('test5', test5), ('single_4l', single_4l)):
        path = os.path.join(work_root, f'smoke_{name}.rou.xml')
        out[name] = generate_routes(network, DEMAND, path, seed=1, rate_vph=3600)
    return out


def check(m):
    assert m['loaded'] > 0 and m['inserted'] > 0
    assert m['delay'] >= 0 and 0 <= m['throughput'] <= 1


@pytest.mark.parametrize('name', ['fixed', 'actuated', 'maxpressure', 'maxpressure_det'])
@pytest.mark.parametrize('net_name', ['test5', 'single_4l'])
def test_baseline(request, routes, work_root, net_name, name):
    network = request.getfixturevalue(net_name)
    tls, controller = baseline(name)
    out = os.path.join(work_root, 'smoke', f'{net_name}_{name}')
    check(run_episode(network, controller, routes[net_name], DURATION, out, seed=1, tls=tls))


def test_webster(test5, routes, work_root):
    plan = webster_programs(test5, [routes['test5']], os.path.join(work_root, 'webster.add.xml'))
    out = os.path.join(work_root, 'smoke', 'webster')
    check(run_episode(test5, None, routes['test5'], DURATION, out, seed=1, tls='webster',
                      additional=[plan]))


def test_shield_counts_overrides(test5, routes, work_root):
    tls, controller = baseline('maxpressure')
    out = os.path.join(work_root, 'smoke', 'shield')
    m = run_episode(test5, controller, routes['test5'], DURATION, out, seed=1, tls=tls,
                    max_green=15, max_red=20)
    check(m)
    assert m['max_green_overrides'] > 0


@pytest.mark.parametrize('mode', ['grid', 'detector'])
def test_rl_policy(test5, routes, work_root, mode):
    from tsc.controllers.rl import RLController, make_agents
    from tsc.experiments.runs import make_observer
    cfg = Config()
    cfg.observation.mode = mode
    cfg.observation.noise = 0.05
    cfg.agent.hidden = [16, 16]            # small and fast; the policy is untrained anyway
    observer = make_observer(test5, cfg, seed=0)
    controller = RLController(make_agents(test5, observer, cfg.agent), observer, epsilon=0.1)
    out = os.path.join(work_root, 'smoke', f'rl_{mode}')
    check(run_episode(test5, controller, routes['test5'], DURATION, out, seed=1))


def test_detector_maxpressure_noise(test5, routes, work_root):
    """Noisy detector max-pressure runs and is reproducible for the same seed."""
    from tsc.controllers import DetectorMaxPressure
    results = []
    for i in range(2):
        controller = DetectorMaxPressure(noise=0.05, dropout=0.02, seed=7)
        out = os.path.join(work_root, 'smoke', f'mp_det_noisy_{i}')
        results.append(run_episode(test5, controller, routes['test5'], DURATION, out, seed=1))
    check(results[0])
    assert results[0]['delay'] == results[1]['delay']
