"""The arterial corridor: approach chains, O/D demand, green wave (step 2E.1)."""

import os

import pytest

from tsc.config import DemandConfig
from tsc.demand import generate_routes, od_matrix, scenario_flip

CORRIDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'networks', 'corridor', 'corridor.net.xml')


@pytest.fixture(scope='module')
def corridor(work_root):
    from tests.conftest import make_network
    return make_network(CORRIDOR, work_root)


def test_layout(corridor):
    assert corridor.tl_ids == [f'S{i}' for i in range(7)]
    # arterial through, protected left from the bay, cross street
    assert all(len(g) == 3 for g in corridor.greens.values())
    assert corridor.neighbours['S0'] == ['S1'] and corridor.neighbours['S3'] == ['S2', 'S4']


def test_approach_chains(corridor, test5):
    # every arterial approach to a signal ends in a bay with the road before it upstream
    bays = [e for e in corridor.upstream_edges if e.endswith('_bay')]
    assert len(bays) == 14 and all(corridor.upstream_edges[e] for e in bays)
    assert corridor.upstream_edges['S3_S4_bay'] == ['S3_S4']
    assert corridor.downstream_edges['S3_S4'] == ['S3_S4_bay']
    assert corridor.upstream_edges['R0_S2_bay'] == ['R0_S2']    # stops at the priority junction
    assert not any(test5.approach_edges.values())       # no lane splits in the test network


def test_od_shares(corridor):
    demand = DemandConfig(generator='od', major_share=0.6, tidal_share=0.7, tidal_random=False)
    pts, pairs, probs = od_matrix(corridor, demand)
    name = [p[0] for p in pts]
    share = {(name[o], name[d]): p for (o, d), p in zip(pairs, probs)}
    assert abs(sum(probs) - 1) < 1e-9
    assert abs(share[('W', 'E')] - 0.42) < 1e-9 and abs(share[('E', 'W')] - 0.18) < 1e-9
    pts, pairs, probs = od_matrix(corridor, demand, flip=True)
    share = {(name[o], name[d]): p for (o, d), p in zip(pairs, probs)}
    assert abs(share[('E', 'W')] - 0.42) < 1e-9


def test_od_routes_and_direction(corridor, work_root):
    demand = DemandConfig(generator='od', profile='flat', duration=600, tidal_share=0.7)
    path = os.path.join(work_root, 'od.rou.xml')
    generate_routes(corridor, demand, path, seed=5, rate_vph=1800)
    with open(path) as f:
        n = f.read().count('<vehicle ')
    assert 200 < n < 400                                  # Poisson around 300
    flips = {scenario_flip(demand, s) for s in range(20)}
    assert flips == {False, True}


@pytest.mark.sumo
def test_green_wave(corridor, work_root):
    from tsc.controllers.webster import green_wave_programs
    from tsc.simulation import run_episode
    demand = DemandConfig(generator='od', profile='flat', duration=120, tidal_share=0.7,
                          tidal_random=False)
    route = generate_routes(corridor, demand, os.path.join(work_root, 'gw.rou.xml'), 1, 2000)
    plan = green_wave_programs(corridor, [route], os.path.join(work_root, 'gw.add.xml'))
    with open(plan) as f:
        text = f.read()
    assert text.count('<tlLogic') == 7 and 'offset="0"' not in text.replace('offset="0.00"', '')
    m = run_episode(corridor, None, route, 120, os.path.join(work_root, 'smoke', 'gw'),
                    seed=1, tls='webster', additional=[plan])
    assert m['inserted'] > 0


def test_directional_messages(corridor):
    """Directional messages depend on the receiver and on what the sender is serving."""
    from tests.test_observation import fake_sim
    from tsc.observation import DIR_MSG_SIZE, Observer
    obs = Observer(corridor, mode='detector', comm=True, message='directional')
    assert obs.dim == obs.local_dim + corridor.max_neighbours * DIR_MSG_SIZE
    sim = fake_sim(corridor, halting=2)
    for head in sim.heads.values():
        head.mode = 'green'
    s = obs.observe(sim, 'S3')
    assert s.shape == (obs.dim,)
    # arterial stage (0) sends through traffic to S4; the protected-left stage (1) sends none
    sim.heads['S3'].stage = 0
    assert obs.directional_message(sim, 'S3', 'S4')[2] == 0.5     # 2 of 4 links toward S4
    sim.heads['S3'].stage = 1
    assert obs.directional_message(sim, 'S3', 'S4')[2] == 0.0
    sim.heads['S3'].mode = 'yellow'
    assert obs.directional_message(sim, 'S3', 'S4')[2] == 0.0
    # messages to different receivers differ (S3 → S2 vs S3 → S4 use different lanes)
    sim.heads['S3'].mode = 'green'
    sim.heads['S3'].stage = 0
    assert obs.toward_lanes[('S3', 'S2')] != obs.toward_lanes[('S3', 'S4')]


@pytest.mark.sumo
def test_directional_policy_runs(corridor, work_root):
    from tsc.config import Config
    from tsc.controllers.rl import RLController, make_agents
    from tsc.experiments.runs import make_observer
    from tsc.simulation import run_episode
    cfg = Config()
    cfg.observation.mode = 'detector'
    cfg.observation.message = 'directional'
    cfg.agent.hidden = [16, 16]
    demand = DemandConfig(generator='od', profile='flat', duration=90, tidal_share=0.7)
    route = generate_routes(corridor, demand, os.path.join(work_root, 'dir.rou.xml'), 2, 2500)
    observer = make_observer(corridor, cfg, seed=0)
    controller = RLController(make_agents(corridor, observer, cfg.agent), observer, epsilon=0.2)
    m = run_episode(corridor, controller, route, 90, os.path.join(work_root, 'smoke', 'dir'),
                    seed=1, max_green=60, max_red=90)
    assert m['inserted'] > 0
    assert any(v > 0 for v in observer.released.values())
