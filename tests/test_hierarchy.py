"""Hierarchical control (step 2H): signal groups, the manager, the actuated expert."""

import os

import numpy as np
import pytest

from tsc.config import Config, DemandConfig

CORRIDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'networks', 'corridor', 'corridor.net.xml')


@pytest.fixture(scope='module')
def corridor(work_root):
    from tests.conftest import make_network
    return make_network(CORRIDOR, work_root)


def test_groups(corridor, test5):
    # signals 220-380 m apart: one coordinated group at 400 m, split up at 250 m
    assert corridor.groups(400.0) == [corridor.tl_ids]
    small = corridor.groups(250.0)
    assert len(small) > 1
    assert sorted(t for g in small for t in g) == corridor.tl_ids     # a partition
    assert test5.groups(400.0) == [test5.tl_ids]
    # strict: no group across a single road longer than the range
    assert corridor.groups(10.0) == [[t] for t in corridor.tl_ids]


def test_make_groups(corridor):
    from tsc.controllers.hierarchy import make_groups
    assert make_groups(corridor, 'network', 400.0) == [corridor.tl_ids]
    assert make_groups(corridor, 'junction', 400.0) == [[t] for t in corridor.tl_ids]
    with pytest.raises(ValueError):
        make_groups(corridor, 'district', 400.0)


def test_gap_actuated_loops(corridor):
    from tsc.controllers.select import GapActuated
    stop = GapActuated(corridor, loops='stopline')
    sumo = GapActuated(corridor, loops='sumo')
    for tl_id, stages in sumo.served.items():
        full = stop.served[tl_id]
        for k, lanes in enumerate(stages):
            # a lane that stays green in the next stage does not extend this one
            assert set(lanes) == set(full[k]) - set(full[(k + 1) % len(full)])
    with pytest.raises(ValueError):
        GapActuated(corridor, loops='radar')


def hierarchy_cfg(**hierarchy):
    cfg = Config()
    cfg.signals.control = 'hierarchy'
    cfg.signals.experts = ['maxpressure', 'actuated']     # no plan needed
    cfg.signals.actuated_loops = 'sumo'
    cfg.observation.mode = 'detector'
    for k, v in hierarchy.items():
        setattr(cfg.hierarchy, k, v)
    return cfg


@pytest.mark.sumo
@pytest.mark.parametrize('scope', ['network', 'group'])
def test_manager_learns_and_switches(corridor, work_root, scope):
    from tsc.experiments.runs import make_controller, make_observer
    from tsc.demand import generate_routes
    from tsc.simulation import run_episode
    cfg = hierarchy_cfg(scope=scope, group_range=250.0, interval=20, hidden=[8], updates=2,
                        batch_size=4)
    demand = DemandConfig(generator='od', profile='flat', duration=200, tidal_share=0.7)
    route = generate_routes(corridor, demand, os.path.join(work_root, 'hier.rou.xml'), 3, 2500)
    controller = make_controller(corridor, make_observer(corridor, cfg, 0), cfg)
    assert controller.agents == {} and set(controller.learners()) == {'manager'}
    actions = controller.actions
    actions.set_training(True, 1.0)                       # random experts, learning
    m = run_episode(corridor, controller, route, 200, os.path.join(work_root, 'smoke', scope),
                    seed=1, decision_interval=1)
    assert m['inserted'] > 0
    manager = actions.manager
    n_groups = len(manager.groups)
    assert len(manager.agent.buffer) == n_groups * (200 // 20)   # decisions at t = 0, 20 .. 200
    shares = manager.shares()
    assert np.isclose(shares.sum(), 1.0) and (shares > 0).all()     # both experts ran
    assert actions.episode_stats().keys() == {'mgr_maxpressure', 'mgr_actuated'}


@pytest.mark.sumo
def test_fixed_manager_is_the_expert(corridor, work_root):
    """A fixed manager runs one expert everywhere: same result as that expert on its own."""
    from tsc.controllers.select import FixedSelection
    from tsc.experiments.runs import make_controller, make_observer
    from tsc.demand import generate_routes
    from tsc.simulation import run_episode
    cfg = hierarchy_cfg(manager='fixed', fixed_expert=1, interval=30)
    cfg.observation.noise = 0.0
    demand = DemandConfig(generator='od', profile='flat', duration=150, tidal_share=0.7)
    route = generate_routes(corridor, demand, os.path.join(work_root, 'fixed.rou.xml'), 4, 2500)
    hier = make_controller(corridor, make_observer(corridor, cfg, 0), cfg)
    alone = FixedSelection(corridor, default=1, experts=cfg.signals.experts,
                           actuated_loops='sumo')
    out = {}
    for name, ctrl in (('hier', hier), ('alone', alone)):
        out[name] = run_episode(corridor, ctrl, route, 150,
                                os.path.join(work_root, 'smoke', f'fixed_{name}'), seed=1,
                                decision_interval=1)
    assert out['hier']['delay'] == pytest.approx(out['alone']['delay'])
    assert hier.actions.manager.shares()[1] == 1.0


class Forced:
    """Drives HierarchyActions with one fixed worker action (no learning)."""

    needs_plan = True

    def __init__(self, actions, action):
        self.actions, self.action = actions, action

    def set_plans(self, plans):
        self.actions.set_plans(plans)

    def reset(self, sim):
        self.actions.reset(sim)

    def tick(self, sim):
        self.actions.tick(sim)

    def act(self, sim, tl_id):
        return self.actions.stage(sim, tl_id, self.action)


@pytest.fixture(scope='module')
def corridor_plans(corridor, work_root):
    from tsc.controllers.webster import green_wave_programs
    from tsc.demand import generate_routes
    from tsc.plans import load_plans
    demand = DemandConfig(generator='od', profile='flat', duration=300, tidal_share=0.7,
                          tidal_random=False)
    route = generate_routes(corridor, demand, os.path.join(work_root, 'res.rou.xml'), 5, 3000)
    plan = green_wave_programs(corridor, [route], os.path.join(work_root, 'res_gw.add.xml'))
    return route, load_plans(plan, corridor)


@pytest.mark.sumo
@pytest.mark.parametrize('expert', [0, 1, 2, 3])
def test_agreeing_worker_is_the_expert(corridor, corridor_plans, work_root, expert):
    """A worker that always agrees changes nothing: same result as the expert alone."""
    from tsc.controllers.hierarchy import AGREE, HierarchyActions
    from tsc.simulation import run_episode
    route, plans = corridor_plans
    out = {}
    for worker in ('off', 'residual'):
        cfg = hierarchy_cfg(manager='fixed', fixed_expert=expert, worker=worker)
        cfg.signals.experts = ['greenwave', 'coord_actuated', 'maxpressure', 'actuated']
        cfg.observation.noise = 0.0
        ctrl = Forced(HierarchyActions(corridor, cfg), AGREE)
        ctrl.set_plans(plans)
        out[worker] = run_episode(corridor, ctrl, route, 300,
                                  os.path.join(work_root, 'smoke', f'agree_{expert}_{worker}'),
                                  seed=1, decision_interval=1)
    assert out['residual']['delay'] == pytest.approx(out['off']['delay'])


@pytest.mark.sumo
@pytest.mark.parametrize('action', [1, 2])
def test_overriding_worker_changes_the_timing(corridor, corridor_plans, work_root, action):
    from tsc.controllers.hierarchy import AGREE, HierarchyActions
    from tsc.simulation import run_episode
    route, plans = corridor_plans
    out = {}
    for a in (AGREE, action):
        cfg = hierarchy_cfg(manager='fixed', fixed_expert=1, worker='residual')
        actions = HierarchyActions(corridor, cfg)
        out[a] = run_episode(corridor, Forced(actions, a), route, 300,
                             os.path.join(work_root, 'smoke', f'override_{action}_{a}'),
                             seed=1, decision_interval=1)
        stats = actions.episode_stats()
    assert out[action]['delay'] != pytest.approx(out[AGREE]['delay'])
    assert stats['wrk_hold' if action == 1 else 'wrk_switch'] > 0


@pytest.mark.sumo
def test_both_levels_learn(corridor, work_root):
    """One training episode with a learning manager and residual workers."""
    from tsc import config as config_mod
    from tsc.experiments.train import train_one
    cfg = hierarchy_cfg(interval=60, hidden=[8], updates=2, batch_size=4, worker='residual')
    cfg = config_mod.replace(cfg, name='hier_smoke', output_dir=os.path.join(work_root, 'out'),
                             network={'path': CORRIDOR},
                             demand={'generator': 'od', 'duration': 300, 'tidal_share': 0.7},
                             agent={'hidden': [16], 'shared': True, 'learn_start': 50,
                                    'batch_size': 16},
                             train={'seeds': [0], 'episodes': 1, 'val_every': 1,
                                    'val_levels': [2500.0], 'level_min': 2500.0,
                                    'level_max': 2500.0})
    out = os.path.join(work_root, 'out', 'hier_smoke')
    train_one(cfg, out)
    files = os.listdir(out)
    assert 'final_manager.weights.h5' in files and 'best_S0.weights.h5' in files
    with open(os.path.join(out, 'train_log.csv')) as f:
        header = f.readline()
    assert 'mgr_actuated' in header and 'wrk_hold' in header
