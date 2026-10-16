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
    assert 'final_manager.pt' in files and 'best_S0.pt' in files
    with open(os.path.join(out, 'train_log.csv')) as f:
        header = f.readline()
    assert 'mgr_actuated' in header and 'wrk_hold' in header


class Q:
    """Stand-in manager network with fixed Q-values."""

    def __init__(self, q):
        self.q = np.array(q, dtype=float)

    def q_values(self, state):
        return self.q.copy()

    def act(self, state, mask, epsilon=0.0):
        return int(np.argmax(self.q))


@pytest.mark.parametrize('q, expected', [([-10.0, -9.8], 0),     # better by 2 % < margin
                                         ([-10.0, -9.0], 1),     # better by 10 %
                                         ([-10.0, -11.0], 0)])
def test_default_expert_margin(corridor, q, expected):
    from tsc.controllers.hierarchy import Manager, make_groups
    m = Manager(corridor, make_groups(corridor, 'network', 400.0), 2, agent=Q(q))
    m.default, m.margin = 0, 0.05
    assert m._act(None, np.ones(2, dtype=bool)) == expected
    m.default = None                                  # no default: plain argmax
    assert m._act(None, np.ones(2, dtype=bool)) == int(np.argmax(q))


@pytest.mark.sumo
def test_delay_reward(corridor, work_root):
    """'delay' is minus the time lost per lane: zero on empty roads, below 'queue' as it
    also counts slow (not only halting) vehicles."""
    from tsc.demand import generate_routes
    from tsc.reward import Reward
    from tsc.simulation import Simulation
    demand = DemandConfig(generator='od', profile='flat', duration=200, tidal_share=0.7)
    route = generate_routes(corridor, demand, os.path.join(work_root, 'delay.rou.xml'), 6, 3000)
    sim = Simulation(corridor, route, 200, os.path.join(work_root, 'smoke', 'delay'), seed=1).start()
    delay, queue = Reward(corridor, 'delay'), Reward(corridor, 'queue')
    assert all(r == 0.0 for r in delay.all(sim).values())
    lower = 0
    while not sim.done:
        sim.step()
        d, q = delay.all(sim), queue.all(sim)
        assert all(d[t] <= 0.0 for t in d)
        lower += sum(d[t] < q[t] for t in d)
    sim.close()
    assert lower > 0


@pytest.mark.sumo
def test_default_expert_fallback(corridor, work_root):
    """default_expert = 'auto': every expert is validated alone before training, the best
    one is the default and the first 'best' checkpoint, which loads as that expert alone."""
    import json
    from tsc import config as config_mod
    from tsc.experiments.runs import load_policy
    from tsc.experiments.train import train_one
    cfg = hierarchy_cfg(interval=60, hidden=[8], updates=2, batch_size=4,
                        default_expert='auto', reward='delay')
    cfg = config_mod.replace(cfg, name='hier_default', output_dir=os.path.join(work_root, 'out'),
                             network={'path': CORRIDOR},
                             demand={'generator': 'od', 'duration': 300, 'tidal_share': 0.7},
                             train={'seeds': [0], 'episodes': 1, 'val_every': 5,
                                    'val_levels': [2500.0], 'level_min': 2500.0,
                                    'level_max': 2500.0})
    out = os.path.join(work_root, 'out', 'hier_default')
    best = os.path.join(out, 'best_hierarchy.json')
    if os.path.exists(best):
        os.remove(best)
    train_one(cfg, out)                    # validates once more at the last episode
    with open(os.path.join(out, 'experts_val.json')) as f:
        experts = json.load(f)
    vals = experts['val_delay']
    assert set(vals) == {'maxpressure', 'actuated'}
    assert experts['default'] == min(vals, key=vals.get)
    with open(best) as f:
        meta = json.load(f)
    assert meta['default'] == experts['default']
    policy, _ = load_policy(out, corridor, 'best')
    m = policy.actions.manager
    assert policy.actions.names[m.default] == experts['default']
    if meta['forced'] is not None:         # the learned manager did not beat the default
        assert meta['forced'] == experts['default'] and policy.actions.force_agree

    # acceptance on fresh scenarios: the deployed policy is recorded and loads as such
    from tsc.experiments.train import accept
    result = accept(out, per_level=2)
    assert result['default'] == experts['default'] and result['scenarios'] == 2
    with open(os.path.join(out, 'accept.json')) as f:
        assert json.load(f) == result
    policy, _ = load_policy(out, corridor, 'best')
    forced = policy.actions.manager.forced
    if result['deployed'] == 'learned':
        assert forced is None and result['p'] < 0.05 and result['change'] < 0
    else:
        assert policy.actions.names[forced] == experts['default']


def test_table_look_up_and_hysteresis(corridor):
    from tsc.controllers.hierarchy import Manager, make_groups
    m = Manager(corridor, make_groups(corridor, 'network', 400.0), 3, policy='table')
    m.table = [((0.1, 0.0), 0, True), ((0.3, 0.1), 1, True), ((0.35, 0.6), 2, False)]
    m.hysteresis = 2
    m.choice, m.candidate, m.streak, m.flows = [0], None, 0, [None]
    m._look_up((0.29, 0.12))
    assert m.choice == [1]                       # first measured decision: switch at once
    m._look_up((0.34, 0.55))
    assert m.choice == [1]                       # past saturation: jam picks expert 2 ...
    m._look_up((0.36, 0.65))
    assert m.choice == [2] and not m.worker_on   # ... after two decisions in a row
    m._look_up((0.09, 0.01))
    m._look_up((0.36, 0.65))
    assert m.choice == [2]                       # a one-off reading does not switch

    # keys per interval: the k-th reading is compared with the levels' k-th readings
    m.table = [([(0.05, 0.0), (0.1, 0.0)], 0, True), ([(0.1, 0.0), (0.3, 0.1)], 1, True)]
    m.choice, m.candidate, m.streak, m.flows = [1], None, 0, [None]
    m._look_up((0.1, 0.0))
    assert m.choice == [1]                       # 1st reading: like level 1 filling up
    m.choice, m.candidate, m.streak, m.flows = [1], None, 0, [None, None, None]
    m._look_up((0.1, 0.0))
    assert m.choice == [0]                       # later (last key reused): like level 0


@pytest.mark.sumo
def test_table_manager(corridor, work_root):
    """manager = 'table': every expert at every level, the best one per level, saved with
    the run and loaded as the policy."""
    import pandas as pd
    from tsc import config as config_mod
    from tsc.experiments.runs import load_policy
    from tsc.experiments.train import train_one
    cfg = hierarchy_cfg(manager='table', interval=60, table_levels=[1000.0, 3500.0],
                        table_scenarios=1)
    cfg = config_mod.replace(cfg, name='hier_table', output_dir=os.path.join(work_root, 'out'),
                             workers=1, network={'path': CORRIDOR},
                             demand={'generator': 'od', 'duration': 300, 'tidal_share': 0.7},
                             train={'seeds': [0]})
    out = os.path.join(work_root, 'out', 'hier_table')
    train_one(cfg, out)
    df = pd.read_csv(os.path.join(out, 'table.csv'))
    assert len(df) == 2 * 2 and set(df.expert) == {'maxpressure', 'actuated'}
    best = df.loc[df.groupby('level').delay.idxmin()].set_index('level').expert
    policy, _ = load_policy(out, corridor, 'best')
    table = policy.actions.manager.table
    assert [policy.actions.names[e] for _, e, _ in table] == list(best)
    assert table[0][0][:, 0].mean() < table[1][0][:, 0].mean()   # more demand, more flow
    assert len(table[0][0]) == 300 // 60         # one key per interval


@pytest.mark.sumo
def test_expert_workers(corridor, work_root):
    """Expert-specific workers: a worker run (fixed expert + residual worker) starts from its
    expert alone and is accepted or not on fresh scenarios; a table run with
    worker = 'expert' then uses it whenever that expert is active."""
    import json
    from tsc import config as config_mod
    from tsc.controllers.hierarchy import AGREE, ExpertWorker
    from tsc.experiments.runs import load_policy
    from tsc.experiments.train import accept, train_one
    small = {'network': {'path': CORRIDOR}, 'workers': 1,
             'demand': {'generator': 'od', 'duration': 300, 'tidal_share': 0.7},
             'agent': {'hidden': [16], 'shared': True, 'learn_start': 50, 'batch_size': 16}}
    cfg = hierarchy_cfg(manager='fixed', fixed_expert=1, worker='residual')
    cfg.reward.kind = 'delay'
    cfg = config_mod.replace(cfg, name='wk_actuated', output_dir=os.path.join(work_root, 'out'),
                             train={'seeds': [0], 'episodes': 1, 'val_every': 5,
                                    'val_levels': [2500.0], 'level_min': 2500.0,
                                    'level_max': 2500.0}, **small)
    wk = os.path.join(work_root, 'out', 'wk_actuated', 's0')
    train_one(cfg, wk)
    result = accept(wk, per_level=2)
    assert result['default'] == 'actuated'
    policy, _ = load_policy(wk, corridor, 'best')
    assert policy.actions.force_agree == (result['deployed'] == 'actuated')

    # pretend the worker passed, so the table run uses it
    meta_path = os.path.join(wk, 'best_hierarchy.json')
    with open(meta_path) as f:
        meta = json.load(f)
    meta.update(agree=False, forced=None)
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    run = os.path.join(work_root, 'out', 'wk_table', 's0')
    tcfg = hierarchy_cfg(manager='table', worker='expert', interval=60,
                         table_levels=[1000.0, 3500.0], table_scenarios=1,
                         worker_runs={'actuated': wk.replace('s0', 's{seed}')})
    tcfg = config_mod.replace(tcfg, name='wk_table', output_dir=os.path.join(work_root, 'out'),
                              train={'seeds': [0]}, **small)
    train_one(tcfg, run)
    policy, _ = load_policy(run, corridor, 'best')
    actions = policy.actions
    assert set(actions.expert_workers) == {1}                   # actuated has a worker
    assert all(isinstance(a, ExpertWorker) for a in policy.agents.values())
    assert policy.learners() == {}                               # nothing to save
    tl = corridor.tl_ids[0]
    worker = actions.expert_workers[1][tl]
    state = np.zeros(worker.n_states, dtype=np.float32)
    mask = np.ones(3, dtype=bool)
    actions.manager.choice = [0] * len(actions.manager.groups)  # max-pressure: no worker
    assert policy.agents[tl].act(state, mask) == AGREE
    actions.manager.choice = [1] * len(actions.manager.groups)  # actuated: its worker acts
    assert policy.agents[tl].act(state, mask) == worker.act(state, mask)
    actions.manager.worker_on = False                           # unless the table says so
    assert policy.agents[tl].act(state, mask) == AGREE

    # table_alone: actuated is measured with and without its worker, the table picks
    import pandas as pd
    alone = config_mod.replace(tcfg, name='wk_table_alone', hierarchy={'table_alone': True})
    train_one(alone, run + '_alone')
    df = pd.read_csv(os.path.join(run + '_alone', 'table.csv'))
    assert set(df.expert) == {'maxpressure', 'actuated', 'actuated+worker'}
    best = df.loc[df.groupby('level').delay.idxmin()].set_index('level').expert
    policy, _ = load_policy(run + '_alone', corridor, 'best')
    names = policy.actions.names
    assert [names[e] + ('+worker' if w and names[e] == 'actuated' else '')
            for _, e, w in policy.actions.manager.table] == list(best)
    with pytest.raises(ValueError, match='expert'):
        train_one(config_mod.replace(tcfg, hierarchy={'manager': 'learned'}), run + '_x')


def test_advanced_maxpressure_rule(corridor):
    """Advanced-MP keeps the green while weight x vehicles still moving toward the stop line
    (within range) >= the highest efficient pressure, else takes the highest-pressure stage."""
    from types import SimpleNamespace
    from tsc.controllers.maxpressure import AdvancedMaxPressure
    from tsc.network import lane_edge
    tl = corridor.tl_ids[0]
    links = corridor.links[tl]
    served = [{l for i, l, _ in links if g[i] in 'Gg'} for g in corridor.greens[tl]]
    other = next(s for s in range(1, len(served)) if served[s] - served[0])
    queued = sorted(served[other] - served[0])[0]          # a lane only stage `other` serves
    length = corridor.lane_length[queued]
    moving = sorted(served[0])[0]

    def api(halting, near):
        lane = SimpleNamespace(
            getLastStepHaltingNumber=lambda l: halting.get(l, 0),
            getLastStepVehicleIDs=lambda l: [f'v{k}' for k in range(near)] if l == moving else [])
        edge = SimpleNamespace(getLaneNumber=lambda e: 1, getLastStepHaltingNumber=lambda e: 0)
        pos = corridor.lane_length[moving] - 30.0                  # 30 m before the stop line
        vehicle = SimpleNamespace(getSpeed=lambda v: 5.0, getLanePosition=lambda v: pos)
        return SimpleNamespace(lane=lane, edge=edge, vehicle=vehicle)

    sim = SimpleNamespace(network=corridor, heads={tl: SimpleNamespace(stage=0)})
    amp = AdvancedMaxPressure(weight=1.0, effective_range=100.0)
    amp.reset(sim)
    sim.api = api({queued: 6}, near=10)       # 10 moving on stage 0 >= pressure 6: keep
    assert amp.act(sim, tl) == 0
    sim.api = api({queued: 12}, near=10)      # pressure 12 > 10: switch
    assert amp.act(sim, tl) != 0
    amp.effective_range = 20.0                # the moving vehicles are out of range: switch
    sim.api = api({queued: 6}, near=10)
    assert amp.act(sim, tl) != 0
    assert length > 0 and lane_edge(queued)


@pytest.mark.sumo
def test_approach_detectors_run(corridor, work_root):
    """SUMO accepts the approach detectors (multi-lane spans included) and they see traffic."""
    from tsc.demand import generate_routes
    from tsc.detectors import e2a_id
    from tsc.observation import Observer
    from tsc.simulation import Simulation
    demand = DemandConfig(generator='od', profile='flat', duration=200, tidal_share=0.7)
    route = generate_routes(corridor, demand, os.path.join(work_root, 'appr.rou.xml'), 7, 3000)
    sim = Simulation(corridor, route, 200, os.path.join(work_root, 'smoke', 'appr'), seed=1,
                     approach_length=150.0).start()
    obs = Observer(corridor, mode='detector', comm=False, approach_length=150.0)
    seen = 0
    while not sim.done:
        sim.step()
        seen += sum(sim.api.lanearea.getLastStepVehicleNumber(e2a_id(l))
                    for t in corridor.tl_ids for l in corridor.in_lanes[t])
    assert seen > 0 and obs.observe(sim, corridor.tl_ids[0]).shape == (obs.dim,)
    sim.close()
