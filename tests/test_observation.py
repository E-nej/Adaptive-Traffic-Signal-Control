"""Observation shapes and contents, with a fake simulation (no SUMO run needed)."""

from types import SimpleNamespace

import numpy as np
import pytest

from tsc.observation import DET_FEATS, MSG_SIZE, N_CELLS, Observer
from tsc.reward import Reward
from tsc.simulation import approach_halting


def fake_sim(network, vehicles=None, halting=0, stage=0, green_time=0):
    """vehicles: {lane: [position from the lane start, ...]}"""
    vehicles = vehicles or {}
    ids = {lane: [f'{lane}#{i}' for i in range(len(p))] for lane, p in vehicles.items()}
    pos = {f'{lane}#{i}': x for lane, p in vehicles.items() for i, x in enumerate(p)}
    api = SimpleNamespace(
        lane=SimpleNamespace(
            getLastStepVehicleIDs=lambda l: ids.get(l, []),
            getLastStepHaltingNumber=lambda l: halting,
            getLastStepVehicleNumber=lambda l: len(ids.get(l, [])),
            getLastStepMeanSpeed=lambda l: network.lane_speed.get(l, 10.0),
            getWaitingTime=lambda l: 0.0,
            getCO2Emission=lambda l: 0.0,
        ),
        vehicle=SimpleNamespace(getLanePosition=lambda v: pos[v]),
        lanearea=SimpleNamespace(
            getJamLengthVehicle=lambda d: halting,
            getLastStepOccupancy=lambda d: 50.0,
            getLastStepMeanSpeed=lambda d: -1.0,
        ),
        inductionloop=SimpleNamespace(getLastStepVehicleNumber=lambda d: 1),
    )
    heads = {tl: SimpleNamespace(stage=stage, green_time=green_time) for tl in network.tl_ids}
    sim = SimpleNamespace(api=api, heads=heads, network=network)
    # like Simulation.halting(), but reading the (possibly patched) fake API on every call
    sim.halting = lambda: {tl: approach_halting(sim.api, network, tl) for tl in network.tl_ids}
    return sim


def expected_dim(network, mode, comm):
    per_lane = N_CELLS + 1 if mode == 'grid' else DET_FEATS
    local = network.max_in_lanes * per_lane + network.max_greens + 1
    return local + (network.max_neighbours * MSG_SIZE if comm else 0)


@pytest.mark.parametrize('mode', ['grid', 'detector'])
@pytest.mark.parametrize('comm', [True, False])
@pytest.mark.parametrize('net_name', ['test5', 'single_4l'])
def test_shapes(request, net_name, mode, comm):
    network = request.getfixturevalue(net_name)
    obs = Observer(network, mode=mode, comm=comm)
    sim = fake_sim(network)
    assert obs.dim == expected_dim(network, mode, comm)
    for tl in network.tl_ids:
        s = obs.observe(sim, tl)
        assert s.shape == (obs.dim,) and s.dtype == np.float32
        mask = obs.action_mask(tl)
        assert mask.shape == (network.max_greens,) and mask.sum() == len(network.greens[tl])


def test_test5_dimensions(test5):
    # the dimensions the v0.2-era trained weights were built for
    assert Observer(test5, mode='grid', comm=True).dim == 63
    assert Observer(test5, mode='grid', comm=False).dim == 47
    assert Observer(test5, mode='detector', comm=True).dim == 35


def test_grid_uses_distance_to_stop_line(test5):
    tl = test5.tl_ids[0]
    lane = test5.in_lanes[tl][0]
    length = test5.lane_length[lane]
    obs = Observer(test5, mode='grid', comm=False)
    # vehicles 3 m and 50 m before the stop line -> cells 0 and 5 (bounds 7,14,21,28,40,55,...)
    s = obs.observe(fake_sim(test5, {lane: [length - 3.0, length - 50.0]}), tl)
    cells = s[:N_CELLS]
    assert cells[0] == 1.0 and cells[5] == 1.0 and cells.sum() == 2.0


def test_phase_features(test5):
    tl = test5.tl_ids[0]
    obs = Observer(test5, mode='detector', comm=False)
    s = obs.observe(fake_sim(test5, stage=1, green_time=30), tl)
    phase = s[test5.max_in_lanes * DET_FEATS:]
    assert list(phase) == [0.0, 1.0, 0.5]


def test_detector_noise_is_seeded(test5):
    tl = test5.tl_ids[0]
    a = Observer(test5, mode='detector', noise=0.1, dropout=0.2, seed=3)
    b = Observer(test5, mode='detector', noise=0.1, dropout=0.2, seed=3)
    sim = fake_sim(test5, halting=2)
    for _ in range(5):
        assert np.array_equal(a.observe(sim, tl), b.observe(sim, tl))


@pytest.mark.parametrize('kind', ['wait', 'multi', 'queue', 'det_queue', 'queue_pressure'])
def test_rewards_are_per_lane(test5, kind):
    tl = test5.tl_ids[0]
    r = Reward(test5, kind)
    value = r(fake_sim(test5, halting=2), tl)
    assert np.isfinite(value) and value <= 0


def test_unknown_reward():
    with pytest.raises(ValueError):
        Reward(SimpleNamespace(tl_ids=[]), 'nope')


def test_message_modes(test5):
    """'zero' blanks every neighbour slot; 'wrong' fills them with other junctions' messages."""
    obs = Observer(test5, mode='grid', comm=True)
    queues = {tl: i + 1 for i, tl in enumerate(test5.tl_ids)}   # distinct halting per junction
    sim = fake_sim(test5)
    sim.api.lane.getLastStepHaltingNumber = lambda l: next(
        q for tl, q in queues.items() if l in test5.in_lanes[tl])
    tl = 'J0'
    real = obs.observe(sim, tl)[obs.local_dim:]
    obs.set_message_mode('zero')
    assert not obs.observe(sim, tl)[obs.local_dim:].any()
    obs.set_message_mode('wrong', seed=0)
    wrong = obs.observe(sim, tl)[obs.local_dim:].reshape(-1, MSG_SIZE)
    n = len(test5.neighbours[tl])
    own = obs.message(sim, tl)
    assert not any(np.allclose(m, own) for m in wrong[:n])   # never the junction's own message
    assert not np.allclose(wrong.ravel(), real)
    with pytest.raises(ValueError):
        obs.set_message_mode('bogus')


def test_cooperative_reward(test5):
    """α = 0 gives the local rewards; α > 0 adds α × the neighbours' rewards."""
    sim = fake_sim(test5, halting=1)
    halting = {tl: i + 1 for i, tl in enumerate(test5.tl_ids)}
    sim.api.lane.getLastStepHaltingNumber = lambda l: next(
        h for tl, h in halting.items() if l in test5.in_lanes[tl])
    local = Reward(test5, 'queue').all(sim)
    assert local == {tl: Reward(test5, 'queue')(sim, tl) for tl in test5.tl_ids}
    coop = Reward(test5, 'queue', neighbour_weight=0.5).all(sim)
    for tl in test5.tl_ids:
        expected = local[tl] + 0.5 * sum(local[n] for n in test5.neighbours[tl])
        assert abs(coop[tl] - expected) < 1e-12


def test_approach_detector_feature(test5):
    """approach_length > 0 adds one feature per lane: vehicles moving toward the stop line
    in the longer area detector (count minus jam), normalised by its capacity."""
    tl = test5.tl_ids[0]
    obs = Observer(test5, mode='detector', comm=False, approach_length=150.0)
    plain = Observer(test5, mode='detector', comm=False)
    assert obs.dim == plain.dim + test5.max_in_lanes
    sim = fake_sim(test5, halting=2)
    sim.api.lanearea.getLastStepVehicleNumber = lambda d: 7 if d.startswith('e2a_') else 3
    s = obs.observe(sim, tl)
    lane = test5.in_lanes[tl][0]
    per_lane = s[:test5.max_in_lanes * (DET_FEATS + 1)].reshape(test5.max_in_lanes, DET_FEATS + 1)
    assert per_lane[0, -1] == pytest.approx((7 - 2) / obs.a_cap[lane])
    assert np.array_equal(per_lane[0, :DET_FEATS], plain.observe(sim, tl)[:DET_FEATS])


def test_smdp_discount_is_used():
    """A transition's own discount replaces gamma in the target (SMDP: gamma ** seconds)."""
    from tsc.agents.dqn import DQNAgent
    agent = DQNAgent(2, 2, gamma=0.9, lr=0.01, batch_size=1, learn_start=1, hidden=(8,))
    s = np.array([1.0, 0.0], dtype=np.float32)
    mask = np.array([True, True])
    agent.store(s, 0, 1.0, s, mask)
    assert agent.buffer[-1][5] == 0.9                     # default: gamma per decision
    agent.buffer.clear()
    agent.store(s, 0, 1.0, s, mask, discount=0.0)         # terminal-like: target = r
    for _ in range(400):
        agent.replay()
    assert agent.q_values(s)[0] == pytest.approx(1.0, abs=0.1)


def test_n_step_returns():
    """n_step = 3 stores the discounted sum of 3 rewards, bootstrapped 3 decisions later, per
    junction (key); the episode's last, shorter sequences are stored at its end."""
    from tsc.agents.dqn import DQNAgent
    agent = DQNAgent(1, 2, gamma=0.5, n_step=3, hidden=(4,))
    mask = np.array([True, True])
    state = [np.array([float(i)], dtype=np.float32) for i in range(6)]
    for i in range(4):
        agent.store(state[i], 0, 1.0, state[i + 1], mask, key='a')
    agent.store(state[0], 1, 8.0, state[5], mask, key='b')          # another junction
    assert len(agent.buffer) == 2
    s, a, r, s2, _, d = agent.buffer[0]
    assert s[0] == 0 and s2[0] == 3 and r == pytest.approx(1 + 0.5 + 0.25) and d == 0.125
    assert agent.buffer[1][0][0] == 1
    agent.end_episode()
    assert len(agent.buffer) == 5                       # 'a': 2 shorter, 'b': 1
    assert sorted(round(t[5], 3) for t in list(agent.buffer)[2:]) == [0.25, 0.5, 0.5]


def test_dueling_head_learns():
    from tsc.agents.dqn import DQNAgent
    agent = DQNAgent(2, 3, lr=0.01, batch_size=1, learn_start=1, hidden=(8,), dueling=True)
    s = np.array([1.0, 0.0], dtype=np.float32)
    agent.store(s, 2, 1.0, s, np.array([True] * 3), discount=0.0)
    for _ in range(400):
        agent.replay()
    assert agent.q_values(s)[2] == pytest.approx(1.0, abs=0.1)
