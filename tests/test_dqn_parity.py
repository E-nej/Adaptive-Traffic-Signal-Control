"""CPU reference parity of the DQN agent (parity level 1, see CLAUDE.md).

The current tsc.agents.dqn.DQNAgent must reproduce the frozen v0.2.0 implementation
(tests/reference_dqn.py) bit for bit: the same initial weights, and after every
replay() step the same loss, online weights, target weights, optimizer state and
consumption of Python's and NumPy's global RNGs.  Both agents run from the same RNG
state at every step (saved and restored around the reference call).
"""

import random

import numpy as np
import pytest
import tensorflow as tf
from reference_dqn import DQNAgent as ReferenceAgent

from tsc.agents.dqn import DQNAgent

N_STATES, N_ACTIONS, N_VALID = 150, 4, 3
STEPS = 60

MODES = {
    'soft_double': dict(double=True, target_update='soft'),
    'hard_single': dict(double=False, target_update='hard', hard_every=7),
    'hard_double': dict(double=True, target_update='hard', hard_every=7),
}


def build(cls, **kw):
    """Build an agent from a fixed global seed, as train_one seeds before make_agents."""
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)
    return cls(N_STATES, N_ACTIONS, batch_size=32, learn_start=32, buffer_size=500, **kw)


def transitions(n, seed=123):
    rng = np.random.default_rng(seed)
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[:N_VALID] = True
    for _ in range(n):
        yield (rng.random(N_STATES, dtype=np.float32), int(rng.integers(N_VALID)),
               -float(rng.random()), rng.random(N_STATES, dtype=np.float32), mask)


def rng_state():
    return random.getstate(), np.random.get_state()


def set_rng_state(state):
    random.setstate(state[0])
    np.random.set_state(state[1])


def assert_same_rng(a, b):
    assert a[0] == b[0], 'Python random state differs'
    for x, y in zip(a[1], b[1]):
        assert np.array_equal(x, y), 'NumPy global RNG state differs'


def assert_same_arrays(xs, ys, what):
    assert len(xs) == len(ys), what
    for i, (x, y) in enumerate(zip(xs, ys)):
        assert np.array_equal(x, y), f'{what}[{i}] differs'


def opt_state(agent):
    return [v.numpy() for v in agent.opt.variables]


def assert_same_agent(ref, new, step):
    assert_same_arrays(ref.q.get_weights(), new.q.get_weights(), f'online weights after step {step}')
    assert_same_arrays(ref.q_target.get_weights(), new.q_target.get_weights(),
                       f'target weights after step {step}')
    assert_same_arrays(opt_state(ref), opt_state(new), f'optimizer state after step {step}')


@pytest.mark.parametrize('mode', MODES)
def test_replay_matches_reference(mode):
    ref = build(ReferenceAgent, **MODES[mode])
    new = build(DQNAgent, **MODES[mode])
    assert_same_agent(ref, new, 'init')

    data = list(transitions(STEPS + 40))
    for t in data[:40]:                       # buffer above learn_start
        ref.store(*t)
        new.store(*t)
    for step, t in enumerate(data[40:]):
        ref.store(*t)
        new.store(*t)
        before = rng_state()
        loss_ref = ref.replay()
        after_ref = rng_state()
        set_rng_state(before)
        loss_new = new.replay()
        assert_same_rng(after_ref, rng_state())
        assert loss_ref == loss_new, f'loss differs at step {step}'
        assert_same_agent(ref, new, step)


def test_act_matches_reference():
    ref = build(ReferenceAgent)
    new = build(DQNAgent)
    states = [t[0] for t in transitions(200, seed=7)]
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[:N_VALID] = True
    for i, s in enumerate(states):
        eps = (0.0, 0.3, 1.0)[i % 3]
        before = rng_state()
        a_ref = ref.act(s, mask, eps)
        after_ref = rng_state()
        set_rng_state(before)
        a_new = new.act(s, mask, eps)
        assert_same_rng(after_ref, rng_state())
        assert a_ref == a_new, f'action differs at state {i}'
        assert np.array_equal(ref.q_values(s), new.q_values(s)), f'Q-values differ at state {i}'
