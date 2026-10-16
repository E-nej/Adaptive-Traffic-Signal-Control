"""CPU reference parity of the DQN agent (parity level 1, see CLAUDE.md).

The current tsc.agents.dqn.DQNAgent must reproduce the frozen pre-speedup implementation
(tests/reference_dqn.py) bit for bit: the same initial weights, and after every
replay() step the same loss, online weights, target weights, optimizer state and
consumption of Python's and NumPy's global RNGs.  Both agents run from the same RNG
state at every step (saved and restored around the reference call).  The modes cover
the agent options: target update, Double DQN, SMDP discounts, n-step returns (two
junctions sharing the agent), dueling, the ref_action head and the phase-wise network.
"""

import random

import numpy as np
import pytest
import torch
from reference_dqn import DQNAgent as ReferenceAgent

from tsc.agents.dqn import DQNAgent

N_STATES, N_ACTIONS, N_VALID = 150, 4, 3
N_FEATS = 12                        # phase-wise network: (N_ACTIONS stages, N_FEATS)
STEPS = 60

MODES = {
    'soft_double': dict(double=True, target_update='soft'),
    'hard_single': dict(double=False, target_update='hard', hard_every=7),
    'hard_double': dict(double=True, target_update='hard', hard_every=7),
    'smdp': dict(double=True, target_update='soft'),
    'n_step': dict(double=True, target_update='soft', n_step=3),
    'dueling': dict(double=True, target_update='soft', dueling=True),
    'ref_action': dict(double=True, target_update='soft', ref_action=0),
    'phase_net': dict(double=True, target_update='soft', phase_net=True, hidden=(64, 64)),
}


def n_states(kw):
    return (N_ACTIONS, N_FEATS) if kw.get('phase_net') else N_STATES


def build(cls, **kw):
    """Build an agent from a fixed global seed, as train_one seeds before make_agents."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    return cls(n_states(kw), N_ACTIONS, batch_size=32, learn_start=32, buffer_size=500, **kw)


def transitions(n, shape=N_STATES, seed=123, smdp=False):
    """(s, a, r, s2, mask, discount, key); discount None (gamma) unless smdp; two keys, as
    two junctions sharing one agent."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[:N_VALID] = True
    for i in range(n):
        discount = 0.95 ** int(rng.integers(5, 40)) if smdp else None
        yield (rng.random(shape, dtype=np.float32), int(rng.integers(N_VALID)),
               -float(rng.random()), rng.random(shape, dtype=np.float32), mask,
               discount, f'tl{i % 2}')


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


def assert_same_agent(ref, new, step):
    a, b = ref.state_arrays(), new.state_arrays()
    for part in ('online', 'target', 'optimizer'):
        assert_same_arrays(a[part], b[part], f'{part} after step {step}')


@pytest.mark.parametrize('mode', MODES)
def test_replay_matches_reference(mode):
    kw = MODES[mode]
    ref = build(ReferenceAgent, **kw)
    new = build(DQNAgent, **kw)
    assert_same_agent(ref, new, 'init')

    data = list(transitions(STEPS + 40, n_states(kw), smdp=mode == 'smdp'))
    for t in data[:40]:                       # buffer above learn_start
        ref.store(*t)
        new.store(*t)
    for step, t in enumerate(data[40:]):
        ref.store(*t)
        new.store(*t)
        if step == STEPS // 2:                # an episode end (flushes the n-step queues)
            ref.end_episode()
            new.end_episode()
        before = rng_state()
        loss_ref = ref.replay()
        after_ref = rng_state()
        set_rng_state(before)
        loss_new = new.replay()
        assert_same_rng(after_ref, rng_state())
        assert loss_ref == loss_new, f'loss differs at step {step}'
        assert_same_agent(ref, new, step)


@pytest.mark.parametrize('mode', ['soft_double', 'dueling', 'ref_action', 'phase_net'])
def test_act_matches_reference(mode):
    kw = MODES[mode]
    ref = build(ReferenceAgent, **kw)
    new = build(DQNAgent, **kw)
    states = [t[0] for t in transitions(200, n_states(kw), seed=7)]
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
