"""The PyTorch agent against recorded runs of the former TensorFlow agent.

tests/fixtures/make_tf_reference.py recorded, per agent mode, the TensorFlow agent's
initial weights, the transitions, every replay() loss, the weights at some steps and the
Q-values / greedy actions on probe states.  Here the PyTorch agent starts from the same
weights, gets the same transitions and the same replay batches (random.sample from the
same seed), and must reproduce all of it up to float32 rounding (the kernels sum in a
different order, so bit equality is not expected).
"""

import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fixtures'))
import make_tf_reference as ref  # noqa: E402

from tsc.agents.dqn import DQNAgent  # noqa: E402

# The fused Adam / foreach updates round differently from Keras' op-by-op updates, which
# compounds to ~5e-7 in the weights over 60 steps (at most 0.43 of this tolerance in any
# mode).  Real mistakes are far outside it: PyTorch's own epsilon placement gives 18-870x
# the tolerance, global instead of per-weight gradient clipping 2600-8300x where it clips.
RTOL, ATOL = 1e-4, 1e-5


def load(name):
    with np.load(os.path.join(ref.OUT, f'{name}.npz')) as f:
        data = dict(f)

    def weights(key):
        n = sum(1 for k in data if k.startswith(f'{key}/'))
        return [data[f'{key}/{i}'] for i in range(n)]
    return data, weights


def transitions(data):
    out = []
    for i in range(len(data['data/a'])):
        d = data['data/discount'][i]
        out.append((data['data/s'][i], int(data['data/a'][i]), float(data['data/r'][i]),
                    data['data/s2'][i], data['data/mask2'][i],
                    None if np.isnan(d) else float(d), str(data['data/key'][i])))
    return out


def assert_close(xs, ys, what):
    assert len(xs) == len(ys), what
    for i, (x, y) in enumerate(zip(xs, ys)):
        np.testing.assert_allclose(x, y, rtol=RTOL, atol=ATOL, err_msg=f'{what}[{i}]')


@pytest.mark.parametrize('mode', ref.MODES)
def test_matches_tensorflow(mode):
    kw = ref.MODES[mode]
    data, weights = load(mode)
    agent = DQNAgent(ref.n_states(kw), ref.N_ACTIONS, **ref.COMMON, **kw)
    agent.set_weights(weights('init'))
    probes, mask = data['probes'], data['mask']
    np.testing.assert_allclose(np.stack([agent.q_values(p) for p in probes]), data['q_init'],
                               rtol=RTOL, atol=ATOL)
    assert [agent.act(p, mask, 0.0) for p in probes] == list(data['act_init'])

    steps = transitions(data)
    for t in steps[:ref.WARMUP]:
        agent.store(*t)
    random.seed(1)
    for step, t in enumerate(steps[ref.WARMUP:]):
        agent.store(*t)
        if step == ref.END_EPISODE_AT:
            agent.end_episode()
        loss = agent.replay()
        expected = data['losses'][step]
        if np.isnan(expected):
            assert loss is None, f'step {step}'
        else:
            assert loss == pytest.approx(expected, rel=RTOL, abs=ATOL), f'loss at step {step}'
        if step in ref.SAVE_AT:
            state = agent.state_arrays()
            assert_close(state['online'], weights(f'online_{step}'), f'online weights, step {step}')
            assert_close(state['target'], weights(f'target_{step}'), f'target weights, step {step}')
    assert len(agent.buffer) == int(data['buffer_len'])
    np.testing.assert_allclose(np.stack([agent.q_values(p) for p in probes]), data['q_final'],
                               rtol=RTOL, atol=ATOL)
    assert [agent.act(p, mask, 0.0) for p in probes] == list(data['act_final'])
