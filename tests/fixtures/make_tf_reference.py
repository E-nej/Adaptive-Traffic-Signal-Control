"""Record the TensorFlow agent's behaviour as fixtures for the PyTorch port (run once).

    uv run python tests/fixtures/make_tf_reference.py      # on the last TensorFlow commit

For every agent mode it writes tests/fixtures/tf_reference/<mode>.npz: the initial online
weights (Keras layout: kernel (in, out), bias, layer by layer in the canonical order
below), the transitions fed to store(), the loss of every replay() step, the online and
target weights after some steps, and Q-values / greedy actions on probe states before and
after training.  tests/test_tf_equivalence.py loads the initial weights into the PyTorch
agent, replays the same steps (the same random.sample batches) and compares.

Canonical layer order: the hidden layers in order, then the output layer (plain), or V
then A (dueling, ref_action), or for the phase-wise network the per-stage hidden layers,
the layer after the concatenation, and the scalar output.
"""

import os
import random
import sys

import numpy as np

HIDDEN = (32, 24)
N_STATES, N_ACTIONS, N_VALID = 20, 4, 3
N_FEATS = 12                      # phase-wise network: (N_ACTIONS stages, N_FEATS)
WARMUP, STEPS = 40, 60
SAVE_AT = (0, 1, 5, 20, STEPS - 1)
END_EPISODE_AT = STEPS // 2
COMMON = dict(batch_size=16, learn_start=16, buffer_size=500, lr=1e-3, tau=0.05,
              hidden=HIDDEN)

MODES = {
    'soft_double': dict(double=True, target_update='soft'),
    'hard_single': dict(double=False, target_update='hard', hard_every=7),
    'hard_double': dict(double=True, target_update='hard', hard_every=7),
    'smdp': dict(double=True, target_update='soft'),
    'n_step': dict(double=True, target_update='soft', n_step=3),
    'dueling': dict(double=True, target_update='soft', dueling=True),
    'ref_action': dict(double=True, target_update='soft', ref_action=0),
    'phase_net': dict(double=True, target_update='soft', phase_net=True),
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tf_reference')


def n_states(kw):
    return (N_ACTIONS, N_FEATS) if kw.get('phase_net') else N_STATES


def transitions(n, shape, smdp, seed=123):
    rng = np.random.default_rng(seed)
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[:N_VALID] = True
    out = []
    for i in range(n):
        discount = 0.95 ** int(rng.integers(5, 40)) if smdp else None
        out.append((rng.random(shape, dtype=np.float32), int(rng.integers(N_VALID)),
                    -float(rng.random()), rng.random(shape, dtype=np.float32), mask.copy(),
                    discount, f'tl{i % 2}'))
    return out


def canonical(model, kw):
    """Keras weights as [kernel, bias, ...] in the canonical layer order (see module doc)."""
    dense = [layer for layer in model.layers if layer.weights]
    n_hidden = len(HIDDEN)
    if kw.get('dueling') or kw.get('ref_action', -1) >= 0:
        hidden, rest = dense[:n_hidden], dense[n_hidden:]
        v = [layer for layer in rest if layer.units == 1]
        a = [layer for layer in rest if layer.units == N_ACTIONS]
        assert len(v) == len(a) == 1, [layer.units for layer in rest]
        dense = hidden + v + a
    out = []
    for layer in dense:
        out += [w.numpy() for w in layer.weights]       # kernel, bias
    assert len(out) == len(model.get_weights())
    return out


def record(name, kw):
    from tsc.agents.dqn import DQNAgent
    import tensorflow as tf
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)
    agent = DQNAgent(n_states(kw), N_ACTIONS, **COMMON, **kw)
    data = transitions(WARMUP + STEPS, n_states(kw), name == 'smdp')
    probes = np.random.default_rng(7).random((20, *np.atleast_1d(n_states(kw))), dtype=np.float32)
    mask = data[0][4]

    def greedy(a):
        return np.array([a.act(p, mask, 0.0) for p in probes])

    out = {'init': canonical(agent.q, kw),
           'q_init': np.stack([agent.q_values(p) for p in probes]), 'act_init': greedy(agent)}
    for t in data[:WARMUP]:
        agent.store(*t)
    random.seed(1)                              # the replay batches (random.sample)
    losses = []
    for step, t in enumerate(data[WARMUP:]):
        agent.store(*t)
        if step == END_EPISODE_AT:
            agent.end_episode()
        loss = agent.replay()
        losses.append(np.nan if loss is None else loss)
        if step in SAVE_AT:
            out[f'online_{step}'] = canonical(agent.q, kw)
            out[f'target_{step}'] = canonical(agent.q_target, kw)
    out['q_final'] = np.stack([agent.q_values(p) for p in probes])
    out['act_final'] = greedy(agent)

    flat = {'losses': np.array(losses, dtype=np.float64), 'probes': probes,
            'mask': mask, 'buffer_len': np.array(len(agent.buffer))}
    for key, value in out.items():
        if isinstance(value, list):
            for i, w in enumerate(value):
                flat[f'{key}/{i}'] = w
        else:
            flat[key] = value
    for i, field in enumerate(('s', 'a', 'r', 's2', 'mask2', 'discount')):
        col = [t[i] for t in data]
        if field == 'discount':
            col = [np.nan if d is None else d for d in col]
        flat[f'data/{field}'] = np.array(col)
    flat['data/key'] = np.array([t[6] for t in data])
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f'{name}.npz'), **flat)
    print(f'{name}: {len(losses)} steps, final loss {losses[-1]:.6f}')


if __name__ == '__main__':
    for name in sys.argv[1:] or MODES:
        record(name, MODES[name])
