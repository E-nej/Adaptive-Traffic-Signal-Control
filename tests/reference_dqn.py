"""FROZEN reference copy of tsc.agents.dqn.DQNAgent as of main a1057e9 (v0.2.0 plus the
2F-2I options: done/discount column, n-step, dueling, ref_action, phase_net), i.e. the agent
before the training speedups of perf/training-pipeline.

Test oracle for tests/test_dqn_parity.py: every optimisation of the real agent must
reproduce this implementation bit for bit (weights, target weights, optimizer state,
losses and RNG consumption).  Do not edit; only the thread configuration was dropped
(it is process-wide and already done by tsc.agents.dqn).
"""

import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras


def build_qnet(n_in, n_out, hidden=(400, 400), dueling=False, ref_action=-1):
    """The paper's network: first hidden layer linear, the others ReLU, linear output.

    dueling: the output is V(s) + A(s, a) - mean_a A(s, a) (Wang et al. 2016).
    ref_action >= 0: Q(s, ref) = V(s) and Q(s, a) = V(s) + A(s, a) for the other actions,
    with the advantage layer initialised to zero.  All actions start with the same value and
    argmax ties go to the lowest index, so with ref_action = 0 the untrained policy takes
    action 0 (a worker's AGREE: it starts as its expert, as in residual policy learning).
    V is learned from every transition, an action's advantage only from that action's
    transitions, so untried actions do not look better (or worse) than the reference just
    because V has moved.
    """
    if ref_action >= 0:
        x = inp = keras.Input(shape=(n_in,))
        for i, units in enumerate(hidden):
            x = keras.layers.Dense(units, activation=None if i == 0 else 'relu')(x)
        v = keras.layers.Dense(1)(x)
        adv = keras.layers.Dense(n_out, kernel_initializer='zeros', bias_initializer='zeros')(x)
        keep = np.ones((1, n_out), dtype='float32')
        keep[0, ref_action] = 0.0                # the reference action has no advantage
        out = v + adv * keep
        return keras.Model(inp, out)
    if dueling:
        x = inp = keras.Input(shape=(n_in,))
        for i, units in enumerate(hidden):
            x = keras.layers.Dense(units, activation=None if i == 0 else 'relu')(x)
        v = keras.layers.Dense(1)(x)
        adv = keras.layers.Dense(n_out)(x)
        out = v + adv - keras.ops.mean(adv, axis=1, keepdims=True)
        return keras.Model(inp, out)
    layers = [keras.Input(shape=(n_in,))]
    for i, units in enumerate(hidden):
        layers.append(keras.layers.Dense(units, activation=None if i == 0 else 'relu'))
    layers.append(keras.layers.Dense(n_out, activation='linear'))
    return keras.Sequential(layers)


def build_phase_qnet(n_stages, n_feats, hidden=(64, 64)):
    """Phase-wise Q-network (FRAP, Zheng et al. 2019; MPLight, Chen et al. 2020).

    Input (n_stages, n_feats): one row per green stage (observation layout 'phase').  Every
    stage is embedded by the same MLP; its Q-value comes from its own embedding and the mean
    embedding of all stages (the competition between stages).  The weights do not depend on
    the junction, its orientation or its number of stages, so one network serves every
    junction (padded stages are masked by the agent).
    """
    x = keras.Input(shape=(n_stages, n_feats))
    h = x
    for units in hidden:
        h = keras.layers.Dense(units, activation='relu')(h)
    ctx = keras.ops.mean(h, axis=1, keepdims=True) * keras.ops.ones_like(h[:, :, :1])
    z = keras.layers.Concatenate()([h, ctx])
    z = keras.layers.Dense(hidden[-1], activation='relu')(z)
    q = keras.layers.Dense(1)(z)
    return keras.Model(x, keras.ops.squeeze(q, axis=-1))


class DQNAgent:
    @classmethod
    def from_config(cls, n_states, n_actions, cfg):
        """cfg: config.AgentConfig."""
        return cls(n_states, n_actions, double=cfg.double, target_update=cfg.target_update,
                   tau=cfg.tau, hard_every=cfg.hard_every, gamma=cfg.gamma, lr=cfg.lr,
                   batch_size=cfg.batch_size, buffer_size=cfg.buffer_size,
                   learn_start=cfg.learn_start, hidden=tuple(cfg.hidden),
                   dueling=cfg.dueling, n_step=cfg.n_step, ref_action=cfg.ref_action,
                   phase_net=cfg.phase_net)

    def __init__(self, n_states, n_actions, double=True, target_update='soft',
                 tau=0.005, hard_every=1000, gamma=0.95, lr=2.5e-4,
                 batch_size=100, buffer_size=50000, learn_start=500, hidden=(400, 400),
                 dueling=False, n_step=1, ref_action=-1, phase_net=False):
        """n_states: input size, or (n_stages, n_feats) for the phase-wise network."""
        self.n_states = n_states
        self.n_actions = n_actions
        self.double = double
        self.target_update = target_update
        self.tau = tau
        self.hard_every = hard_every
        self.gamma = gamma
        self.batch_size = batch_size
        self.learn_start = max(learn_start, batch_size)
        self.buffer = deque(maxlen=buffer_size)
        self.grad_steps = 0
        self.n_step = n_step
        self.recent = {}                  # key -> last transitions not yet n steps old

        if phase_net:
            build = lambda: build_phase_qnet(*n_states, hidden=hidden)  # noqa: E731
        else:
            build = lambda: build_qnet(n_states, n_actions, hidden, dueling, ref_action)  # noqa: E731
        self.q = build()
        self.q_target = build()
        self.q_target.set_weights(self.q.get_weights())
        self.opt = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        self.huber = keras.losses.Huber()

    # ------------------------------------------------------------------

    def q_values(self, state):
        return self._predict(tf.convert_to_tensor(state[None, ...]))[0].numpy()

    def act(self, state, mask, epsilon=0.0):
        valid = np.flatnonzero(mask)
        if np.random.random() < epsilon:
            return int(np.random.choice(valid))
        q = self.q_values(state)
        q[~mask] = -np.inf
        return int(np.argmax(q))

    def store(self, s, a, r, s2, mask2, discount=None, key=None):
        """discount: of the next state's value (default gamma; SMDP: gamma ** seconds).

        With n_step > 1 the stored transition is n decisions long: the discounted sum of
        n rewards, bootstrapped from the state n decisions later.  `key` separates the
        decision sequences of junctions sharing one agent.
        """
        d = self.gamma if discount is None else discount
        if self.n_step == 1:
            self.buffer.append((s, a, r, s2, mask2, d))
            return
        recent = self.recent.setdefault(key, deque())
        recent.append((s, a, r, s2, mask2, d))
        if len(recent) == self.n_step:
            self.buffer.append(self._n_step(recent))
            recent.popleft()

    def end_episode(self):
        """Store the last, shorter n-step transitions of the episode."""
        for recent in self.recent.values():
            while recent:
                self.buffer.append(self._n_step(recent))
                recent.popleft()
        self.recent = {}

    @staticmethod
    def _n_step(seq):
        s, a = seq[0][:2]
        ret, disc = 0.0, 1.0
        for _, _, r, s2, mask2, d in seq:
            ret += disc * r
            disc *= d
        return s, a, ret, s2, mask2, disc

    def replay(self):
        if len(self.buffer) < self.learn_start:
            return None
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s2, m2, d = (np.array(x) for x in zip(*batch))
        loss = self._train_step(
            tf.convert_to_tensor(s, tf.float32), tf.convert_to_tensor(a, tf.int32),
            tf.convert_to_tensor(r, tf.float32), tf.convert_to_tensor(s2, tf.float32),
            tf.convert_to_tensor(m2, tf.bool), tf.convert_to_tensor(d, tf.float32),
        )
        self.grad_steps += 1
        if self.target_update == 'soft':
            self._soft_update()
        elif self.grad_steps % self.hard_every == 0:
            self.q_target.set_weights(self.q.get_weights())
        return float(loss)

    # ------------------------------------------------------------------

    @tf.function
    def _predict(self, x):
        return self.q(x, training=False)

    @tf.function
    def _train_step(self, s, a, r, s2, m2, d):
        neg_inf = tf.fill(tf.shape(m2), -1e9)
        q_next_target = self.q_target(s2, training=False)
        if self.double:
            q_next_online = tf.where(m2, self.q(s2, training=False), neg_inf)
            best = tf.argmax(q_next_online, axis=1, output_type=tf.int32)
            next_v = tf.gather(q_next_target, best, axis=1, batch_dims=1)
        else:
            next_v = tf.reduce_max(tf.where(m2, q_next_target, neg_inf), axis=1)
        target = tf.stop_gradient(r + d * next_v)
        with tf.GradientTape() as tape:
            q_sa = tf.gather(self.q(s, training=True), a, axis=1, batch_dims=1)
            loss = self.huber(target, q_sa)
        grads = tape.gradient(loss, self.q.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q.trainable_variables))
        return loss

    @tf.function
    def _soft_update(self):
        for t, w in zip(self.q_target.trainable_variables, self.q.trainable_variables):
            t.assign(self.tau * w + (1.0 - self.tau) * t)

    # ------------------------------------------------------------------

    def get_weights(self):
        return self.q.get_weights()

    def set_weights(self, w):
        self.q.set_weights(w)
        self.q_target.set_weights(w)

    def save(self, path):
        self.q.save_weights(path)

    def load(self, path):
        self.q.load_weights(path)
        self.q_target.set_weights(self.q.get_weights())
