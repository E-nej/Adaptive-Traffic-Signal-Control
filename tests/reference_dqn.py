"""FROZEN reference copy of tsc.agents.dqn.DQNAgent as of v0.2.0 (commit 9d69577).

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


def build_qnet(n_in, n_out, hidden=(400, 400)):
    """The paper's network: first hidden layer linear, the others ReLU, linear output."""
    layers = [keras.Input(shape=(n_in,))]
    for i, units in enumerate(hidden):
        layers.append(keras.layers.Dense(units, activation=None if i == 0 else 'relu'))
    layers.append(keras.layers.Dense(n_out, activation='linear'))
    return keras.Sequential(layers)


class DQNAgent:
    @classmethod
    def from_config(cls, n_states, n_actions, cfg):
        """cfg: config.AgentConfig."""
        return cls(n_states, n_actions, double=cfg.double, target_update=cfg.target_update,
                   tau=cfg.tau, hard_every=cfg.hard_every, gamma=cfg.gamma, lr=cfg.lr,
                   batch_size=cfg.batch_size, buffer_size=cfg.buffer_size,
                   learn_start=cfg.learn_start, hidden=tuple(cfg.hidden))

    def __init__(self, n_states, n_actions, double=True, target_update='soft',
                 tau=0.005, hard_every=1000, gamma=0.95, lr=2.5e-4,
                 batch_size=100, buffer_size=50000, learn_start=500, hidden=(400, 400)):
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

        self.q = build_qnet(n_states, n_actions, hidden)
        self.q_target = build_qnet(n_states, n_actions, hidden)
        self.q_target.set_weights(self.q.get_weights())
        self.opt = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        self.huber = keras.losses.Huber()

    # ------------------------------------------------------------------

    def q_values(self, state):
        return self._predict(tf.convert_to_tensor(state[None, :]))[0].numpy()

    def act(self, state, mask, epsilon=0.0):
        valid = np.flatnonzero(mask)
        if np.random.random() < epsilon:
            return int(np.random.choice(valid))
        q = self.q_values(state)
        q[~mask] = -np.inf
        return int(np.argmax(q))

    def store(self, s, a, r, s2, mask2):
        self.buffer.append((s, a, r, s2, mask2))

    def replay(self):
        if len(self.buffer) < self.learn_start:
            return None
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s2, m2 = (np.array(x) for x in zip(*batch))
        loss = self._train_step(
            tf.convert_to_tensor(s, tf.float32), tf.convert_to_tensor(a, tf.int32),
            tf.convert_to_tensor(r, tf.float32), tf.convert_to_tensor(s2, tf.float32),
            tf.convert_to_tensor(m2, tf.bool),
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
    def _train_step(self, s, a, r, s2, m2):
        neg_inf = tf.fill(tf.shape(m2), -1e9)
        q_next_target = self.q_target(s2, training=False)
        if self.double:
            q_next_online = tf.where(m2, self.q(s2, training=False), neg_inf)
            best = tf.argmax(q_next_online, axis=1, output_type=tf.int32)
            next_v = tf.gather(q_next_target, best, axis=1, batch_dims=1)
        else:
            next_v = tf.reduce_max(tf.where(m2, q_next_target, neg_inf), axis=1)
        target = tf.stop_gradient(r + self.gamma * next_v)
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
