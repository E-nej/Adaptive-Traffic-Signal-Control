import os
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import load_model  # type: ignore
import tensorflow as tf

import helpers  # sets up SUMO path and imports traci as a side effect
import traci  # type: ignore
from Model import Model

tf.keras.utils.disable_interactive_logging()

# Values each TL broadcasts to its neighbours via CommunicationModule.
# [normalised_queue, normalised_vehicle_count, normalised_avg_speed, normalised_phase]
COMM_MSG_SIZE = 4


class TLAgentMA:
    """Single-intersection DQN agent for the multi-agent scenario.

    Owns a Q-network and replay buffer but has no simulation loop — the
    shared training loop in main_multiagent.py drives step() and reset().
    """

    def __init__(self, tl_id, num_local_states, num_neighbours,
                 total_episodes, qmodel_filename=None, learn=True, num_vehicles=500):
        self.tl_id = tl_id
        # full state = local 88-dim + neighbour messages
        self.num_states = num_local_states + num_neighbours * COMM_MSG_SIZE
        self.num_actions = 4
        self.total_episodes = total_episodes
        self.num_vehicles = num_vehicles

        self.discount = 0.95
        self.batch_size = 100
        self.polyak = 0.005   # soft target update coefficient
        self.replay_buffer = deque(maxlen=50000)

        self.epsilon_start = 1.0
        self.epsilon_min   = 0.05
        self.epsilon_decay = 0.995  # hits floor (~0.05) around episode 600

        self.green_duration = 10
        self.yellow_duration = 4

        self.QModel = None
        self.TargetQModel = None
        self._load_models(qmodel_filename, learn)

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------

    def _load_models(self, qmodel_filename, learn):
        self.QModel = Model(self.num_states, self.num_actions)
        self.TargetQModel = Model(self.num_states, self.num_actions)
        self.TargetQModel.set_weights(self.QModel.get_weights())  # start in sync

        if qmodel_filename and os.path.exists(qmodel_filename) and not learn:
            loaded = load_model(qmodel_filename)
            self.QModel.model.set_weights(loaded.get_weights())
            self.TargetQModel.model.set_weights(loaded.get_weights())
            print(f'{self.tl_id}: weights loaded from {qmodel_filename}')

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def select_action(self, episode, state, learn=True):
        """state: np.array shape (1, num_states) — already preprocessed."""
        if learn:
            epsilon = max(self.epsilon_min,
                          self.epsilon_start * (self.epsilon_decay ** episode))
            if np.random.random() <= epsilon:
                return np.random.randint(self.num_actions)
        return int(np.argmax(self.QModel.predict(state)))

    # ------------------------------------------------------------------
    # Experience replay
    # ------------------------------------------------------------------

    def store(self, curr_state, action, reward, next_state, done):
        """States must be preprocessed (shape [1, num_states])."""
        self.replay_buffer.append((curr_state, action, reward, next_state, done))

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        mini_batch = random.sample(self.replay_buffer, self.batch_size)

        curr_states = np.array([m[0][0] for m in mini_batch])
        next_states = np.array([m[3][0] for m in mini_batch])

        q_curr        = self.QModel.predict(curr_states)
        q_next_online = self.QModel.predict(next_states)        # action selection
        q_next_target = self.TargetQModel.predict(next_states)  # action evaluation (Double DQN)

        for i, (_, action, reward, _, done) in enumerate(mini_batch):
            if done:
                q_curr[i][action] = reward
            else:
                best_next = np.argmax(q_next_online[i])
                q_curr[i][action] = reward + self.discount * q_next_target[i][best_next]

        self.QModel.model.train_on_batch(curr_states, q_curr)

    def sync_target(self):
        """Polyak (soft) update: θ_target ← τ·θ_online + (1−τ)·θ_target."""
        q_w = self.QModel.get_weights()
        t_w = self.TargetQModel.get_weights()
        self.TargetQModel.set_weights([
            self.polyak * qw + (1 - self.polyak) * tw
            for qw, tw in zip(q_w, t_w)
        ])

    # ------------------------------------------------------------------
    # Phase control  (called by the shared training loop)
    # ------------------------------------------------------------------

    def set_green_phase(self, action):
        helpers.set_green_phase(self.tl_id, action)

    def set_yellow_phase(self, old_action):
        helpers.set_yellow_phase(self.tl_id, old_action)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def preprocess(self, state):
        return np.reshape(state, [1, self.num_states])

    def build_comm_message(self, env):
        """4-value message broadcast to neighbours: [queue, count, speed, phase]."""
        roads = env.tl_config[self.tl_id]["incoming_roads"]
        nv = max(1, self.num_vehicles)
        return [
            env._get_queue(roads)         / nv,
            env._get_vehicle_count(roads) / nv,
            env._get_avg_speed(roads)     / 25,
            traci.trafficlight.getPhase(self.tl_id) / 7,
        ]
