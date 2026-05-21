import os
import sys

sumo_paths = [
    '/usr/share/sumo/tools',
    'C:\\Program Files (x86)\\Eclipse\\Sumo\\tools',
    'C:\\Program Files\\Sumo\\tools',
]
s_h = os.environ.get('SUMO_HOME')
if s_h:
    sumo_paths.insert(0, os.path.join(s_h, 'tools'))
for p in sumo_paths:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)

import traci  # type: ignore
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import load_model  # type: ignore
import tensorflow as tf

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
                 total_episodes, qmodel_filename=None, learn=True):
        self.tl_id = tl_id
        # full state = local 88-dim + neighbour messages
        self.num_states  = num_local_states + num_neighbours * COMM_MSG_SIZE
        self.num_actions = 4
        self.total_episodes = total_episodes

        self.discount     = 0.95
        self.batch_size   = 100
        self.tau          = 20
        self.replay_buffer = deque(maxlen=50000)

        # TL phase indices (same order as single-agent setup)
        self.PHASE_NS_GREEN   = 0
        self.PHASE_NS_YELLOW  = 1
        self.PHASE_NSL_GREEN  = 2
        self.PHASE_NSL_YELLOW = 3
        self.PHASE_EW_GREEN   = 4
        self.PHASE_EW_YELLOW  = 5
        self.PHASE_EWL_GREEN  = 6
        self.PHASE_EWL_YELLOW = 7

        self.green_duration  = 10
        self.yellow_duration = 4

        self.QModel       = None
        self.TargetQModel = None
        self._load_models(qmodel_filename, learn)

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------

    def _load_models(self, qmodel_filename, learn):
        self.QModel       = Model(self.num_states, self.num_actions)
        self.TargetQModel = Model(self.num_states, self.num_actions)

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
            epsilon = max(0.05, 1 - episode / self.total_episodes)
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

        q_curr = self.QModel.predict(curr_states)
        q_next = self.TargetQModel.predict(next_states)

        for i, (_, action, reward, _, done) in enumerate(mini_batch):
            q_curr[i][action] = reward if done else reward + self.discount * np.max(q_next[i])

        self.QModel.model.train_on_batch(curr_states, q_curr)

    def sync_target(self):
        self.TargetQModel.set_weights(self.QModel.get_weights())

    # ------------------------------------------------------------------
    # Phase control  (called by the shared training loop)
    # ------------------------------------------------------------------

    def set_green_phase(self, action):
        phase_map = {
            0: self.PHASE_NS_GREEN,
            1: self.PHASE_NSL_GREEN,
            2: self.PHASE_EW_GREEN,
            3: self.PHASE_EWL_GREEN,
        }
        traci.trafficlight.setPhase(self.tl_id, phase_map[action])

    def set_yellow_phase(self, old_action):
        traci.trafficlight.setPhase(self.tl_id, old_action * 2 + 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def preprocess(self, state):
        return np.reshape(state, [1, self.num_states])

    def build_comm_message(self, env):
        """4-value message broadcast to neighbours: [queue, count, speed, phase]."""
        roads = env.tl_config[self.tl_id]["incoming_roads"]
        return [
            env._get_queue(roads)         / 500,
            env._get_vehicle_count(roads) / 500,
            env._get_avg_speed(roads)     / 25,
            traci.trafficlight.getPhase(self.tl_id) / 7,
        ]
