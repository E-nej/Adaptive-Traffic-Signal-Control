"""RL policy as a signal controller: one DQN agent per junction."""

import os

from tsc.controllers.base import Controller


def make_agents(network, observer, agent_cfg):
    from tsc.agents.dqn import DQNAgent   # TensorFlow is only imported when RL is used
    return {
        tl_id: DQNAgent.from_config(observer.dim, network.max_greens, agent_cfg)
        for tl_id in network.tl_ids
    }


class RLController(Controller):
    """Greedy (or epsilon-greedy) policy of one agent per junction."""

    def __init__(self, agents, observer, epsilon=0.0):
        self.agents = agents
        self.obs = observer
        self.epsilon = epsilon

    def reset(self, sim):
        self.obs.reset(sim)

    def tick(self, sim):
        self.obs.tick(sim)

    def act(self, sim, tl_id):
        s = self.obs.observe(sim, tl_id)
        return self.agents[tl_id].act(s, self.obs.action_mask(tl_id), self.epsilon)

    def load(self, run_dir, which='best'):
        for tl_id, agent in self.agents.items():
            agent.load(weights_path(run_dir, which, tl_id))
        return self


def weights_path(run_dir, which, tl_id):
    return os.path.join(run_dir, f'{which}_{tl_id}.weights.h5')
