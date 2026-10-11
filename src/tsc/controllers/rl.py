"""RL policy as a signal controller: one DQN agent per junction (or one shared agent).

What an action means depends on the control mode ([signals] control):

'stages' : the action is the next green stage (StageActions)
'plan'   : the junction runs a coordinated plan and the action adjusts it
           (FOLLOW / HOLD / EARLY, see controllers.plan.PlanActions)
'select' : the action picks which controller runs the junction for the next
           ~cycle (green wave / coordinated-actuated / max-pressure, see
           controllers.select.SelectorActions)

With a shared agent (agent.shared) all junctions use one network, trained on
the transitions of every junction; a one-hot junction id is appended to the
observation so the policy can still differ per junction.
"""

import os

import numpy as np

from tsc.controllers.base import Controller
from tsc.controllers.plan import PlanActions
from tsc.controllers.select import SelectorActions


class StageActions:
    """Action = index of the next green stage (the original action space)."""

    extra_dim = 0

    def __init__(self, network, observer):
        self.n_actions = network.max_greens
        self.observer = observer

    def set_plans(self, plans):
        pass

    def reset(self, sim):
        pass

    def tick(self, sim):
        pass

    def decides(self, sim, tl_id):
        return True

    def mask(self, sim, tl_id):
        return self.observer.action_mask(tl_id)

    def stage(self, sim, tl_id, action):
        return action

    def features(self, sim, tl_id):
        return np.zeros(0, dtype=np.float32)


def make_actions(network, observer, cfg):
    """Action space of a Config (signals.control)."""
    s = cfg.signals
    if s.control == 'plan':
        return PlanActions(network, interval=s.decision_interval, max_extension=s.max_extension)
    if s.control == 'select':
        o = cfg.observation
        return SelectorActions(network, select_interval=s.select_interval,
                               interval=s.decision_interval, max_extension=s.max_extension,
                               noise=o.noise, dropout=o.dropout, e2_length=o.e2_length,
                               comm=o.comm, seed=cfg.train.seeds[0])
    if s.control == 'stages':
        return StageActions(network, observer)
    raise ValueError(f'Unknown control mode {s.control!r}')


def make_agents(network, observer, agent_cfg, actions=None):
    from tsc.agents.dqn import DQNAgent   # TensorFlow is only imported when RL is used
    n_actions = actions.n_actions if actions is not None else network.max_greens
    dim = observer.dim + (actions.extra_dim if actions is not None else 0)
    if agent_cfg.shared:
        agent = DQNAgent.from_config(dim + len(network.tl_ids), n_actions, agent_cfg)
        return {tl_id: agent for tl_id in network.tl_ids}
    return {tl_id: DQNAgent.from_config(dim, n_actions, agent_cfg) for tl_id in network.tl_ids}


class RLController(Controller):
    """Greedy (or epsilon-greedy) policy of the agents."""

    def __init__(self, agents, observer, epsilon=0.0, actions=None, shared=False):
        self.agents = agents
        self.obs = observer
        self.epsilon = epsilon
        self.actions = actions if actions is not None else StageActions(observer.network, observer)
        self.shared = shared
        self.tl_index = {tl_id: i for i, tl_id in enumerate(sorted(agents))}
        self.last = {}

    @property
    def needs_plan(self):
        return getattr(self.actions, 'needs_plan', False)

    def set_plans(self, plans):
        self.actions.set_plans(plans)

    def reseed(self, seed):
        """Independent sensor noise per evaluation scenario (selector's max-pressure)."""
        if hasattr(self.actions, 'reseed'):
            self.actions.reseed(seed)

    def reset(self, sim):
        self.obs.reset(sim)
        self.actions.reset(sim)
        self.last = {}

    def tick(self, sim):
        self.obs.tick(sim)
        self.actions.tick(sim)

    def state(self, sim, tl_id):
        """Policy input: observation, action-space features, junction id (shared agent)."""
        s = self.obs.observe(sim, tl_id)
        extra = self.actions.features(sim, tl_id)
        if self.shared:
            one_hot = np.zeros(len(self.tl_index), dtype=np.float32)
            one_hot[self.tl_index[tl_id]] = 1.0
            extra = np.concatenate([extra, one_hot])
        return np.concatenate([s, extra]).astype(np.float32) if len(extra) else s

    def act(self, sim, tl_id):
        if self.actions.decides(sim, tl_id) or tl_id not in self.last:
            s = self.state(sim, tl_id)
            self.last[tl_id] = self.agents[tl_id].act(s, self.actions.mask(sim, tl_id),
                                                      self.epsilon)
        return self.actions.stage(sim, tl_id, self.last[tl_id])

    def load(self, run_dir, which='best'):
        for tl_id, agent in self.agents.items():
            agent.load(weights_path(run_dir, which, tl_id))
        return self


def weights_path(run_dir, which, tl_id):
    return os.path.join(run_dir, f'{which}_{tl_id}.weights.h5')
