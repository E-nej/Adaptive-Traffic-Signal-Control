"""Hierarchical control (step 2H): a manager chooses the experts, junctions run them.

Top level (Manager): every `interval` s (default 5 min) it picks one expert
(green wave, coordinated-actuated, max-pressure, actuated; [signals] experts) for
each *group* of junctions:

    scope = 'network'  : one expert for the whole network
    scope = 'group'    : one expert per group of signals linked by roads of at most
                         `group_range` m (e.g. a coordinated arterial vs isolated
                         junctions, network.groups)
    scope = 'junction' : one expert per junction

All groups share one Q-network.  Its input does not depend on the network's size or
layout (aggregates of the group's detectors, the same for the whole network, the
group's current expert, its share of the signals), so a manager trained on one
network can run on another.  Its reward is minus the mean queue per lane in the group
over the interval.

Bottom level: the junctions of a group run the chosen expert (SelectorActions, so a
change of expert is safe: SignalHead intergreens, a plan expert rejoins the plan).
worker = 'off' runs the experts unmodified; per-junction RL that fine-tunes the active
expert comes on top of this.
"""

import numpy as np

from tsc.controllers.select import SelectorActions
from tsc.detectors import e1_id, e2_capacity, e2_id

SCOPES = ('network', 'group', 'junction')
GROUP_FEATS = 5            # jam ratio, occupancy, speed ratio, flow, jam now
SAT_FLOW = 0.5             # veh/s per lane (1800 veh/h): flow normalisation


def make_groups(network, scope, group_range):
    if scope == 'network':
        return [list(network.tl_ids)]
    if scope == 'group':
        return network.groups(group_range)
    if scope == 'junction':
        return [[t] for t in network.tl_ids]
    raise ValueError(f'Unknown hierarchy scope {scope!r}; choose from {SCOPES}')


class GroupSensor:
    """Detector aggregates of every group over the manager's interval.

    E2 area detectors are read every `sample_every` s, the stop-line loops every second.
    Readings are noise-free: summed over many detectors, per-detector noise averages out.
    """

    def __init__(self, network, groups, e2_length=50.0, sample_every=5):
        self.network = network
        self.groups = groups
        self.sample_every = sample_every
        self.lanes = [[l for t in g for l in network.in_lanes[t]] for g in groups]
        self.all_lanes = [l for ls in self.lanes for l in ls]
        self.cap = {l: e2_capacity(network, l, e2_length) for l in self.all_lanes}

    def reset(self):
        n = len(self.groups)
        self.sums = np.zeros((n + 1, 3))    # jam, occupancy, speed (+ whole network)
        self.flow = np.zeros(n + 1)
        self.now = np.zeros(n + 1)
        self.samples = 0
        self.seconds = 0

    def tick(self, sim):
        api = sim.api
        passed = {l: api.inductionloop.getLastStepVehicleNumber(e1_id(l)) for l in self.all_lanes}
        for i, lanes in enumerate(self.lanes + [self.all_lanes]):
            self.flow[i] += sum(passed[l] for l in lanes) / len(lanes)
        self.seconds += 1
        if sim.t % self.sample_every:
            return
        for i, lanes in enumerate(self.lanes + [self.all_lanes]):
            jam = sum(api.lanearea.getJamLengthVehicle(e2_id(l)) for l in lanes)
            jam /= sum(self.cap[l] for l in lanes)
            occ = np.mean([api.lanearea.getLastStepOccupancy(e2_id(l)) for l in lanes]) / 100.0
            speeds = [api.lanearea.getLastStepMeanSpeed(e2_id(l)) / self.network.lane_speed[l]
                      for l in lanes]
            speed = np.mean([1.0 if v < 0 else min(1.0, v) for v in speeds])
            self.sums[i] += (jam, occ, speed)
            self.now[i] = jam
        self.samples += 1

    def features(self):
        """(n_groups + 1, GROUP_FEATS): [mean jam, occupancy, speed, flow, jam now]; the
        last row is the whole network.  Restarts the averaging."""
        mean = self.sums / max(1, self.samples)
        if not self.samples:
            mean[:, 2] = 1.0                    # empty roads: free speed
        flow = self.flow / max(1, self.seconds) / SAT_FLOW
        out = np.column_stack([mean, flow, self.now]).astype(np.float32)
        self.reset()
        return out


class Manager:
    """Top-level policy: one expert per group every `interval` s.

    policy = 'learned' : DQN over the group features (shared by all groups)
    policy = 'fixed'   : always expert `fixed_expert` (a uniform-expert baseline that runs
                         through exactly the same machinery)
    """

    def __init__(self, network, groups, n_experts, interval=300, policy='learned',
                 fixed_expert=0, reward='queue', reward_scale=1.0, agent=None, updates=20,
                 e2_length=50.0):
        if policy not in ('learned', 'fixed'):
            raise ValueError(f'Unknown manager policy {policy!r}')
        self.network = network
        self.groups = groups
        self.n_experts = n_experts
        self.interval = interval
        self.policy = policy
        self.fixed_expert = fixed_expert
        self.reward_scale = reward_scale
        self.updates = updates
        self.agent = agent
        self.sensor = GroupSensor(network, groups, e2_length)
        from tsc.reward import Reward
        self.reward_fn = Reward(network, reward)
        self.size = np.array([len(g) / len(network.tl_ids) for g in groups], dtype=np.float32)
        self.dim = 2 * GROUP_FEATS + n_experts + 1
        self.learning = False
        self.epsilon = 0.0

    def reset(self, sim):
        self.sensor.reset()
        self.reward_fn.reset()
        self.acc = np.zeros(len(self.groups))
        self.acc_n = 0
        self.choice = [self.fixed_expert if self.policy == 'fixed' else 0] * len(self.groups)
        self.pending = None
        self.counts = np.zeros(self.n_experts)
        self._decide()

    def tick(self, sim):
        self.sensor.tick(sim)
        r = self.reward_fn.all(sim)
        self.acc += [np.mean([r[t] for t in g]) for g in self.groups]
        self.acc_n += 1
        if sim.t % self.interval == 0:
            self._decide()

    def states(self):
        feats = self.sensor.features()
        net = feats[-1]
        out = []
        for i, choice in enumerate(self.choice):
            own = np.zeros(self.n_experts, dtype=np.float32)
            own[choice] = 1.0
            out.append(np.concatenate([feats[i], net, own, self.size[i:i + 1]]))
        return out

    def _decide(self):
        states = self.states()
        if self.policy == 'learned':
            mask = np.ones(self.n_experts, dtype=bool)
            if self.learning and self.pending is not None:
                rewards = self.reward_scale * self.acc / max(1, self.acc_n)
                for (s, a), r, s2 in zip(self.pending, rewards, states):
                    self.agent.store(s, a, float(r), s2, mask)
                for _ in range(self.updates):
                    self.agent.replay()
            self.choice = [self.agent.act(s, mask, self.epsilon) for s in states]
            self.pending = list(zip(states, self.choice))
        self.acc[:] = 0.0
        self.acc_n = 0
        for g, c in zip(self.groups, self.choice):
            self.counts[c] += len(g)

    def expert_of(self):
        """{tl_id: expert index} of the current choice."""
        return {t: c for g, c in zip(self.groups, self.choice) for t in g}

    def shares(self):
        """Share of junction-decisions per expert in this episode."""
        total = self.counts.sum()
        return self.counts / total if total else self.counts


class HierarchyActions:
    """Action space of the bottom level under a manager ([signals] control = 'hierarchy').

    With worker = 'off' the junctions have no RL agent of their own: every junction runs the
    expert the manager chose for its group (has_workers = False).
    """

    needs_plan = True
    n_actions = 1
    extra_dim = 0

    def __init__(self, network, cfg, seed=0):
        h, s, o = cfg.hierarchy, cfg.signals, cfg.observation
        if h.worker != 'off':
            raise ValueError(f'Unknown worker mode {h.worker!r}')
        self.network = network
        self.has_workers = False
        self.selector = SelectorActions(network, interval=s.decision_interval,
                                        max_extension=s.max_extension, noise=o.noise,
                                        dropout=o.dropout, e2_length=o.e2_length, comm=False,
                                        seed=seed, experts=s.experts,
                                        actuated_loops=s.actuated_loops)
        self.names = self.selector.names
        groups = make_groups(network, h.scope, h.group_range)
        agent = None
        if h.manager == 'learned':
            from tsc.agents.dqn import DQNAgent
            dim = 2 * GROUP_FEATS + len(self.names) + 1
            agent = DQNAgent(dim, len(self.names), double=True, target_update='soft',
                             tau=h.tau, gamma=h.gamma, lr=h.lr, batch_size=h.batch_size,
                             buffer_size=h.buffer_size, learn_start=h.batch_size,
                             hidden=tuple(h.hidden))
        self.manager = Manager(network, groups, len(self.names), interval=h.interval,
                               policy=h.manager, fixed_expert=h.fixed_expert, reward=h.reward,
                               reward_scale=h.reward_scale, agent=agent, updates=h.updates,
                               e2_length=o.e2_length)
        self.group_of = {t: i for i, g in enumerate(groups) for t in g}

    # training hooks ---------------------------------------------------

    def set_training(self, learning, epsilon):
        self.manager.learning = learning
        self.manager.epsilon = epsilon

    def learners(self):
        """Extra networks to save / load with the run: {name: agent}."""
        return {'manager': self.manager.agent} if self.manager.agent is not None else {}

    def episode_stats(self):
        return {f'mgr_{n}': round(float(x), 3) for n, x in zip(self.names, self.manager.shares())}

    # Controller-like interface ----------------------------------------

    def set_plans(self, plans):
        self.selector.set_plans(plans)

    def reseed(self, seed):
        self.selector.reseed(seed)

    def reset(self, sim):
        self.selector.reset(sim)
        self.manager.reset(sim)

    def tick(self, sim):
        self.selector.tick(sim)
        self.manager.tick(sim)

    def decides(self, sim, tl_id):
        return False

    def mask(self, sim, tl_id):
        return np.ones(1, dtype=bool)

    def stage(self, sim, tl_id, action=0):
        expert = self.manager.choice[self.group_of[tl_id]]
        return self.selector.stage(sim, tl_id, expert)

    def features(self, sim, tl_id):
        return np.zeros(0, dtype=np.float32)


__all__ = ['HierarchyActions', 'Manager', 'SCOPES', 'make_groups']
