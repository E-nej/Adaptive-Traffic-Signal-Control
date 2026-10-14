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

Bottom level: the junctions of a group run the chosen expert (ExpertPool, so a
change of expert is safe: SignalHead intergreens, a plan expert rejoins the plan).
worker = 'off' runs the experts unmodified; worker = 'residual' adds a per-junction agent
that fine-tunes the active expert (HierarchyActions).  Both levels train in the same run.
"""

import numpy as np

from tsc.controllers.plan import EARLY, FOLLOW, HOLD, PlanActions
from tsc.controllers.select import ExpertPool
from tsc.detectors import e1_id, e2_capacity, e2_id

SCOPES = ('network', 'group', 'junction')
GROUP_FEATS = 5            # jam ratio, occupancy, speed ratio, flow, jam now
SAT_FLOW = 0.5             # veh/s per lane (1800 veh/h): flow normalisation
WORKERS = ('off', 'residual', 'expert')
AGREE, HOLD_W, SWITCH = 0, 1, 2        # worker actions
N_WORKER_ACTIONS = 3


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
    policy = 'table'   : the expert that was best at the demand level whose network flow
                         and jam ratio are nearest to those measured over the last interval
                         (jam tells levels apart past saturation, where flow stops rising).
                         `table`: [(keys, expert)], keys = [(flow, jam)] per interval of the
                         episode, so the k-th reading is compared with the levels' k-th
                         readings (the network fills up from empty, the demand profile
                         varies within the hour); built by running every expert at a grid
                         of levels (experiments.train.build_table).  Same expert for every
                         group; a change needs `hysteresis` decisions in a row, except at
                         the first decision with a measured flow.
    """

    def __init__(self, network, groups, n_experts, interval=300, policy='learned',
                 fixed_expert=0, reward='queue', reward_scale=1.0, agent=None, updates=20,
                 e2_length=50.0):
        if policy not in ('learned', 'fixed', 'table'):
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
        self.default = None     # expert run unless another one is better by `margin`
        self.margin = 0.0
        self.forced = None      # expert run everywhere, whatever the Q-values (fallback)
        self.table = []         # policy 'table': [([(flow, jam) per interval], expert,
                                #                   worker on)]
        self.hysteresis = 2
        self.worker_on = True   # table: the active expert's worker acts

    def reset(self, sim):
        self.sensor.reset()
        self.reward_fn.reset()
        self.acc = np.zeros(len(self.groups))
        self.acc_n = 0
        if self.forced is not None:
            first = self.forced
        elif self.policy == 'fixed':
            first = self.fixed_expert
        else:
            first = self.default if self.default is not None else 0
        self.choice = [first] * len(self.groups)
        self.pending = None
        self.counts = np.zeros(self.n_experts)
        self.flows = []                 # network (flow, jam) of every interval (table building)
        self.candidate, self.streak = None, 0
        self.worker_on = True           # table: the active expert's worker acts
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
        measured = self.acc_n > 0               # not the decision at t = 0
        states = self.states()
        net = states[0][GROUP_FEATS:2 * GROUP_FEATS]    # network row: [jam, .., flow, ..]
        flow = (float(net[3]), float(net[0]))
        if measured:
            self.flows.append(flow)
        if self.forced is not None:
            self.choice = [self.forced] * len(self.groups)
            self.pending = None
        elif self.policy == 'table':
            if measured:
                self._look_up(flow)
        elif self.policy == 'learned':
            mask = np.ones(self.n_experts, dtype=bool)
            if self.learning and self.pending is not None:
                rewards = self.reward_scale * self.acc / max(1, self.acc_n)
                for (s, a), r, s2 in zip(self.pending, rewards, states):
                    self.agent.store(s, a, float(r), s2, mask)
                for _ in range(self.updates):
                    self.agent.replay()
            self.choice = [self._act(s, mask) for s in states]
            self.pending = list(zip(states, self.choice))
        self.acc[:] = 0.0
        self.acc_n = 0
        for g, c in zip(self.groups, self.choice):
            self.counts[c] += len(g)

    def _look_up(self, flow):
        i = len(self.flows) - 1                   # this reading's index in the episode
        keys = [np.atleast_2d(entry[0]) for entry in self.table]   # one (flow, jam): every interval
        keys = np.array([k[min(i, len(k) - 1)] for k in keys])
        scale = np.maximum(keys.max(axis=0) - keys.min(axis=0), 1e-6)
        dist = np.abs((keys - np.asarray(flow)) / scale).sum(axis=1)
        _, expert, worker = self.table[int(np.argmin(dist))]
        first = self.candidate is None
        self.streak = self.streak + 1 if (expert, worker) == self.candidate else 1
        self.candidate = (expert, worker)
        if first or self.streak >= self.hysteresis:
            self.choice = [expert] * len(self.groups)
            self.worker_on = worker

    def _act(self, state, mask):
        """Epsilon-greedy; with a default expert the greedy choice leaves the default only
        if its Q-value is higher by margin·|Q(default)| (switching has costs Q misses)."""
        if self.default is None:
            return self.agent.act(state, mask, self.epsilon)
        if self.epsilon and np.random.random() < self.epsilon:
            return self.agent.act(state, mask, 1.0)
        q = self.agent.q_values(state)
        best, d = int(np.argmax(q)), self.default
        return best if q[best] - q[d] > self.margin * abs(q[d]) else d

    def expert_of(self):
        """{tl_id: expert index} of the current choice."""
        return {t: c for g, c in zip(self.groups, self.choice) for t in g}

    def shares(self):
        """Share of junction-decisions per expert in this episode."""
        total = self.counts.sum()
        return self.counts / total if total else self.counts


class HierarchyActions:
    """Action space of the bottom level under a manager ([signals] control = 'hierarchy').

    worker = 'off'      : no junction agents (has_workers = False); every junction runs the
                          expert the manager chose for its group.
    worker = 'residual' : one agent per junction (or shared) fine-tunes the active expert.
                          Every `decision_interval` s of green after min green it picks
                              AGREE  : do what the expert does (the expert keeps control
                                       second by second until the next decision)
                              HOLD   : keep the green although the expert wants to switch
                                       (at most max_extension s past the expert per green)
                              SWITCH : end the green now although the expert holds it
                          On a plan expert HOLD / SWITCH are the plan's HOLD / EARLY, so the
                          plan's timing stays the reference.  Input: the junction's
                          observation + the active expert, whether it wants to switch, the
                          extension used and the plan features (zeros without a plan).
    worker = 'expert'   : as 'residual', but each expert has its own worker, trained with
                          that expert always active ([hierarchy] worker_runs: a run with
                          manager = 'fixed' per expert).  A junction uses the worker of the
                          expert active there; experts without one (or whose worker did not
                          pass acceptance) run alone.  Nothing is trained in this mode.
    """

    needs_plan = True

    def __init__(self, network, cfg, seed=0):
        h, s, o = cfg.hierarchy, cfg.signals, cfg.observation
        if h.worker not in WORKERS:
            raise ValueError(f'Unknown worker mode {h.worker!r}; choose from {WORKERS}')
        self.network = network
        self.has_workers = h.worker in ('residual', 'expert')
        self.worker_mode = h.worker
        self.interval = s.decision_interval
        self.max_extension = s.max_extension
        self.selector = ExpertPool(network, interval=s.decision_interval,
                                   max_extension=s.max_extension, noise=o.noise,
                                   dropout=o.dropout, e2_length=o.e2_length, seed=seed,
                                   experts=s.experts, actuated_loops=s.actuated_loops,
                                   hold_guard=s.hold_guard, rl_runs=s.rl_experts)
        self.names = self.selector.names
        self.n_actions = N_WORKER_ACTIONS if self.has_workers else 1
        self.extra_dim = len(self.names) + 2 + PlanActions.extra_dim if self.has_workers else 0
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
        self.manager.margin = h.margin
        self.manager.hysteresis = h.hysteresis
        if h.manager == 'learned' and h.default_expert not in ('', 'auto'):
            self.set_default(self.names.index(h.default_expert))
        self.group_of = {t: i for i, g in enumerate(groups) for t in g}
        self.force_agree = False    # workers always AGREE (single-expert validation, fallback)
        self.expert_workers = {}    # worker = 'expert': expert index -> {tl_id: agent}
        if h.worker == 'expert':
            self._load_expert_workers(network, h.worker_runs, seed)

    def _load_expert_workers(self, network, runs, seed):
        from tsc.experiments.runs import load_policy
        for name, run_dir in runs.items():
            run_dir = run_dir.format(seed=seed)
            policy, run_cfg = load_policy(run_dir, network, 'best')
            rh = run_cfg.hierarchy
            trained_for = run_cfg.signals.experts[rh.fixed_expert]
            if rh.manager != 'fixed' or rh.worker != 'residual' or trained_for != name:
                raise ValueError(f'{run_dir}: not a worker run for expert {name!r} '
                                 f'(manager {rh.manager!r}, worker {rh.worker!r}, '
                                 f'expert {trained_for!r})')
            if not policy.actions.force_agree:          # else: its expert alone was better
                self.expert_workers[self.names.index(name)] = policy.agents

    def make_agents(self, dim):
        """worker = 'expert': per junction, the worker of whichever expert is active (None
        otherwise: the usual trainable agents)."""
        if self.worker_mode != 'expert':
            return None
        for agents in self.expert_workers.values():
            n = next(iter(agents.values())).n_states
            if n != dim:
                raise ValueError(f'worker input size {n} != {dim}: the worker runs need the '
                                 f'same observation, experts and agent sharing as this run')
        return {t: ExpertWorker(self, t) for t in self.network.tl_ids}

    def set_default(self, expert):
        self.manager.default = expert

    def force(self, expert, agree=True):
        """Run `expert` everywhere (workers agreeing); None = back to the learned policy."""
        self.manager.forced = expert
        self.force_agree = agree and expert is not None

    # saved with the weights: which expert the manager defaults to / is forced to
    def save_meta(self, run_dir, which):
        import json
        import os
        m = self.manager
        meta = {'default': None if m.default is None else self.names[m.default],
                'forced': None if m.forced is None else self.names[m.forced],
                'agree': self.force_agree,
                'table': [[np.asarray(k).tolist(), self.names[e], w] for k, e, w in m.table]}
        with open(os.path.join(run_dir, f'{which}_hierarchy.json'), 'w') as f:
            json.dump(meta, f)

    def load_meta(self, run_dir, which):
        import json
        import os
        path = os.path.join(run_dir, f'{which}_hierarchy.json')
        if not os.path.exists(path):        # runs from before the default expert
            return
        with open(path) as f:
            meta = json.load(f)
        if meta['default'] is not None:
            self.set_default(self.names.index(meta['default']))
        self.manager.forced = None if meta['forced'] is None else self.names.index(meta['forced'])
        self.force_agree = meta['agree']
        self.manager.table = [(np.asarray(t[0]), self.names.index(t[1]), t[2] if len(t) > 2 else True)
                              for t in meta.get('table', [])]

    # training hooks ---------------------------------------------------

    def set_training(self, learning, epsilon):
        self.manager.learning = learning
        self.manager.epsilon = epsilon

    def learners(self):
        """Extra networks to save / load with the run: {name: agent}."""
        return {'manager': self.manager.agent} if self.manager.agent is not None else {}

    def episode_stats(self):
        stats = {f'mgr_{n}': round(float(x), 3) for n, x in zip(self.names, self.manager.shares())}
        if self.has_workers:
            total = max(1, self.overrides.sum() + self.agrees)
            stats['wrk_hold'] = round(float(self.overrides[0] / total), 3)
            stats['wrk_switch'] = round(float(self.overrides[1] / total), 3)
        return stats

    # Controller-like interface ----------------------------------------

    def set_plans(self, plans):
        self.selector.set_plans(plans)

    def reseed(self, seed):
        self.selector.reseed(seed)

    def reset(self, sim):
        self.selector.reset(sim)
        self.manager.reset(sim)
        self.last = {t: AGREE for t in self.network.tl_ids}     # worker action of this green
        self.held = {t: 0 for t in self.network.tl_ids}         # s held past the expert
        self.green = {t: None for t in self.network.tl_ids}     # (stage, start) of the green
        self.cache = {}                                         # tl -> (t, proposal)
        self.overrides = np.zeros(2)
        self.agrees = 0

    def tick(self, sim):
        self.selector.tick(sim)
        self.manager.tick(sim)

    def expert(self, tl_id):
        return self.manager.choice[self.group_of[tl_id]]

    def decides(self, sim, tl_id):
        if not self.has_workers:
            return False
        head = sim.heads[tl_id]
        return (head.timer - head.min_green) % self.interval == 0

    def _new_green(self, sim, tl_id):
        """Reset the worker's per-green state when a new green has started."""
        head = sim.heads[tl_id]
        key = (head.stage, sim.t - head.timer)
        if self.green[tl_id] != key:
            self.green[tl_id] = key
            self.last[tl_id] = AGREE
            self.held[tl_id] = 0

    def _proposal(self, sim, tl_id):
        """What the active expert does now: a stage (other experts) or a plan action
        (plan experts), computed once per second."""
        hit = self.cache.get(tl_id)
        if hit is not None and hit[0] == sim.t:
            return hit[1]
        e = self.expert(tl_id)
        self.selector.activate(sim, tl_id, e)
        if e in self.selector.plan_experts:
            out = self.selector.experts[e].policy(sim, tl_id)
        else:
            out = self.selector.propose(sim, tl_id)
        self.cache[tl_id] = (sim.t, out)
        return out

    def _wants_switch(self, sim, tl_id):
        e, p = self.expert(tl_id), self._proposal(sim, tl_id)
        if e in self.selector.plan_experts:
            acts = self.selector.experts[e].actions
            return p == EARLY or (p == FOLLOW and acts.count[tl_id] < acts.plans[tl_id].counter(sim.t))
        return p != sim.heads[tl_id].stage

    def mask(self, sim, tl_id):
        if not self.has_workers:
            return np.ones(1, dtype=bool)
        self._new_green(sim, tl_id)
        e, p = self.expert(tl_id), self._proposal(sim, tl_id)
        mask = np.zeros(N_WORKER_ACTIONS, dtype=bool)
        mask[AGREE] = True
        if self.force_agree:
            return mask
        if e in self.selector.plan_experts:
            plan_mask = self.selector.experts[e].actions.mask(sim, tl_id)
            mask[HOLD_W] = plan_mask[HOLD] and p != HOLD
            mask[SWITCH] = plan_mask[EARLY] and p != EARLY
        else:
            wants = self._wants_switch(sim, tl_id)
            mask[HOLD_W] = wants and self.held[tl_id] < self.max_extension
            mask[SWITCH] = not wants and len(self.network.greens[tl_id]) > 1
        return mask

    def stage(self, sim, tl_id, action=AGREE):
        if not self.has_workers:
            return self.selector.stage(sim, tl_id, self.expert(tl_id))
        self._new_green(sim, tl_id)
        e, p = self.expert(tl_id), self._proposal(sim, tl_id)
        head = sim.heads[tl_id]
        deciding = self.decides(sim, tl_id)
        if deciding:
            action = action if self.mask(sim, tl_id)[action] else AGREE
            self.last[tl_id] = action
            if action == AGREE:
                self.agrees += 1
            else:
                self.overrides[action - 1] += 1
        action = self.last[tl_id]
        if e in self.selector.plan_experts:
            acts = self.selector.experts[e].actions
            if deciding:
                choice = {AGREE: p, HOLD_W: HOLD, SWITCH: EARLY}[action]
                acts.last[tl_id] = choice if acts.mask(sim, tl_id)[choice] else FOLLOW
            return acts.stage(sim, tl_id, acts.last[tl_id])
        if action == HOLD_W and p != head.stage and self.held[tl_id] < self.max_extension:
            self.held[tl_id] += 1
            return head.stage
        if action == SWITCH and deciding and p == head.stage:
            return (head.stage + 1) % len(self.network.greens[tl_id])
        return p

    def features(self, sim, tl_id):
        if not self.has_workers:
            return np.zeros(0, dtype=np.float32)
        self._new_green(sim, tl_id)
        e = self.expert(tl_id)
        own = np.zeros(len(self.names), dtype=np.float32)
        own[e] = 1.0
        plan = np.zeros(PlanActions.extra_dim, dtype=np.float32)
        if e in self.selector.plan_experts:
            plan = self.selector.experts[e].actions.features(sim, tl_id)
        wants = float(self._wants_switch(sim, tl_id))
        held = self.held[tl_id] / max(1, self.max_extension)
        return np.concatenate([own, [wants, held], plan]).astype(np.float32)


class ExpertWorker:
    """Junction agent for worker = 'expert': acts with the worker of the active expert
    (AGREE if it has none).  Not trainable; the workers come from their own runs."""

    def __init__(self, actions, tl_id):
        self.actions, self.tl_id = actions, tl_id

    def act(self, state, mask, epsilon=0.0):
        agents = self.actions.expert_workers.get(self.actions.expert(self.tl_id))
        if agents is None or self.actions.force_agree or not self.actions.manager.worker_on:
            return AGREE
        return agents[self.tl_id].act(state, mask, 0.0)


__all__ = ['ExpertWorker', 'HierarchyActions', 'Manager', 'SCOPES', 'make_groups']
