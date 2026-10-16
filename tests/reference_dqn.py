"""FROZEN reference copy of tsc.agents.dqn.DQNAgent as of the PyTorch port (tsc 0.3.0).

Test oracle for tests/test_dqn_parity.py: every optimisation of the real agent must
reproduce this implementation bit for bit (weights, target weights, optimizer state,
losses and RNG consumption).  Do not edit; only the thread configuration was dropped
(it is process-wide and already done by tsc.agents.dqn).  The previous oracle was the
TensorFlow agent; tests/test_tf_equivalence.py checks this one against recorded runs of it.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def dense(n_in, n_out, zero=False):
    """Linear layer initialised as Keras' Dense: Glorot-uniform kernel (or zeros), zero bias."""
    layer = torch.nn.utils.skip_init(nn.Linear, n_in, n_out)
    with torch.no_grad():
        if zero:
            layer.weight.zero_()
        else:
            nn.init.xavier_uniform_(layer.weight)
        layer.bias.zero_()
    return layer


class QNet(nn.Module):
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

    def __init__(self, n_in, n_out, hidden=(400, 400), dueling=False, ref_action=-1):
        super().__init__()
        sizes = [n_in, *hidden]
        self.hidden = nn.ModuleList(dense(a, b) for a, b in zip(sizes, sizes[1:]))
        self.head = 'residual' if ref_action >= 0 else 'dueling' if dueling else 'plain'
        if self.head == 'plain':
            self.out = dense(sizes[-1], n_out)
        else:
            self.v = dense(sizes[-1], 1)
            self.adv = dense(sizes[-1], n_out, zero=self.head == 'residual')
        if self.head == 'residual':
            keep = torch.ones(1, n_out)
            keep[0, ref_action] = 0.0            # the reference action has no advantage
            self.register_buffer('keep', keep)

    def forward(self, x):
        for i, layer in enumerate(self.hidden):
            x = layer(x) if i == 0 else F.relu(layer(x))
        if self.head == 'plain':
            return self.out(x)
        v, adv = self.v(x), self.adv(x)
        if self.head == 'dueling':
            return v + adv - adv.mean(dim=1, keepdim=True)
        return v + adv * self.keep


class PhaseQNet(nn.Module):
    """Phase-wise Q-network (FRAP, Zheng et al. 2019; MPLight, Chen et al. 2020).

    Input (n_stages, n_feats): one row per green stage (observation layout 'phase').  Every
    stage is embedded by the same MLP; its Q-value comes from its own embedding and the mean
    embedding of all stages (the competition between stages).  The weights do not depend on
    the junction, its orientation or its number of stages, so one network serves every
    junction (padded stages are masked by the agent).
    """

    def __init__(self, n_stages, n_feats, hidden=(64, 64)):
        super().__init__()
        sizes = [n_feats, *hidden]
        self.embed = nn.ModuleList(dense(a, b) for a, b in zip(sizes, sizes[1:]))
        self.mix = dense(2 * sizes[-1], sizes[-1])
        self.out = dense(sizes[-1], 1)

    def forward(self, x):
        h = x
        for layer in self.embed:
            h = F.relu(layer(h))
        ctx = h.mean(dim=1, keepdim=True).expand_as(h)
        z = F.relu(self.mix(torch.cat([h, ctx], dim=-1)))
        return self.out(z).squeeze(-1)


def build_qnet(n_in, n_out, hidden=(400, 400), dueling=False, ref_action=-1):
    return QNet(n_in, n_out, hidden, dueling, ref_action)


def build_phase_qnet(n_stages, n_feats, hidden=(64, 64)):
    return PhaseQNet(n_stages, n_feats, hidden)


class KerasAdam:
    """Adam as Keras 3 implements it, with Keras' `clipnorm`, on PyTorch's fused Adam.

    Every weight's gradient is clipped to norm `clipnorm` on its own (g * c / max(|g|, c));
    then m += (g - m)(1 - b1), v += (g^2 - v)(1 - b2) and
    w -= m * lr sqrt(1 - b2^t) / (1 - b1^t) / (sqrt(v) + eps).  torch.optim.Adam adds its eps
    to the bias-corrected sqrt(v / (1 - b2^t)) instead, so it is given
    eps / sqrt(1 - b2^t) at every step, which makes the two updates the same (up to rounding).
    clip_grad_norm_ would clip the global norm, not each weight's.
    """

    def __init__(self, params, lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7, clipnorm=None):
        self.params = list(params)
        self.beta_2, self.epsilon, self.clipnorm = beta_2, epsilon, clipnorm
        self.iterations = 0
        fast = {'fused': True} if self.params[0].device.type in ('cpu', 'cuda') else {'foreach': True}
        self.adam = torch.optim.Adam(self.params, lr=lr, betas=(beta_1, beta_2), eps=epsilon,
                                     **fast)

    def zero_grad(self):
        self.adam.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self):
        grads = [p.grad for p in self.params]
        if self.clipnorm:
            norms = torch._foreach_norm(grads)
            torch._foreach_clamp_min_(norms, self.clipnorm)
            if self.clipnorm != 1.0:
                torch._foreach_mul_(grads, self.clipnorm)
            torch._foreach_div_(grads, norms)
        self.iterations += 1
        self.adam.param_groups[0]['eps'] = (
            self.epsilon / (1.0 - self.beta_2 ** self.iterations) ** 0.5)
        self.adam.step()

    def state_arrays(self):
        out = [np.array(self.iterations)]
        for key in ('exp_avg', 'exp_avg_sq'):
            out += [self.adam.state[p][key].detach().cpu().numpy() if p in self.adam.state
                    else np.zeros(tuple(p.shape), np.float32) for p in self.params]
        return out


def keras_weights(model):
    """Parameters as Keras' get_weights(): per layer the kernel (in, out), then the bias."""
    return [np.ascontiguousarray(p.detach().cpu().numpy().T if p.ndim == 2 else p.detach().cpu().numpy())
            for p in model.parameters()]


def set_keras_weights(model, weights):
    with torch.no_grad():
        params = list(model.parameters())
        if len(params) != len(weights):
            raise ValueError(f'{len(weights)} arrays for {len(params)} parameters')
        for p, w in zip(params, weights):
            w = np.asarray(w, dtype=np.float32)
            p.copy_(torch.from_numpy(np.ascontiguousarray(w.T if p.ndim == 2 else w)))


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
                 dueling=False, n_step=1, ref_action=-1, phase_net=False, device=None):
        """n_states: input size, or (n_stages, n_feats) for the phase-wise network.
        device: torch device of the networks (default: tsc.devices.torch_device(), the
        device this process was configured for, else the CPU)."""
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
        if device is None:
            from tsc.devices import torch_device
            device = torch_device()
        self.device = torch.device(device)

        if phase_net:
            build = lambda: build_phase_qnet(*n_states, hidden=hidden)  # noqa: E731
        else:
            build = lambda: build_qnet(n_states, n_actions, hidden, dueling, ref_action)  # noqa: E731
        self.q = build().to(self.device)
        self.q_target = build().to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.requires_grad_(False)
        self._online_params = list(self.q.parameters())
        self._target_params = list(self.q_target.parameters())
        self.opt = KerasAdam(self._online_params, lr=lr, clipnorm=1.0)

    # ------------------------------------------------------------------

    def q_values(self, state):
        x = torch.from_numpy(np.asarray(state, dtype=np.float32)[None, ...]).to(self.device)
        with torch.inference_mode():
            return self.q(x)[0].cpu().numpy().copy()

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
        loss = self._train_step(self._tensor(s, np.float32), self._tensor(a, np.int64),
                                self._tensor(r, np.float32), self._tensor(s2, np.float32),
                                self._tensor(m2, bool), self._tensor(d, np.float32))
        self.grad_steps += 1
        if self.target_update == 'soft':
            self._polyak()
        elif self.grad_steps % self.hard_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss)

    # ------------------------------------------------------------------

    def _tensor(self, x, dtype):
        return torch.from_numpy(np.asarray(x, dtype=dtype)).to(self.device)

    def _train_step(self, s, a, r, s2, m2, d):
        """One Double/plain DQN gradient step."""
        with torch.no_grad():
            neg_inf = torch.full(m2.shape, -1e9, device=self.device)
            q_next_target = self.q_target(s2)
            if self.double:
                q_next_online = torch.where(m2, self.q(s2), neg_inf)
                best = torch.argmax(q_next_online, dim=1)
                next_v = q_next_target.gather(1, best[:, None]).squeeze(1)
            else:
                next_v = torch.where(m2, q_next_target, neg_inf).max(dim=1).values
            target = r + d * next_v
        q_sa = self.q(s).gather(1, a[:, None]).squeeze(1)
        loss = F.huber_loss(q_sa, target, delta=1.0)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.detach()

    @torch.no_grad()
    def _polyak(self):
        """target <- tau * online + (1 - tau) * target, one fused op over all weights."""
        torch._foreach_lerp_(self._target_params, self._online_params, self.tau)

    # ------------------------------------------------------------------

    def get_weights(self):
        """Online weights as numpy arrays in Keras' layout (kernel (in, out), bias)."""
        return keras_weights(self.q)

    def set_weights(self, w):
        set_keras_weights(self.q, w)
        self.q_target.load_state_dict(self.q.state_dict())

    def state_arrays(self):
        """Everything training updates, as numpy arrays (parity fingerprints)."""
        return {'online': keras_weights(self.q), 'target': keras_weights(self.q_target),
                'optimizer': self.opt.state_arrays()}

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.q_target.load_state_dict(self.q.state_dict())
