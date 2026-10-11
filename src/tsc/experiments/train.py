"""Train RL signal controllers on randomised, realistic demand.

Example:
    python train.py --config full --seed 0 --episodes 150 --out runs/train/full_s0

Configs (ablation of the paper's extensions):
    base        : DQN, hard target update, 'wait' reward, no communication (paper's EA)
    full        : Double DQN, soft update, 'multi' reward, communication     (paper's VA)
    full_nocomm : full without communication
    queue       : full, but reward = -queue
    det_queue   : full, but reward = -jam length seen by detectors (step 2, realistic)
    det_queue_nocomm : det_queue without communication

Step 2 (realistic sensing + safety): --obs detector --noise 0.05 --dropout 0.02
--max-green 60 --max-red 90
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np

from scenario import Scenario
from sim import Simulation, run_controller
from observation import Observer, Reward
from rl_agent import DQNAgent

CONFIGS = {
    'base':        dict(comm=False, double=False, target_update='hard', reward='wait'),
    'full':        dict(comm=True,  double=True,  target_update='soft', reward='multi'),
    'full_nocomm': dict(comm=False, double=True,  target_update='soft', reward='multi'),
    'queue':       dict(comm=True,  double=True,  target_update='soft', reward='queue'),
    'det_queue':   dict(comm=True,  double=True,  target_update='soft', reward='det_queue'),
    'det_queue_nocomm': dict(comm=False, double=True, target_update='soft', reward='det_queue'),
}

# Route seeds: training uses seed*100000 + episode, validation 700000+, evaluation 900000+
VALIDATION = [(1200, 700001), (2400, 700002)]


class RLController:
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


def make_agents(scn, observer, cfg):
    return {
        tl_id: DQNAgent(observer.dim, scn.max_greens, double=cfg['double'],
                        target_update=cfg['target_update'])
        for tl_id in scn.tl_ids
    }


def load_run(run_dir, scn, which='best'):
    with open(os.path.join(run_dir, 'config.json')) as f:
        meta = json.load(f)
    cfg = CONFIGS[meta['config']]
    observer = Observer(scn, mode=meta.get('obs', 'grid'), comm=cfg['comm'],
                        noise=meta.get('noise', 0.0), dropout=meta.get('dropout', 0.0),
                        seed=meta['seed'] + 12345)
    agents = make_agents(scn, observer, cfg)
    for tl_id, agent in agents.items():
        agent.load(os.path.join(run_dir, f'{which}_{tl_id}.weights.h5'))
    return RLController(agents, observer), meta


def shield_kwargs(meta):
    return {'max_green': meta.get('max_green'), 'max_red': meta.get('max_red')}


def validate(scn, controller, duration, work_dir, profile, shield):
    delays = []
    for level, seed in VALIDATION:
        route = scn.generate_routes(os.path.join(work_dir, 'val.rou.xml'), seed, level, duration, profile)
        m = run_controller(scn, controller, route, duration, os.path.join(work_dir, 'val'),
                           seed=seed, **shield)
        delays.append(m['delay'])
    return float(np.mean(delays))


def train(args):
    cfg = CONFIGS[args.config]
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)
    random.seed(args.seed)
    import tensorflow as tf
    tf.random.set_seed(args.seed)

    scn = Scenario(args.net, os.path.join('runs', 'scenario'))
    observer = Observer(scn, mode=args.obs, comm=cfg['comm'], noise=args.noise,
                        dropout=args.dropout, seed=args.seed)
    shield = shield_kwargs(vars(args))
    reward_fn = Reward(scn, cfg['reward'])
    agents = make_agents(scn, observer, cfg)
    greedy = RLController(agents, observer)
    rng = np.random.default_rng(args.seed)

    with open(os.path.join(args.out, 'config.json'), 'w') as f:
        json.dump({**vars(args), **cfg}, f, indent=2)

    log_path = os.path.join(args.out, 'train_log.csv')
    log = None
    best_val = np.inf
    eps_end_ep = max(1, int(args.episodes * args.eps_fraction))

    for ep in range(args.episodes):
        t0 = time.time()
        epsilon = max(args.eps_min, 1.0 - (1.0 - args.eps_min) * ep / eps_end_ep)
        level = float(rng.uniform(args.level_min, args.level_max))
        route_seed = args.seed * 100000 + ep
        route = scn.generate_routes(os.path.join(args.out, 'train.rou.xml'),
                                    route_seed, level, args.duration, args.profile)

        sim = Simulation(scn, route, args.duration, os.path.join(args.out, 'episode'),
                         seed=route_seed, **shield).start()
        reward_fn.reset()
        observer.reset(sim)
        pending = {}
        acc = {tl_id: [0.0, 0] for tl_id in scn.tl_ids}
        rewards, losses = [], []

        while not sim.done:
            for tl_id in sim.decision_points():
                agent = agents[tl_id]
                s = observer.observe(sim, tl_id)
                mask = observer.action_mask(tl_id)
                if tl_id in pending:
                    ps, pa = pending[tl_id]
                    r = acc[tl_id][0] / max(1, acc[tl_id][1])
                    agent.store(ps, pa, r, s, mask)
                    rewards.append(r)
                    loss = agent.replay()
                    if loss is not None:
                        losses.append(loss)
                a = agent.act(s, mask, epsilon)
                sim.heads[tl_id].request(a)
                pending[tl_id] = (s, a)
                acc[tl_id] = [0.0, 0]
            sim.step()
            observer.tick(sim)
            for tl_id in scn.tl_ids:
                acc[tl_id][0] += reward_fn(sim, tl_id)
                acc[tl_id][1] += 1

        m = sim.close()
        row = {'episode': ep, 'level': round(level, 1), 'epsilon': round(epsilon, 3),
               'mean_reward': float(np.mean(rewards)) if rewards else 0.0,
               'mean_loss': float(np.mean(losses)) if losses else float('nan'), **m}

        if (ep + 1) % args.val_every == 0 or ep == args.episodes - 1:
            val = validate(scn, greedy, args.duration, args.out, args.profile, shield)
            row['val_delay'] = val
            if val < best_val:
                best_val = val
                for tl_id, agent in agents.items():
                    agent.save(os.path.join(args.out, f'best_{tl_id}.weights.h5'))
        row['seconds'] = round(time.time() - t0, 1)

        if log is None:
            fields = list(row.keys()) + (['val_delay'] if 'val_delay' not in row else [])
            log_file = open(log_path, 'w', newline='')
            log = csv.DictWriter(log_file, fieldnames=fields, extrasaction='ignore')
            log.writeheader()
        log.writerow(row)
        log_file.flush()
        print(f"[{args.config} s{args.seed}] ep {ep:3d} level {level:6.0f} eps {epsilon:.2f} "
              f"delay {m['delay']:7.1f} thr {m['throughput']:.2f} "
              f"{'val %.1f ' % row['val_delay'] if 'val_delay' in row else ''}({row['seconds']}s)",
              flush=True)

    for tl_id, agent in agents.items():
        agent.save(os.path.join(args.out, f'final_{tl_id}.weights.h5'))
    log_file.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', choices=CONFIGS, default='full')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--episodes', type=int, default=150)
    p.add_argument('--duration', type=int, default=3600)
    p.add_argument('--level-min', type=float, default=600)
    p.add_argument('--level-max', type=float, default=3000)
    p.add_argument('--profile', default='peak')
    p.add_argument('--obs', default='grid')
    p.add_argument('--noise', type=float, default=0.0, help='detector noise (std, normalised units)')
    p.add_argument('--dropout', type=float, default=0.0, help='probability a detector reading fails')
    p.add_argument('--max-green', type=int, default=None)
    p.add_argument('--max-red', type=int, default=None)
    p.add_argument('--eps-min', type=float, default=0.05)
    p.add_argument('--eps-fraction', type=float, default=0.6)
    p.add_argument('--val-every', type=int, default=10)
    p.add_argument('--net', default='intersection/test.net.xml')
    p.add_argument('--out', required=True)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
