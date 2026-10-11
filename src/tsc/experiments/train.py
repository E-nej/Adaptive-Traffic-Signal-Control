"""Train RL signal controllers on randomised, realistic demand.

    tsc train configs/step1/full.toml --seeds 0-4

Every episode samples a demand level uniformly from [level_min, level_max] and
generates a fresh scenario (seed 100000·seed + episode).  Every `val_every`
episodes the greedy policy is validated on fixed scenarios (700001+) and the
best one is kept (`best_<tl>.weights.h5`); the last one is saved as `final_*`.

Output per training seed: <out>/s<seed>/ with config.toml, run_info.json,
train_log.csv and the weights.
"""

import csv
import os
import random
import time

import numpy as np

from tsc import config as config_mod
from tsc import seeds as seeds_mod
from tsc.demand import generate_routes
from tsc.experiments.common import network_for, run_parallel, save_run_info, sim_kwargs
from tsc.experiments.runs import make_observer
from tsc.reward import Reward
from tsc.simulation import Simulation, run_episode


def default_out(cfg):
    return os.path.join(cfg.output_dir, 'train', cfg.name)


def train(cfg, seeds=None, out_root=None, workers=None):
    """Train one run per seed (in parallel); returns the run directories."""
    seeds = list(seeds if seeds is not None else cfg.train.seeds)
    for s in seeds:
        seeds_mod.check_train_seed(s)
    out_root = out_root or default_out(cfg)
    network_for(cfg)            # build the network once before the workers start
    tasks = [(config_mod.replace(cfg, train={'seeds': [s]}), os.path.join(out_root, f's{s}'))
             for s in seeds]
    return list(run_parallel(_train_task, tasks, workers or cfg.workers))


def _train_task(task):
    cfg, out_dir = task
    train_one(cfg, out_dir)
    return out_dir


def validate(network, cfg, controller, work_dir):
    """Mean delay of the greedy policy on the validation scenarios."""
    delays = []
    for i, level in enumerate(cfg.train.val_levels):
        seed = seeds_mod.validation(i)
        route = generate_routes(network, cfg.demand, os.path.join(work_dir, 'val.rou.xml'), seed, level)
        m = run_episode(network, controller, route, cfg.demand.duration,
                        os.path.join(work_dir, 'val'), seed=seed, **sim_kwargs(cfg))
        delays.append(m['delay'])
    return float(np.mean(delays))


def train_one(cfg, out_dir):
    """Train one run; cfg.train.seeds must hold exactly one seed."""
    (seed,) = cfg.train.seeds
    seeds_mod.check_train_seed(seed)
    tc = cfg.train
    save_run_info(cfg, out_dir)
    work_dir = os.path.join(out_dir, 'work')
    os.makedirs(work_dir, exist_ok=True)

    np.random.seed(seed)
    random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    from tsc.controllers.rl import RLController, make_agents

    network = network_for(cfg)
    observer = make_observer(network, cfg, seed)
    reward_fn = Reward(network, cfg.reward.kind)
    agents = make_agents(network, observer, cfg.agent)
    greedy = RLController(agents, observer)
    rng = np.random.default_rng(seed)

    log_path = os.path.join(out_dir, 'train_log.csv')
    log = log_file = None
    best_val = np.inf
    eps_end_ep = max(1, int(tc.episodes * tc.eps_fraction))
    label = f'{cfg.name} s{seed}'

    for ep in range(tc.episodes):
        t0 = time.time()
        epsilon = max(tc.eps_min, 1.0 - (1.0 - tc.eps_min) * ep / eps_end_ep)
        level = float(rng.uniform(tc.level_min, tc.level_max))
        route_seed = seeds_mod.train(seed, ep)
        route = generate_routes(network, cfg.demand, os.path.join(work_dir, 'train.rou.xml'),
                                route_seed, level)

        sim = Simulation(network, route, cfg.demand.duration, os.path.join(work_dir, 'episode'),
                         seed=route_seed, **sim_kwargs(cfg)).start()
        reward_fn.reset()
        observer.reset(sim)
        pending = {}
        acc = {tl_id: [0.0, 0] for tl_id in network.tl_ids}
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
            for tl_id in network.tl_ids:
                acc[tl_id][0] += reward_fn(sim, tl_id)
                acc[tl_id][1] += 1

        m = sim.close()
        row = {'episode': ep, 'level': round(level, 1), 'epsilon': round(epsilon, 3),
               'mean_reward': float(np.mean(rewards)) if rewards else 0.0,
               'mean_loss': float(np.mean(losses)) if losses else float('nan'), **m}

        if (ep + 1) % tc.val_every == 0 or ep == tc.episodes - 1:
            val = validate(network, cfg, greedy, work_dir)
            row['val_delay'] = val
            if val < best_val:
                best_val = val
                for tl_id, agent in agents.items():
                    agent.save(os.path.join(out_dir, f'best_{tl_id}.weights.h5'))
        row['seconds'] = round(time.time() - t0, 1)

        if log is None:
            fields = list(row.keys()) + (['val_delay'] if 'val_delay' not in row else [])
            log_file = open(log_path, 'w', newline='')
            log = csv.DictWriter(log_file, fieldnames=fields, extrasaction='ignore')
            log.writeheader()
        log.writerow(row)
        log_file.flush()
        print(f"[{label}] ep {ep:3d} level {level:6.0f} eps {epsilon:.2f} "
              f"delay {m['delay']:7.1f} thr {m['throughput']:.2f} "
              f"{'val %.1f ' % row['val_delay'] if 'val_delay' in row else ''}({row['seconds']}s)",
              flush=True)

    for tl_id, agent in agents.items():
        agent.save(os.path.join(out_dir, f'final_{tl_id}.weights.h5'))
    if log_file is not None:
        log_file.close()
