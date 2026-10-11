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
from tsc.demand import generate_routes, is_directional, scenario_flip
from tsc.experiments.common import network_for, run_parallel, save_run_info, sim_kwargs
from tsc.experiments.runs import make_controller, make_observer
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
    network = network_for(cfg)  # build the network once before the workers start
    if cfg.signals.control == 'plan':
        from tsc.experiments.evaluate import plan_library
        plan_library(network, cfg)      # and the plans
    tasks = [(config_mod.replace(cfg, train={'seeds': [s]}), os.path.join(out_root, f's{s}'))
             for s in seeds]
    return list(run_parallel(_train_task, tasks, workers or cfg.workers))


def _train_task(task):
    cfg, out_dir = task
    train_one(cfg, out_dir)
    return out_dir


def validate(network, cfg, controller, work_dir, library=None):
    """Mean delay of the greedy policy on the validation scenarios.

    Plan control: every scenario runs once with its matching plan and once with the
    opposite direction's plan of the same level (a plan that does not fit the demand).
    """
    delays = []
    for i, level in enumerate(cfg.train.val_levels):
        seed = seeds_mod.validation(i)
        route = generate_routes(network, cfg.demand, os.path.join(work_dir, 'val.rou.xml'), seed, level)
        variants = [None]
        if library is not None:
            flip = scenario_flip(cfg.demand, seed) if is_directional(cfg.demand) else None
            near = nearest_level(cfg.train.plan_levels, level)
            variants = [library[(near, flip)]]
            if flip is not None:
                variants.append(library[(near, not flip)])
        for plans in variants:
            if plans is not None:
                controller.set_plans(plans)
            m = run_episode(network, controller, route, cfg.demand.duration,
                            os.path.join(work_dir, 'val'), seed=seed, **sim_kwargs(cfg))
            delays.append(m['delay'])
    return float(np.mean(delays))


def nearest_level(levels, level):
    return min(levels, key=lambda x: abs(x - level))


def load_library(network, cfg):
    """{(level, flip): {tl_id: StagePlan}} of the plan library (plan control)."""
    from tsc.experiments.evaluate import plan_library
    from tsc.plans import load_plans
    return {key: load_plans(path, network) for key, path in plan_library(network, cfg).items()}


def train_one(cfg, out_dir):
    """Train one run; cfg.train.seeds must hold exactly one seed."""
    (seed,) = cfg.train.seeds
    seeds_mod.check_train_seed(seed)
    tc = cfg.train
    save_run_info(cfg, out_dir)
    work_dir = os.path.join(out_dir, 'work')
    os.makedirs(work_dir, exist_ok=True)

    # Import TensorFlow/Keras *before* seeding, as the pre-package train.py did: importing
    # them after seeding shifts the RNG streams (weights, replay sampling), so the same
    # seed would no longer reproduce the same run.
    import tensorflow as tf
    import tsc.agents.dqn  # noqa: F401
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    network = network_for(cfg)
    observer = make_observer(network, cfg, seed)
    reward_fn = Reward(network, cfg.reward.kind, cfg.reward.neighbour_weight)
    greedy = make_controller(network, observer, cfg)
    agents, actions = greedy.agents, greedy.actions
    rng = np.random.default_rng(seed)
    library = load_library(network, cfg) if cfg.signals.control == 'plan' else None
    plan_rng = np.random.default_rng(seed + 31337)      # separate stream: plan choice
    scale = cfg.reward.scale

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

        if library is not None:
            flip = scenario_flip(cfg.demand, route_seed) if is_directional(cfg.demand) else None
            keys = sorted(library, key=str)
            if plan_rng.random() < tc.plan_match:
                key = (nearest_level(tc.plan_levels, level), flip)
            else:
                key = keys[plan_rng.integers(len(keys))]
            greedy.set_plans(library[key])

        sim = Simulation(network, route, cfg.demand.duration, os.path.join(work_dir, 'episode'),
                         seed=route_seed, **sim_kwargs(cfg)).start()
        reward_fn.reset()
        greedy.reset(sim)
        pending = {}
        acc = {tl_id: [0.0, 0] for tl_id in network.tl_ids}
        rewards, losses = [], []

        while not sim.done:
            for tl_id in sim.decision_points():
                if actions.decides(sim, tl_id) or tl_id not in pending:
                    agent = agents[tl_id]
                    s = greedy.state(sim, tl_id)
                    mask = actions.mask(sim, tl_id)
                    if tl_id in pending:
                        ps, pa = pending[tl_id]
                        r = scale * acc[tl_id][0] / max(1, acc[tl_id][1])
                        agent.store(ps, pa, r, s, mask)
                        rewards.append(r)
                        loss = agent.replay()
                        if loss is not None:
                            losses.append(loss)
                    a = agent.act(s, mask, epsilon)
                    greedy.last[tl_id] = a
                    pending[tl_id] = (s, a)
                    acc[tl_id] = [0.0, 0]
                sim.heads[tl_id].request(actions.stage(sim, tl_id, greedy.last[tl_id]))
            sim.step()
            observer.tick(sim)
            for tl_id, r in reward_fn.all(sim).items():
                acc[tl_id][0] += r
                acc[tl_id][1] += 1

        m = sim.close()
        row = {'episode': ep, 'level': round(level, 1), 'epsilon': round(epsilon, 3),
               'mean_reward': float(np.mean(rewards)) if rewards else 0.0,
               'mean_loss': float(np.mean(losses)) if losses else float('nan'), **m}

        if (ep + 1) % tc.val_every == 0 or ep == tc.episodes - 1:
            val = validate(network, cfg, greedy, work_dir, library)
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
