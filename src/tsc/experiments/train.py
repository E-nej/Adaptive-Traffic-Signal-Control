"""Train RL signal controllers on randomised, realistic demand.

    tsc train configs/step1/full.toml --seeds 0-4

Every episode samples a demand level uniformly from [level_min, level_max] and
generates a fresh scenario (seed 100000·seed + episode).  Every `val_every`
episodes the greedy policy is validated on fixed scenarios (700001+) and the
best one is kept (`best_<tl>.pt`); the last one is saved as `final_*`.

Output per training seed: <out>/s<seed>/ with config.toml, run_info.json,
train_log.csv and the weights.
"""

import copy
import csv
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from tsc import config as config_mod
from tsc import devices
from tsc import seeds as seeds_mod
from tsc.demand import generate_routes, is_directional, scenario_flip
from tsc.experiments.common import (PLAN_CONTROLS, network_for, run_parallel, save_run_info,
                                    sim_kwargs, update_run_info)
from tsc.experiments.runs import make_controller, make_observer
from tsc.reward import Reward
from tsc.simulation import Simulation, run_episode


def default_out(cfg):
    return os.path.join(cfg.output_dir, 'train', cfg.name)


def train(cfg, seeds=None, out_root=None, workers=None, gpu_workers=0):
    """Train one run per seed (in parallel); returns the run directories.

    gpu_workers > 0: the first gpu_workers seeds train on the GPU, the others on the CPU
    (hybrid scheduling); each run records its device in its config.toml.
    """
    seeds = list(seeds if seeds is not None else cfg.train.seeds)
    for s in seeds:
        seeds_mod.check_train_seed(s)
    devices.check(cfg.train.device)
    out_root = out_root or default_out(cfg)
    network = network_for(cfg)  # build the network once before the workers start
    if cfg.signals.control in PLAN_CONTROLS:
        from tsc.experiments.evaluate import plan_library
        plan_library(network, cfg)      # and the plans
    tasks = []
    for i, s in enumerate(seeds):
        train_cfg = {'seeds': [s]}
        if gpu_workers:
            train_cfg['device'] = 'gpu' if i < gpu_workers else 'cpu'
        tasks.append((config_mod.replace(cfg, train=train_cfg), os.path.join(out_root, f's{s}')))
    # A process that has set up a PyTorch device cannot change it: mixed devices
    # need a fresh process per run.
    mixed = len({t[0].train.device for t in tasks}) > 1
    return list(run_parallel(_train_task, tasks, workers or cfg.workers, fresh=mixed))


def _train_task(task):
    cfg, out_dir = task
    train_one(cfg, out_dir)
    return out_dir


def validate(network, cfg, controller, work_dir, library=None, routes=None):
    """Mean delay of the greedy policy on the validation scenarios.

    Plan control: every scenario runs once with its matching plan and once with the
    opposite direction's plan of the same level (a plan that does not fit the demand).
    `routes` (a dict, filled on first use) keeps the generated route files, which are the
    same at every validation of a run.
    """
    return float(np.mean(scenario_delays(network, cfg, controller, work_dir, library,
                                         per_level=cfg.train.val_per_level, routes=routes)))


def scenario_delays(network, cfg, controller, work_dir, library=None, per_level=1,
                    seed_of=seeds_mod.validation, routes=None):
    """Delay of every (scenario, plan variant) at the validation levels, `per_level`
    scenarios per level with seeds `seed_of(i)` (validation: one per level).  `routes`
    caches the route files by seed (see validate)."""
    routes = {} if routes is None else routes
    delays = []
    jobs = [(lv, k) for k in range(per_level) for lv in cfg.train.val_levels]
    for i, (level, _) in enumerate(jobs):
        seed = seed_of(i)
        if seed not in routes:
            routes[seed] = generate_routes(network, cfg.demand,
                                           os.path.join(work_dir, f'val_{seed}.rou.xml'), seed, level)
        route = routes[seed]
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
                            os.path.join(work_dir, 'val'), seed=seed, emissions=False,
                            **sim_kwargs(cfg))
            delays.append(m['delay'])
    return delays


def nearest_level(levels, level):
    return min(levels, key=lambda x: abs(x - level))


def _default_expert(cfg, network, greedy, work_dir, library, out_dir, label, val_routes=None):
    """Hierarchy with a default expert: validate every expert alone (workers agreeing),
    set the default ('auto': the best one) and save it alone as the first 'best'
    checkpoint, so the run deploys it unless the learned policy beats it on validation.
    Returns its validation delay."""
    import json
    actions = greedy.actions
    actions.set_training(False, 0.0)
    vals = {}
    for e, name in enumerate(actions.names):
        actions.force(e)
        vals[name] = validate(network, cfg, greedy, work_dir, library, val_routes)
    name = cfg.hierarchy.default_expert
    if name == 'auto':
        name = min(vals, key=vals.get)
    default = actions.names.index(name)
    actions.set_default(default)
    actions.force(default)
    greedy.save(out_dir, 'best')
    actions.force(None)
    with open(os.path.join(out_dir, 'experts_val.json'), 'w') as f:
        json.dump({'val_delay': vals, 'default': name}, f, indent=1)
    print(f"[{label}] experts alone (val delay): "
          + ', '.join(f'{n} {v:.1f}' for n, v in vals.items()) + f" -> default {name}", flush=True)
    return vals[name]


def _worker_baseline(cfg, network, greedy, work_dir, library, out_dir, label, val_routes=None):
    """Workers under a fixed expert (expert-specific workers): validate the expert alone
    (workers agreeing) and save it as the first 'best' checkpoint, so the run deploys the
    expert alone unless its workers beat it on validation.  Returns its validation delay."""
    actions = greedy.actions
    actions.set_training(False, 0.0)
    actions.force_agree = True
    val = validate(network, cfg, greedy, work_dir, library, val_routes)
    greedy.save(out_dir, 'best')
    actions.force_agree = False
    print(f'[{label}] {actions.names[cfg.hierarchy.fixed_expert]} alone (val delay): {val:.1f}',
          flush=True)
    return val


def _uses_fallback(cfg):
    """Hierarchy runs whose best checkpoint starts as a non-learning fallback: a learned
    manager with a default expert, or workers under a fixed expert."""
    h = cfg.hierarchy
    if cfg.signals.control != 'hierarchy':
        return None
    if h.manager == 'learned' and h.default_expert:
        return 'manager'
    if h.manager == 'fixed' and h.worker == 'residual':
        return 'workers'
    return None


def accept(run_dir, per_level=None, alpha=0.05):
    """Does the best checkpoint beat its fallback?  A learned manager is compared with its
    default expert alone, workers under a fixed expert with that expert alone (workers
    agreeing).  Both run on fresh scenarios (acceptance seeds, `per_level` per validation
    level), paired; the learned policy is kept only if its mean delay is lower and a
    one-sided Wilcoxon test gives p < alpha, else the best checkpoint is switched to the
    fallback.  Validation picked the checkpoint as the best of many noisy readings, so its
    own score is biased in its favour.  Writes accept.json; returns its content."""
    import json
    from scipy import stats
    from tsc.experiments.runs import load_policy, load_run_config
    cfg = load_run_config(run_dir)
    per_level = per_level or cfg.hierarchy.accept_scenarios or 4
    network = network_for(cfg)
    library = load_library(network, cfg) if cfg.signals.control in PLAN_CONTROLS else None
    policy, _ = load_policy(run_dir, network, 'best')
    actions = policy.actions
    manager = actions.manager
    mode = _uses_fallback(cfg)
    if mode == 'manager':
        default = actions.names[manager.default]
        done = manager.forced is not None
    elif mode == 'workers':
        default = actions.names[manager.fixed_expert]
        done = actions.force_agree
    else:
        raise ValueError(f'{run_dir}: nothing to accept (needs hierarchy.default_expert, or '
                         f'workers under a fixed expert)')
    out = {'default': default, 'scenarios': per_level * len(cfg.train.val_levels)}
    if done:                                # validation never beat the fallback
        out.update(deployed=default, reason='best checkpoint is already the fallback')
    else:
        work_dir = os.path.join(run_dir, 'work')
        os.makedirs(work_dir, exist_ok=True)
        routes = {}                         # the same scenarios for both: routes generated once
        args = (network, cfg, policy, work_dir, library, per_level, seeds_mod.acceptance, routes)
        learned = np.array(scenario_delays(*args))
        if mode == 'manager':
            actions.force(manager.default)
        else:
            actions.force_agree = True
        alone = np.array(scenario_delays(*args))
        diff = learned - alone
        p = float(stats.wilcoxon(diff, alternative='less').pvalue) if diff.any() else 1.0
        keep = diff.mean() < 0 and p < alpha
        if keep:
            actions.force(None)             # learned manager, workers active
        actions.save_meta(run_dir, 'best')
        out.update(deployed='learned' if keep else default, p=p,
                   learned=float(learned.mean()), alone=float(alone.mean()),
                   change=float(diff.mean() / alone.mean()))
    with open(os.path.join(run_dir, 'accept.json'), 'w') as f:
        json.dump(out, f, indent=1)
    return out


TABLE_EPISODE = 90000       # table scenarios: training seeds, episodes 90000+ (no overlap)


def _table_task(task):
    """One expert (with its worker if `worker`) on one scenario: (level, expert, delay,
    [(flow, jam) of every interval], worker)."""
    cfg, level, k, expert, worker, work_dir = task
    network = network_for(cfg)
    (seed,) = cfg.train.seeds
    route_seed = seeds_mod.train(seed, TABLE_EPISODE + k)
    controller = make_controller(network, make_observer(network, cfg, seed), cfg)
    controller.actions.force(expert, agree=not worker)
    controller.reseed(route_seed)
    run_dir = os.path.join(work_dir, f'table_{level:.0f}_{k}_{expert}{"w" if worker else ""}')
    os.makedirs(run_dir, exist_ok=True)
    route = generate_routes(network, cfg.demand, os.path.join(run_dir, 'r.rou.xml'), route_seed, level)
    if cfg.signals.control in PLAN_CONTROLS:
        flip = scenario_flip(cfg.demand, route_seed) if is_directional(cfg.demand) else None
        library = load_library(network, cfg)
        controller.set_plans(library[(nearest_level(cfg.train.plan_levels, level), flip)])
    m = run_episode(network, controller, route, cfg.demand.duration, run_dir, seed=route_seed,
                    **sim_kwargs(cfg))
    return (level, expert, m['delay'], [list(f) for f in controller.actions.manager.flows],
            worker)


def build_table(cfg, greedy, out_dir, label):
    """Manager 'table': every expert (with its worker, worker = 'expert') at every level of
    the grid (table_scenarios scenarios with matching plans), the best one per level by mean delay, keyed by the network flow
    and jam ratio the manager measured at that level in each interval (mean over experts
    and scenarios).  With table_alone, an expert that has a worker is also measured without
    it, and the table picks per level.  Saves it as the 'best' and 'final' policy and the
    measurements as table.csv (delays) and table_keys.csv (readings)."""
    import pandas as pd
    h, tc = cfg.hierarchy, cfg.train
    levels = h.table_levels or list(np.linspace(tc.level_min, tc.level_max, 9))
    actions = greedy.actions
    work_dir = os.path.join(out_dir, 'work')
    with_worker = h.worker == 'expert'
    options = [(e, with_worker) for e in actions.selector.available]
    if with_worker and h.table_alone:
        options += [(e, False) for e in sorted(actions.expert_workers)]
    tasks = [(cfg, float(lv), k, e, w, work_dir) for lv in levels
             for k in range(h.table_scenarios) for e, w in options]
    results = list(run_parallel(_table_task, tasks, cfg.workers))
    # an expert without a worker runs alone either way: one option, named after the expert
    def option(e, w):
        return actions.names[e] + ('+worker' if w and e in actions.expert_workers else '')
    df = pd.DataFrame([(r[0], option(r[1], r[4]), r[2]) for r in results],
                      columns=['level', 'expert', 'delay'])
    df.sort_values(['level', 'expert']).to_csv(os.path.join(out_dir, 'table.csv'), index=False)
    mean = df.groupby(['level', 'expert']).delay.mean().unstack()
    best = mean.idxmin(axis=1)
    readings = pd.DataFrame([(r[0], i, f, j) for r in results for i, (f, j) in enumerate(r[3])],
                            columns=['level', 'interval', 'flow', 'jam'])
    keys = readings.groupby(['level', 'interval'])[['flow', 'jam']].mean()
    keys.to_csv(os.path.join(out_dir, 'table_keys.csv'))
    def parse(name):
        expert = name.removesuffix('+worker')
        return actions.names.index(expert), name.endswith('+worker') or not with_worker
    table = [(keys.loc[lv].to_numpy(), *parse(best[lv])) for lv in mean.index]
    actions.manager.table = table
    actions.set_default(parse(best.value_counts().idxmax())[0])
    for which in ('best', 'final'):
        greedy.save(out_dir, which)
    print(f'[{label}] table (level: mean flow, jam -> expert): '
          + ', '.join(f'{lv:.0f}: {keys.loc[lv].flow.mean():.3f}, {keys.loc[lv].jam.mean():.3f} '
                      f'-> {best[lv]}' for lv in mean.index), flush=True)
    return table


def _accept_task(task):
    run_dir, per_level = task
    return run_dir, accept(run_dir, per_level)


def accept_runs(run_dirs, per_level=None, workers=1):
    """accept() for several runs in parallel; yields (run_dir, result)."""
    return run_parallel(_accept_task, [(d, per_level) for d in run_dirs], workers)


def load_library(network, cfg):
    """{(level, flip): {tl_id: StagePlan}} of the plan library (plan control)."""
    from tsc.experiments.evaluate import plan_library
    from tsc.plans import load_plans
    return {key: load_plans(path, network) for key, path in plan_library(network, cfg).items()}


def train_one(cfg, out_dir, on_episode_end=None):
    """Train one run; cfg.train.seeds must hold exactly one seed.

    on_episode_end(ep, info), if given, is called after every episode with the state
    the parity harness inspects (tsc.experiments.parity).  It must not consume RNG.
    """
    (seed,) = cfg.train.seeds
    seeds_mod.check_train_seed(seed)
    tc = cfg.train
    save_run_info(cfg, out_dir)
    work_dir = os.path.join(out_dir, 'work')
    os.makedirs(work_dir, exist_ok=True)

    # Import PyTorch and the agent *before* seeding, so nothing an import does can shift the
    # RNG streams (weights, replay sampling) of a seeded run.  The device is set up around
    # the import (hiding GPUs must happen before it); this consumes no RNG.
    devices.prepare(tc.device)
    import torch
    import tsc.agents.dqn  # noqa: F401
    device = devices.configure(tc.device, tc.gpu_index)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)                 # also seeds every CUDA / ROCm device

    update_run_info(out_dir, device=device)
    network = network_for(cfg)
    observer = make_observer(network, cfg, seed)
    reward_fn = Reward(network, cfg.reward.kind, cfg.reward.neighbour_weight,
                       cfg.reward.pressure_weight)
    greedy = make_controller(network, observer, cfg)
    agents, actions = greedy.agents, greedy.actions
    rng = np.random.default_rng(seed)
    library = load_library(network, cfg) if cfg.signals.control in PLAN_CONTROLS else None
    plan_rng = np.random.default_rng(seed + 31337)      # separate stream: plan choice
    scale = cfg.reward.scale
    smdp, gamma = cfg.agent.smdp, cfg.agent.gamma

    log_path = os.path.join(out_dir, 'train_log.csv')
    log = log_file = None
    best_val = np.inf
    eps_end_ep = max(1, int(tc.episodes * tc.eps_fraction))
    label = f'{cfg.name} s{seed}'

    has_workers = bool(agents)
    set_training = getattr(actions, 'set_training', None)
    if cfg.signals.control == 'hierarchy' and cfg.hierarchy.manager == 'table':
        build_table(cfg, greedy, out_dir, label)        # nothing to train
        return
    if cfg.signals.control == 'hierarchy' and cfg.hierarchy.worker == 'expert':
        raise ValueError('worker = "expert" uses workers trained in their own runs; only '
                         'manager = "table" builds a run with them')
    val_routes = {}                 # validation route files, generated once per run
    fallback = _uses_fallback(cfg)
    if fallback == 'manager':
        best_val = _default_expert(cfg, network, greedy, work_dir, library, out_dir, label,
                                   val_routes)
    elif fallback == 'workers':
        best_val = _worker_baseline(cfg, network, greedy, work_dir, library, out_dir, label,
                                    val_routes)

    # The next episode's routes are generated in the background (a SUMO subprocess) while
    # this episode runs.  Its level is peeked from a copy of `rng`, so the draws from `rng`
    # itself happen exactly as before; two route files alternate.
    prefetch = ThreadPoolExecutor(max_workers=1)
    next_route = None

    def route_file(ep):
        return os.path.join(work_dir, f'train_{"ab"[ep % 2]}.rou.xml')

    for ep in range(tc.episodes):
        t0 = time.time()
        epsilon = max(tc.eps_min, tc.eps_start - (tc.eps_start - tc.eps_min) * ep / eps_end_ep)
        if set_training:
            set_training(True, epsilon)
        level = float(rng.uniform(tc.level_min, tc.level_max))
        route_seed = seeds_mod.train(seed, ep)
        if next_route is None:
            route = generate_routes(network, cfg.demand, route_file(ep), route_seed, level)
        else:
            next_level, future = next_route
            assert next_level == level, 'prefetched level differs from the drawn one'
            route = future.result()
        next_route = None
        if ep + 1 < tc.episodes:
            next_level = float(copy.deepcopy(rng).uniform(tc.level_min, tc.level_max))
            next_route = (next_level, prefetch.submit(
                generate_routes, network, cfg.demand, route_file(ep + 1),
                seeds_mod.train(seed, ep + 1), next_level))

        if library is not None:
            flip = scenario_flip(cfg.demand, route_seed) if is_directional(cfg.demand) else None
            keys = sorted(library, key=str)
            if plan_rng.random() < tc.plan_match:
                key = (nearest_level(tc.plan_levels, level), flip)
            else:
                key = keys[plan_rng.integers(len(keys))]
            greedy.set_plans(library[key])

        sim = Simulation(network, route, cfg.demand.duration, os.path.join(work_dir, 'episode'),
                         seed=route_seed, emissions=False, **sim_kwargs(cfg)).start()
        reward_fn.reset()
        greedy.reset(sim)
        pending = {}
        acc = {tl_id: [0.0, 0] for tl_id in network.tl_ids}
        rewards, losses = [], []

        while not sim.done:
            for tl_id in sim.decision_points():
                if not has_workers:
                    sim.heads[tl_id].request(actions.stage(sim, tl_id, 0))
                    continue
                if actions.decides(sim, tl_id) or tl_id not in pending:
                    agent = agents[tl_id]
                    s = greedy.state(sim, tl_id)
                    mask = actions.mask(sim, tl_id)
                    if tl_id in pending:
                        ps, pa = pending[tl_id]
                        total, secs = acc[tl_id]
                        if smdp:        # discounted sum over the decision's seconds
                            r, discount = scale * total, gamma ** secs
                        else:           # mean per second, one discount per decision
                            r, discount = scale * total / max(1, secs), None
                        agent.store(ps, pa, r, s, mask, discount, key=tl_id)
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
            actions.tick(sim)
            for tl_id, r in reward_fn.all(sim).items():
                acc[tl_id][0] += r * (gamma ** acc[tl_id][1] if smdp else 1.0)
                acc[tl_id][1] += 1

        m = sim.close()
        for agent in {id(a): a for a in (agents or {}).values()}.values():
            if hasattr(agent, 'end_episode'):
                agent.end_episode()
        row = {'episode': ep, 'level': round(level, 1), 'epsilon': round(epsilon, 3),
               'mean_reward': float(np.mean(rewards)) if rewards else 0.0,
               'mean_loss': float(np.mean(losses)) if losses else float('nan'), **m}
        if hasattr(actions, 'episode_stats'):
            row.update(actions.episode_stats())

        improved = False
        if (ep + 1) % tc.val_every == 0 or ep == tc.episodes - 1:
            if set_training:
                set_training(False, 0.0)
            val = validate(network, cfg, greedy, work_dir, library, val_routes)
            row['val_delay'] = val
            if val < best_val:
                improved = True
                best_val = val
                greedy.save(out_dir, 'best')
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
        if on_episode_end is not None:
            on_episode_end(ep, {'row': row, 'improved': improved, 'agents': agents or {},
                                'route': route,
                                'rngs': {'trainer': rng, 'plan': plan_rng, 'observer': observer.rng,
                                         'observer_messages': observer.message_rng}})

    prefetch.shutdown()
    if device['device'] == 'gpu':
        update_run_info(out_dir, gpu_peak_memory_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1))
    greedy.save(out_dir, 'final')
    if log_file is not None:
        log_file.close()
    if fallback and cfg.hierarchy.accept_scenarios:
        result = accept(out_dir)
        print(f"[{label}] acceptance: deployed {result['deployed']} "
              f"({result.get('change', 0):+.1%}, p={result.get('p', float('nan')):.3f})", flush=True)
