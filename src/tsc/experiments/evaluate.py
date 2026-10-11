"""Evaluate baselines and trained RL runs over a demand sweep.

    tsc evaluate configs/baselines.toml
    tsc evaluate configs/step1/evaluate.toml --runs "outputs/train/full/s*"

Every controller is run on the same routes: for each demand level, n_seeds
held-out scenarios (900000+).  The fixed-time plans of a level (Webster,
green wave) are computed from separate calibration scenarios (800000+), like an
engineer designing a plan for typical peak-hour demand.  With tidal demand
there is one plan per direction (like AM / PM time-of-day plans), and each
evaluation scenario runs the plan of its own direction.

Output: <out>/results.csv (one row per controller × level × scenario), plus
config.toml / run_info.json; routes and SUMO outputs are kept under <out>/.
"""

import csv
import glob
import os

from tsc import seeds as seeds_mod
from tsc.controllers import BASELINES, baseline
from tsc.controllers.webster import green_wave_programs, webster_programs
from tsc.demand import generate_routes, is_directional, scenario_flip
from tsc.experiments.common import network_for, run_parallel, save_run_info, sim_kwargs
from tsc.experiments.runs import is_run_dir, load_policy
from tsc.simulation import run_episode


def default_out(cfg):
    return os.path.join(cfg.output_dir, 'eval', cfg.name)


PLAN_BUILDERS = {'webster': webster_programs, 'greenwave': green_wave_programs}
DIRECTIONS = {False: 'fwd', True: 'rev'}


def route_path(work, level, seed, direction=''):
    suffix = f'_{direction}' if direction else ''
    return os.path.join(work, 'routes', f'L{int(level)}_s{seed}{suffix}.rou.xml')


def prepare(network, cfg, work):
    """Generate evaluation + calibration routes (once) and the fixed-time plans of every level.

    Returns {level: {direction: {baseline: plan file}}}; direction is None without tidal
    demand, else False / True (see demand.scenario_flip).
    """
    ec = cfg.evaluate
    os.makedirs(os.path.join(work, 'routes'), exist_ok=True)
    directions = [False, True] if is_directional(cfg.demand) else [None]
    plans = {}
    for level in ec.levels:
        for i in range(ec.n_seeds):
            seed = seeds_mod.evaluation(i)
            if not os.path.exists(route_path(work, level, seed)):
                generate_routes(network, cfg.demand, route_path(work, level, seed), seed, level)
        plans[level] = {}
        for flip in directions:
            tag = DIRECTIONS.get(flip, '')
            calib = []
            for i in range(ec.n_calibration):
                seed = seeds_mod.calibration(i)
                path = route_path(work, level, seed, tag)
                if not os.path.exists(path):
                    generate_routes(network, cfg.demand, path, seed, level, flip=flip)
                calib.append(path)
            plans[level][flip] = {
                name: build(network, calib, os.path.join(
                    work, 'routes', f'{name}_L{int(level)}{"_" + tag if tag else ""}.add.xml'))
                for name, build in PLAN_BUILDERS.items() if name in ec.baselines
            }
    return plans


def find_runs(patterns):
    return sorted({os.path.normpath(d) for pattern in patterns for d in glob.glob(pattern)
                   if is_run_dir(d)})


def run_task(task):
    """One controller at one demand level over all evaluation scenarios (runs in a worker)."""
    cfg, name, run_dir, level, work, plans = task
    ec = cfg.evaluate
    network = network_for(cfg)
    kwargs = sim_kwargs(cfg)
    if run_dir is None:
        tls, controller = baseline(name, cfg)
        meta = {'controller': name, 'config': name, 'train_seed': ''}
    else:
        tls = 'python'
        controller, run_cfg = load_policy(run_dir, network, which=ec.which)
        if cfg.signals.max_green is None and cfg.signals.max_red is None:
            # no shield set for the evaluation: use the one the run was trained with
            kwargs.update(max_green=run_cfg.signals.max_green, max_red=run_cfg.signals.max_red)
        seed = run_cfg.train.seeds[0]
        meta = {'controller': f'{run_cfg.name}_s{seed}', 'config': run_cfg.name, 'train_seed': seed}
    if tls != 'python':
        kwargs = {}

    rows = []
    for i in range(ec.n_seeds):
        seed = seeds_mod.evaluation(i)
        if hasattr(controller, 'reseed'):
            controller.reseed(seed)
        additional = None
        if name in PLAN_BUILDERS:
            flip = scenario_flip(cfg.demand, seed) if is_directional(cfg.demand) else None
            additional = [plans[flip][name]]
        out_dir = os.path.join(work, 'tmp', f"{meta['controller']}{ec.tag}_L{int(level)}_s{seed}")
        m = run_episode(network, controller, route_path(work, level, seed), cfg.demand.duration,
                        out_dir, seed=seed, tls=tls, additional=additional, **kwargs)
        rows.append({**meta, 'controller': meta['controller'] + ec.tag,
                     'config': meta['config'] + ec.tag, 'level': level, 'eval_seed': seed, **m})
    return rows


def evaluate(cfg, out_dir=None, workers=None):
    ec = cfg.evaluate
    for name in ec.baselines:
        if name not in BASELINES:
            raise ValueError(f'Unknown baseline {name!r}; choose from {sorted(BASELINES)}')
    work = out_dir or default_out(cfg)
    save_run_info(cfg, work)
    network = network_for(cfg)
    plans = prepare(network, cfg, work)

    run_dirs = find_runs(ec.runs)
    if ec.runs and not run_dirs:
        raise FileNotFoundError(f'No run directories match {ec.runs}')
    controllers = [(b, None) for b in ec.baselines] + [(None, d) for d in run_dirs]
    tasks = [(cfg, name, d, level, work, plans[level])
             for name, d in controllers for level in ec.levels]

    rows = []
    for n, part in enumerate(run_parallel(run_task, tasks, workers or cfg.workers), 1):
        rows.extend(part)
        r = part[0]
        print(f"[{n}/{len(tasks)}] {r['controller']:20s} level {r['level']:6.0f} "
              f"delay {sum(x['delay'] for x in part) / len(part):8.1f}", flush=True)

    rows.sort(key=lambda r: (r['controller'], r['level'], r['eval_seed']))
    out_csv = os.path.join(work, 'results.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} rows to {out_csv}')
    return out_csv

