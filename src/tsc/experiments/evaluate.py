"""Evaluate baselines and trained RL runs over a demand sweep.

    tsc evaluate configs/baselines.toml
    tsc evaluate configs/step1/evaluate.toml --runs "outputs/train/full/s*"

Every controller is run on the same routes: for each demand level, n_seeds
held-out scenarios (900000+).  The Webster plan for a level is computed from
separate calibration scenarios (800000+), like an engineer designing a plan
for typical peak-hour demand.

Output: <out>/results.csv (one row per controller × level × scenario), plus
config.toml / run_info.json; routes and SUMO outputs are kept under <out>/.
"""

import csv
import glob
import os

from tsc import seeds as seeds_mod
from tsc.controllers import BASELINES, baseline
from tsc.controllers.webster import webster_programs
from tsc.demand import generate_routes
from tsc.experiments.common import network_for, run_parallel, save_run_info, sim_kwargs
from tsc.experiments.runs import is_run_dir, load_policy
from tsc.simulation import run_episode


def default_out(cfg):
    return os.path.join(cfg.output_dir, 'eval', cfg.name)


def route_path(work, level, seed):
    return os.path.join(work, 'routes', f'L{int(level)}_s{seed}.rou.xml')


def prepare(network, cfg, work):
    """Generate evaluation + calibration routes (once) and the Webster plan of every level."""
    ec = cfg.evaluate
    os.makedirs(os.path.join(work, 'routes'), exist_ok=True)
    webster = {}
    for level in ec.levels:
        for i in range(ec.n_seeds):
            seed = seeds_mod.evaluation(i)
            if not os.path.exists(route_path(work, level, seed)):
                generate_routes(network, cfg.demand, route_path(work, level, seed), seed, level)
        calib = []
        for i in range(ec.n_calibration):
            seed = seeds_mod.calibration(i)
            if not os.path.exists(route_path(work, level, seed)):
                generate_routes(network, cfg.demand, route_path(work, level, seed), seed, level)
            calib.append(route_path(work, level, seed))
        if 'webster' in ec.baselines:
            webster[level] = webster_programs(
                network, calib, os.path.join(work, 'routes', f'webster_L{int(level)}.add.xml'))
    return webster


def find_runs(patterns):
    return sorted({os.path.normpath(d) for pattern in patterns for d in glob.glob(pattern)
                   if is_run_dir(d)})


def run_task(task):
    """One controller at one demand level over all evaluation scenarios (runs in a worker)."""
    cfg, name, run_dir, level, work, webster_file = task
    ec = cfg.evaluate
    network = network_for(cfg)
    kwargs = sim_kwargs(cfg)
    additional = None
    if run_dir is None:
        tls, controller = baseline(name)
        meta = {'controller': name, 'config': name, 'train_seed': ''}
        if name == 'webster':
            additional = [webster_file]
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
    webster = prepare(network, cfg, work)

    run_dirs = find_runs(ec.runs)
    if ec.runs and not run_dirs:
        raise FileNotFoundError(f'No run directories match {ec.runs}')
    controllers = [(b, None) for b in ec.baselines] + [(None, d) for d in run_dirs]
    tasks = [(cfg, name, d, level, work, webster.get(level))
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

