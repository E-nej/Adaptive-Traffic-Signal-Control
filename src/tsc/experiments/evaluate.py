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

from tsc import config as config_mod
from tsc import seeds as seeds_mod
from tsc import plans as plans_mod
from tsc.controllers import BASELINES, PLAN_BASELINES, SECOND_BASELINES, baseline
from tsc.controllers.webster import green_wave_programs, webster_programs
from tsc.demand import generate_routes, is_directional, scenario_flip
from tsc.experiments.common import (PLAN_CONTROLS, file_lock, network_for, run_parallel,
                                    save_run_info, sim_kwargs)
from tsc.experiments.runs import is_run_dir, load_policy, load_run_config
from tsc.simulation import run_episode


def default_out(cfg):
    return os.path.join(cfg.output_dir, 'eval', cfg.name)


PLAN_BUILDERS = {'webster': webster_programs, 'greenwave': green_wave_programs}
DIRECTIONS = {False: 'fwd', True: 'rev'}


def route_path(work, level, seed, direction=''):
    suffix = f'_{direction}' if direction else ''
    return os.path.join(work, 'routes', f'L{int(level)}_s{seed}{suffix}.rou.xml')


def prepare(network, cfg, work, plan_names=()):
    """Generate evaluation + calibration routes (once) and the fixed-time plans of every level.

    Returns {level: {direction: {baseline: plan file}}}; direction is None without tidal
    demand, else False / True (see demand.scenario_flip).  Plans are built for the
    fixed-time baselines and for `plan_names` (e.g. 'greenwave' for plan-controlled runs),
    at every evaluation level and at evaluate.plan_level.
    """
    # parallel jobs (several `tsc` commands on one network) share these files: one builds
    # them, the others wait and reuse them
    with file_lock(os.path.join(work, 'routes', '.prepare.lock')):
        return _prepare(network, cfg, work, plan_names)


def _prepare(network, cfg, work, plan_names):
    ec = cfg.evaluate
    os.makedirs(os.path.join(work, 'routes'), exist_ok=True)
    directions = [False, True] if is_directional(cfg.demand) else [None]
    wanted = set(ec.baselines) | set(plan_names)
    levels = list(ec.levels) + ([ec.plan_level] if ec.plan_level not in (None, *ec.levels) else [])
    plans = {}
    for level in levels:
        if level in ec.levels:
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
            plans[level][flip] = {}
            for name, build in PLAN_BUILDERS.items():
                if name not in wanted:
                    continue
                path = os.path.join(work, 'routes',
                                    f'{name}_L{int(level)}{"_" + tag if tag else ""}.add.xml')
                # built once (plans are deterministic); parallel jobs share them
                plans[level][flip][name] = path if os.path.exists(path) else build(network, calib, path)
    return plans


def plan_for(cfg, plans, level, seed, name):
    """Plan file that fixed-time plan `name` uses in evaluation scenario `seed` of `level`
    (its own level and direction, unless evaluate.plan_level / plan_direction say otherwise)."""
    ec = cfg.evaluate
    if ec.plan_direction not in ('match', 'opposite'):
        raise ValueError(f'Unknown plan_direction {ec.plan_direction!r}')
    flip = scenario_flip(cfg.demand, seed) if is_directional(cfg.demand) else None
    if flip is not None and ec.plan_direction == 'opposite':
        flip = not flip
    return plans[ec.plan_level if ec.plan_level is not None else level][flip][name]


def plan_library(network, cfg, work=None):
    """Green-wave plans at train.plan_levels x directions, from the calibration scenarios
    (the same plans the evaluation computes).  Returns {(level, flip): plan file}."""
    import hashlib
    import json
    key = hashlib.sha1(json.dumps(config_mod.to_dict(cfg)['demand'], sort_keys=True).encode())
    work = work or os.path.join(cfg.output_dir, 'plans',
                                f'{os.path.basename(network.work_dir)}-{key.hexdigest()[:8]}')
    lib_cfg = config_mod.replace(cfg, evaluate={'levels': list(cfg.train.plan_levels),
                                                'n_seeds': 0, 'baselines': [], 'plan_level': None})
    plans = prepare(network, lib_cfg, work, plan_names=('greenwave',))
    return {(level, flip): files['greenwave']
            for level, by_dir in plans.items() for flip, files in by_dir.items()}


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
        if name == 'select_assigned':
            import json
            from tsc.experiments.assign import make_fixed
            with open(ec.assignment) as f:
                table = {float(k): v for k, v in json.load(f).items()}
            near = min(table, key=lambda x: abs(x - level))
            tls, controller = 'python', make_fixed(network, cfg, table[near])
            kwargs = sim_kwargs(config_mod.replace(cfg, signals={'control': 'plan'}))
        elif name in SECOND_BASELINES:
            plan_cfg = config_mod.replace(cfg, signals={'control': 'plan'})
            kwargs = sim_kwargs(plan_cfg)
            kwargs.update(max_green=None, max_red=None)   # the plan serves every stage
            tls, controller = baseline(name, plan_cfg, network)
        else:
            tls, controller = baseline(name, cfg, network)
        meta = {'controller': name, 'config': name, 'train_seed': ''}
    else:
        tls = 'python'
        controller, run_cfg = load_policy(run_dir, network, which=ec.which, device=ec.device)
        if run_cfg.signals.control in PLAN_CONTROLS:
            kwargs = sim_kwargs(run_cfg)        # plan timing and the run's own shield
        elif cfg.signals.max_green is None and cfg.signals.max_red is None:
            # no shield set for the evaluation: use the one the run was trained with
            kwargs.update(max_green=run_cfg.signals.max_green, max_red=run_cfg.signals.max_red)
        kwargs['approach_length'] = run_cfg.observation.approach_length   # its detectors
        seed = run_cfg.train.seeds[0]
        tag = f'_{ec.which}' if ec.which != 'best' else ''
        meta = {'controller': f'{run_cfg.name}{tag}_s{seed}', 'config': run_cfg.name + tag,
                'train_seed': seed}
    if tls != 'python':                     # SUMO runs the signals: only the scenario options
        kwargs = {k: kwargs[k] for k in ('begin', 'undeparted')}
    needs_plan = getattr(controller, 'needs_plan', False)
    loaded = {}

    rows = []
    for i in range(ec.n_seeds):
        seed = seeds_mod.evaluation(i)
        if hasattr(controller, 'reseed'):
            controller.reseed(seed)
        additional = None
        if name in PLAN_BUILDERS:
            additional = [plan_for(cfg, plans, level, seed, name)]
        if needs_plan:
            path = plan_for(cfg, plans, level, seed, 'greenwave')
            if path not in loaded:
                loaded[path] = plans_mod.load_plans(path, network)
            controller.set_plans(loaded[path])
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
    run_dirs = find_runs(ec.runs)
    if ec.runs and not run_dirs:
        raise FileNotFoundError(f'No run directories match {ec.runs}')
    plan_runs = any(load_run_config(d).signals.control in PLAN_CONTROLS for d in run_dirs)
    plan_names = ('greenwave',) if plan_runs or set(PLAN_BASELINES) & set(ec.baselines) else ()
    plans = prepare(network, cfg, work, plan_names)

    controllers = [(b, None) for b in ec.baselines] + [(None, d) for d in run_dirs]
    tasks = [(cfg, name, d, level, work, plans)
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

