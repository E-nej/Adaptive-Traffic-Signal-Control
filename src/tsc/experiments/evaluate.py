"""Evaluate baselines and trained RL runs over a demand sweep.

Example:
    python evaluate.py --runs runs/train/* --out runs/eval/step1.csv

Every controller is run on the same routes: for each demand level, --seeds
held-out scenario seeds (900000+).  The Webster plan for a level is computed
from separate calibration seeds (800000+), like an engineer designing a plan
for typical peak-hour demand.
"""

import argparse
import csv
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from scenario import Scenario
from sim import run_controller

SCENARIO_DIR = os.path.join('runs', 'scenario')


def route_path(work, level, seed):
    return os.path.join(work, 'routes', f'L{int(level)}_s{seed}.rou.xml')


def prepare(scn, work, levels, n_seeds, duration, profile):
    os.makedirs(os.path.join(work, 'routes'), exist_ok=True)
    webster = {}
    for level in levels:
        for i in range(n_seeds):
            seed = 900000 + i
            if not os.path.exists(route_path(work, level, seed)):
                scn.generate_routes(route_path(work, level, seed), seed, level, duration, profile)
        calib = []
        for i in range(3):
            seed = 800000 + i
            if not os.path.exists(route_path(work, level, seed)):
                scn.generate_routes(route_path(work, level, seed), seed, level, duration, profile)
            calib.append(route_path(work, level, seed))
        webster[level] = scn.webster_programs(
            calib, os.path.join(work, 'routes', f'webster_L{int(level)}.add.xml'))
    return webster


def run_task(task):
    """One controller at one demand level over all evaluation seeds (runs in a worker)."""
    name, run_dir, level, n_seeds, duration, work, webster_file, net, shield, tag = task
    scn = Scenario(net, SCENARIO_DIR)
    meta = {'controller': name, 'config': name, 'train_seed': ''}
    controller, tls, additional = None, 'python', None
    if name == 'fixed':
        tls = 'fixed'
    elif name == 'webster':
        tls, additional = 'webster', [webster_file]
    elif name == 'actuated':
        tls = 'actuated'
    elif name == 'maxpressure':
        from controllers import MaxPressure
        controller = MaxPressure()
    else:
        from train import load_run
        controller, run_meta = load_run(run_dir, scn, which='best')
        if not any(shield.values()):
            from train import shield_kwargs
            shield = shield_kwargs(run_meta)
        meta = {'controller': name, 'config': run_meta['config'], 'train_seed': run_meta['seed']}

    rows = []
    for i in range(n_seeds):
        seed = 900000 + i
        out_dir = os.path.join(work, 'tmp', f'{name}{tag}_L{int(level)}_s{seed}')
        m = run_controller(scn, controller, route_path(work, level, seed), duration, out_dir,
                           seed=seed, tls=tls, additional=additional,
                           **(shield if tls == 'python' else {}))
        rows.append({**meta, 'controller': meta['controller'] + tag, 'config': meta['config'] + tag,
                     'level': level, 'eval_seed': seed, **m})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='*', default=[], help='trained run directories (globs ok)')
    p.add_argument('--baselines', nargs='*', default=['fixed', 'webster', 'actuated', 'maxpressure'])
    p.add_argument('--levels', nargs='*', type=float, default=[600, 1200, 1800, 2400, 3000, 3600])
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--duration', type=int, default=3600)
    p.add_argument('--profile', default='peak')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max-green', type=int, default=None, help='shield for Python controllers')
    p.add_argument('--max-red', type=int, default=None)
    p.add_argument('--tag', default='', help='suffix for controller names (e.g. _shield)')
    p.add_argument('--net', default='intersection/test.net.xml')
    p.add_argument('--out', default='runs/eval/eval.csv')
    args = p.parse_args()

    work = os.path.dirname(args.out) or '.'
    scn = Scenario(args.net, SCENARIO_DIR)
    webster = prepare(scn, work, args.levels, args.seeds, args.duration, args.profile)

    run_dirs = sorted({d for pattern in args.runs for d in glob.glob(pattern)
                       if os.path.exists(os.path.join(d, 'config.json'))})
    controllers = [(b, None) for b in args.baselines] + \
                  [(os.path.basename(os.path.normpath(d)), d) for d in run_dirs]
    shield = {'max_green': args.max_green, 'max_red': args.max_red}
    tasks = [(name, d, level, args.seeds, args.duration, work, webster[level], args.net, shield,
              args.tag)
             for name, d in controllers for level in args.levels]

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_task, t) for t in tasks]
        for n, fut in enumerate(as_completed(futures), 1):
            part = fut.result()
            rows.extend(part)
            r = part[0]
            print(f"[{n}/{len(tasks)}] {r['controller']:20s} level {r['level']:6.0f} "
                  f"delay {sum(x['delay'] for x in part) / len(part):8.1f}", flush=True)

    rows.sort(key=lambda r: (r['controller'], r['level'], r['eval_seed']))
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} rows to {args.out}')


if __name__ == '__main__':
    main()
