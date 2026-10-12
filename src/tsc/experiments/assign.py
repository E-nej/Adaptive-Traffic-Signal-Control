"""Best fixed controller per junction (step 2G baseline, no learning).

    tsc assign configs/select/corridor_assign.toml

For every evaluation level, a greedy search (coordinate descent) over the
junctions: starting from the green wave everywhere, each junction in turn tries
every expert of [signals] experts and keeps the one with the lowest
mean delay on the calibration scenarios (never the evaluation scenarios), for
`passes` rounds.  Like a traffic engineer's time-of-day plan, the result is one
assignment per demand level: {level: {tl_id: expert index}}.

Output: <out>/assignment.json (the evaluation's 'select_assigned' baseline reads
it through evaluate.assignment) and search_log.csv.
"""

import csv
import json
import os

import numpy as np

from tsc import seeds as seeds_mod
from tsc.controllers.select import FixedSelection
from tsc.demand import is_directional
from tsc.experiments.common import network_for, run_parallel, save_run_info, sim_kwargs
from tsc.experiments.evaluate import DIRECTIONS, prepare, route_path
from tsc.plans import load_plans
from tsc.simulation import run_episode

DEFAULT = 0          # green wave


def default_out(cfg):
    return os.path.join(cfg.output_dir, 'assign', cfg.name)


def _scenarios(cfg):
    """Calibration scenarios used for the search: (seed, tidal direction), both directions."""
    flips = [False, True] if is_directional(cfg.demand) else [None]
    return [(seeds_mod.calibration(i), flip)
            for i in range(cfg.evaluate.n_calibration) for flip in flips]


def _run(task):
    """Mean delay of one assignment at one level over the search scenarios."""
    cfg, level, assignment, work, plan_files = task
    network = network_for(cfg)
    kwargs = selection_kwargs(cfg)
    delays = []
    for seed, flip in _scenarios(cfg):
        ctrl = make_fixed(network, cfg, assignment)
        ctrl.set_plans(load_plans(plan_files[flip], network))
        ctrl.reseed(seed)
        route = route_path(work, level, seed, DIRECTIONS.get(flip, ''))
        m = run_episode(network, ctrl, route, cfg.demand.duration,
                        os.path.join(work, 'tmp', f'L{int(level)}_{os.getpid()}'), seed=seed,
                        **kwargs)
        delays.append(m['delay'])
    return level, json.dumps(assignment, sort_keys=True), float(np.mean(delays))


def selection_kwargs(cfg):
    kwargs = sim_kwargs(cfg)
    kwargs['decision_interval'] = 1         # plan experts switch on the plan's second
    return kwargs


def make_fixed(network, cfg, assignment):
    s, o = cfg.signals, cfg.observation
    return FixedSelection(network, assignment, default=DEFAULT, interval=s.decision_interval,
                          max_extension=s.max_extension, noise=o.noise, dropout=o.dropout,
                          e2_length=o.e2_length, experts=s.experts,
                          actuated_loops=s.actuated_loops)


def assign(cfg, out_dir=None, workers=None, passes=2):
    work = out_dir or default_out(cfg)
    save_run_info(cfg, work)
    network = network_for(cfg)
    plans = prepare(network, cfg, work, plan_names=('greenwave',))
    plan_files = {level: {flip: files['greenwave'] for flip, files in by_dir.items()}
                  for level, by_dir in plans.items()}
    levels = list(cfg.evaluate.levels)
    best = {level: {} for level in levels}
    score = {}
    log = []

    def evaluate_all(candidates):
        tasks = [(cfg, level, a, work, plan_files[level]) for level, a in candidates]
        results = {}
        for level, key, delay in run_parallel(_run, tasks, workers or cfg.workers):
            results[(level, key)] = delay
            log.append({'level': level, 'assignment': key, 'delay': delay})
        return results

    base = evaluate_all([(level, {}) for level in levels])
    for level in levels:
        score[level] = base[(level, json.dumps({}, sort_keys=True))]
        print(f'level {level:6.0f} all green wave: {score[level]:.1f}', flush=True)
    for p in range(passes):
        changed = False
        for tl_id in network.tl_ids:
            cands = []
            for level in levels:
                current = best[level].get(tl_id, DEFAULT)
                for k in range(len(cfg.signals.experts)):
                    if k != current:
                        a = dict(best[level])
                        a[tl_id] = k
                        if k == DEFAULT:
                            a.pop(tl_id)
                        cands.append((level, a))
            results = evaluate_all(cands)
            for level, a in cands:
                d = results[(level, json.dumps(a, sort_keys=True))]
                if d < score[level] - 1e-9:
                    score[level], best[level] = d, a
                    changed = True
            print(f'pass {p} {tl_id}: ' + ', '.join(
                f'{int(level)}: {cfg.signals.experts[best[level].get(tl_id, DEFAULT)]} '
                f'{score[level]:.1f}'
                for level in levels), flush=True)
        if not changed:
            break

    with open(os.path.join(work, 'assignment.json'), 'w') as f:
        json.dump({str(level): best[level] for level in levels}, f, indent=2, sort_keys=True)
    with open(os.path.join(work, 'search_log.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['level', 'assignment', 'delay'])
        w.writeheader()
        w.writerows(log)
    print(f'wrote {os.path.join(work, "assignment.json")}')
    return best
