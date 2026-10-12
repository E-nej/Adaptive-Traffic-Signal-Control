"""Diagnostics of trained policies.

    tsc diagnose messages configs/step2/evaluate.toml --runs "outputs/train/step2_full/s*"

messages: evaluate communicating runs with real neighbour messages, with all
messages set to 0, and with messages of random *other* junctions (see
Observer.message_mode).  Communication carries neighbour-specific information
only if 'wrong' is clearly worse than 'real'; if 'wrong' ≈ 'real' while
'zero' is much worse, the messages are only used as a network-load signal.

Output: <out>/messages.csv and summary.md (delay relative to 'real', paired).
"""

import csv
import os

import pandas as pd

from tsc import seeds as seeds_mod
from tsc.experiments.common import network_for, run_parallel, save_run_info, sim_kwargs
from tsc.experiments.evaluate import find_runs, prepare, route_path
from tsc.experiments.runs import load_policy
from tsc.experiments.stats import paired_table
from tsc.simulation import run_episode

MODES = ('real', 'zero', 'wrong')


def default_out(cfg):
    return os.path.join(cfg.output_dir, 'diagnose', cfg.name)


def _messages_task(task):
    cfg, run_dir, mode, level, work = task
    network = network_for(cfg)
    controller, run_cfg = load_policy(run_dir, network, which=cfg.evaluate.which)
    if not run_cfg.observation.comm:
        raise ValueError(f'{run_dir} was trained without communication')
    controller.obs.set_message_mode(mode, seed=0)
    kwargs = sim_kwargs(cfg)
    if cfg.signals.max_green is None and cfg.signals.max_red is None:
        kwargs.update(max_green=run_cfg.signals.max_green, max_red=run_cfg.signals.max_red)
    name = f'{run_cfg.name}_s{run_cfg.train.seeds[0]}'
    rows = []
    for i in range(cfg.evaluate.n_seeds):
        seed = seeds_mod.evaluation(i)
        out_dir = os.path.join(work, 'tmp', f'{name}_{mode}_L{int(level)}_s{seed}')
        m = run_episode(network, controller, route_path(work, level, seed), cfg.demand.duration,
                        out_dir, seed=seed, **kwargs)
        # 'config' = mode and 'train_seed' = run, so stats.paired averages over runs per scenario
        rows.append({'controller': f'{name}_{mode}', 'config': mode, 'train_seed': run_dir,
                     'level': level, 'eval_seed': seed, **m})
    return rows


def messages(cfg, out_dir=None, workers=None):
    work = out_dir or default_out(cfg)
    save_run_info(cfg, work)
    run_dirs = find_runs(cfg.evaluate.runs)
    if not run_dirs:
        raise FileNotFoundError(f'No run directories match {cfg.evaluate.runs}')
    network = network_for(cfg)
    prepare(network, cfg, work)
    tasks = [(cfg, d, mode, level, work)
             for d in run_dirs for mode in MODES for level in cfg.evaluate.levels]
    rows = []
    for part in run_parallel(_messages_task, tasks, workers or cfg.workers):
        rows.extend(part)
    rows.sort(key=lambda r: (r['train_seed'], r['config'], r['level'], r['eval_seed']))
    out_csv = os.path.join(work, 'messages.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    table = paired_table(pd.DataFrame(rows), ref='real')
    with open(os.path.join(work, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('Delay relative to real messages (paired over scenarios, * = Wilcoxon p < 0.05)\n\n')
        f.write(table.to_markdown())
    print(table.to_string())
    print(f'wrote {len(rows)} rows to {out_csv}')
    return out_csv
