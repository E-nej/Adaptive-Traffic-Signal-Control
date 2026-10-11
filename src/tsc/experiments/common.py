"""Helpers shared by the experiments."""

import datetime
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import tsc
from tsc import config as config_mod
from tsc.network import Network


def network_for(cfg):
    """The prepared network of an experiment (generated files under <output_dir>/networks)."""
    return Network(cfg.network, os.path.join(cfg.output_dir, 'networks'))


def sim_kwargs(cfg):
    """Simulation options for Python-controlled junctions."""
    return dict(decision_interval=cfg.signals.decision_interval, max_green=cfg.signals.max_green,
                max_red=cfg.signals.max_red, e2_length=cfg.observation.e2_length)


def save_run_info(cfg, out_dir):
    """Write the resolved config and provenance (versions, commit, time) next to the results."""
    os.makedirs(out_dir, exist_ok=True)
    config_mod.save(cfg, os.path.join(out_dir, 'config.toml'))
    info = {
        'tsc_version': tsc.__version__,
        'git_commit': _git_commit(),
        'sumo_version': _sumo_version(),
        'started': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(os.path.join(out_dir, 'run_info.json'), 'w') as f:
        json.dump(info, f, indent=2)


def _git_commit():
    try:
        out = subprocess.run(['git', 'describe', '--always', '--dirty', '--tags'],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _sumo_version():
    try:
        from importlib.metadata import version
        return version('eclipse-sumo')
    except Exception:
        return None


def run_parallel(fn, tasks, workers):
    """Yield fn(task) results as they finish; in-process when workers <= 1 (easier debugging)."""
    os.environ.setdefault('TSC_TF_THREADS', '1')   # inherited by the worker processes
    if workers <= 1 or len(tasks) <= 1:
        for t in tasks:
            yield fn(t)
        return
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        futures = [pool.submit(fn, t) for t in tasks]
        for fut in as_completed(futures):
            yield fut.result()
