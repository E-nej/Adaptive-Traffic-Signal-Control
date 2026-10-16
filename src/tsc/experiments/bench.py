"""End-to-end training throughput under CPU / GPU worker mixes.

    tsc bench configs/step1/full.toml --mix cpu:5 --mix gpu:2,cpu:4 --episodes 10 --repeats 3

For every mix and repeat, `gpu:k,cpu:m` starts k GPU and m CPU training runs at the same
time (a fresh process each, so every run can pick its own device).  Each run trains
`episodes` full episodes (one validation at the end, as in real training).  Measured:

    episodes_per_min   steady-state throughput: Σ over runs of 60 / median episode time
                       (episode times from train_log.csv, the final episode with its
                       validation excluded)
    wall_episodes_per_min   all episodes / wall time, including start-up and validation
    episode_s_median / _iqr per device, peak GPU memory and utilisation (nvidia-smi or
                       rocm-smi, when present), mean CPU utilisation and clock

The summary compares each mix with the best CPU-only mix (S = throughput ratio), the
number that decides whether a GPU is worth using on a machine (docs/GPU.md: S >= 1.3).
Results go to <out>/bench.json.  Benchmark runs reuse training seeds 0..6 (scenarios only
matter for the workload, which they keep realistic).
"""

import csv
import json
import os
import shutil
import subprocess
import threading
import time

import numpy as np

from tsc import config as config_mod
from tsc import seeds as seeds_mod
from tsc.experiments.common import network_for, run_parallel, software_stack


def parse_mix(text):
    """'gpu:2,cpu:4' -> {'gpu': 2, 'cpu': 4}."""
    mix = {'gpu': 0, 'cpu': 0}
    for part in text.split(','):
        device, sep, n = part.strip().partition(':')
        if device not in mix or not sep or int(n) < 0:
            raise ValueError(f"Mix must look like 'gpu:2,cpu:4', got {text!r}")
        mix[device] += int(n)
    if not sum(mix.values()):
        raise ValueError(f'Empty mix: {text!r}')
    return mix


def mix_name(mix):
    return '+'.join(f'{d}:{n}' for d, n in mix.items() if n)


def mix_dir(mix):
    """Directory name of a mix ('gpu2_cpu4'): SUMO misreads ':' in paths as host:port."""
    return '_'.join(f'{d}{n}' for d, n in mix.items() if n)


def bench(cfg, mixes, episodes=10, repeats=3, out_dir=None):
    out_dir = out_dir or os.path.join(cfg.output_dir, 'bench', cfg.name)
    os.makedirs(out_dir, exist_ok=True)
    network_for(cfg)
    results = []
    for mix in mixes:
        for rep in range(repeats):
            r = run_mix(cfg, mix, episodes, os.path.join(out_dir, f'{mix_dir(mix)}_r{rep}'))
            r['repeat'] = rep
            results.append(r)
            print(f"[bench] {r['mix']:16s} repeat {rep}: {r['episodes_per_min']:6.2f} episodes/min "
                  f"(wall {r['wall_episodes_per_min']:.2f}), median episode "
                  f"{json.dumps(r['episode_s_median'])}", flush=True)
    summary = summarise(results)
    report = {'config': config_mod.to_dict(cfg), 'episodes': episodes, 'repeats': repeats,
              'stack': software_stack(), 'runs': results, 'summary': summary}
    path = os.path.join(out_dir, 'bench.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=1)
    print_summary(summary)
    print(f'wrote {path}')
    return report


def run_mix(cfg, mix, episodes, out_dir):
    tasks = []
    for device in ('gpu', 'cpu'):
        for _ in range(mix[device]):
            seed = len(tasks) % (seeds_mod.MAX_TRAIN_SEED + 1)
            run_cfg = config_mod.replace(cfg, train={'device': device, 'seeds': [seed],
                                                     'episodes': episodes, 'val_every': episodes})
            tasks.append((run_cfg, os.path.join(out_dir, f'{device}{len(tasks)}')))
    from tsc.experiments.train import _train_task
    sampler = Sampler()
    sampler.start()
    t0 = time.time()
    list(run_parallel(_train_task, tasks, workers=len(tasks), fresh=True))
    wall = time.time() - t0
    usage = sampler.stop()

    times = {'gpu': [], 'cpu': []}
    per_run_rate = []
    n_episodes = 0
    gpu_peak = []                                   # PyTorch's peak allocation per GPU run
    for run_cfg, run_dir in tasks:
        with open(os.path.join(run_dir, 'run_info.json')) as f:
            peak = json.load(f).get('gpu_peak_memory_mb')
        if peak is not None:
            gpu_peak.append(peak)
        with open(os.path.join(run_dir, 'train_log.csv')) as f:
            secs = [float(row['seconds']) for row in csv.DictReader(f)]
        n_episodes += len(secs)
        steady = secs[:-1] or secs                  # the last episode includes validation
        times[run_cfg.train.device].extend(steady)
        per_run_rate.append(60.0 / float(np.median(steady)))
    return {
        'mix': mix_name(mix),
        'workers': dict(mix),
        'episodes_per_min': round(sum(per_run_rate), 3),
        'wall_s': round(wall, 1),
        'wall_episodes_per_min': round(n_episodes / wall * 60.0, 3),
        'episode_s_median': {d: round(float(np.median(t)), 2) for d, t in times.items() if t},
        'episode_s_iqr': {d: round(float(np.subtract(*np.percentile(t, [75, 25]))), 2)
                          for d, t in times.items() if t},
        'gpu_peak_memory_mb_per_run': max(gpu_peak) if gpu_peak else None,
        **usage,
    }


def summarise(results):
    by_mix = {}
    for r in results:
        by_mix.setdefault(r['mix'], []).append(r)
    rows = []
    for name, rs in by_mix.items():
        rates = [r['episodes_per_min'] for r in rs]
        rows.append({'mix': name, 'cpu_only': not rs[0]['workers']['gpu'],
                     'episodes_per_min_median': round(float(np.median(rates)), 3),
                     'episodes_per_min_range': [min(rates), max(rates)]})
    cpu_best = max((r['episodes_per_min_median'] for r in rows if r['cpu_only']), default=None)
    for r in rows:
        r['speedup_vs_best_cpu'] = (round(r['episodes_per_min_median'] / cpu_best, 3)
                                    if cpu_best else None)
    return rows


def print_summary(rows):
    print(f"{'mix':18s} {'episodes/min':>13s} {'range':>17s} {'S vs best CPU':>14s}")
    for r in rows:
        lo, hi = r['episodes_per_min_range']
        s = '' if r['speedup_vs_best_cpu'] is None else f"{r['speedup_vs_best_cpu']:.2f}x"
        print(f"{r['mix']:18s} {r['episodes_per_min_median']:13.2f} {f'{lo:.2f}-{hi:.2f}':>17s} {s:>14s}")


# ----------------------------------------------------------------------
# Resource sampling (best effort: missing tools only leave fields empty)
# ----------------------------------------------------------------------

class Sampler:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.samples = {'cpu_util': [], 'cpu_mhz': [], 'gpu_mem_mb': [], 'gpu_util': []}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        s = self.samples
        mean = lambda xs: round(float(np.mean(xs)), 1) if xs else None   # noqa: E731
        return {'cpu_util_mean': mean(s['cpu_util']), 'cpu_mhz_mean': mean(s['cpu_mhz']),
                'gpu_mem_peak_mb': max(s['gpu_mem_mb'], default=None),
                'gpu_util_mean': mean(s['gpu_util'])}

    def _run(self):
        prev = _cpu_times()
        while not self._stop.wait(self.interval):
            cur = _cpu_times()
            if prev and cur and cur[1] > prev[1]:
                self.samples['cpu_util'].append(100.0 * (1 - (cur[0] - prev[0]) / (cur[1] - prev[1])))
            prev = cur
            mhz = _cpu_mhz()
            if mhz:
                self.samples['cpu_mhz'].append(mhz)
            mem, util = _gpu_usage()
            if mem is not None:
                self.samples['gpu_mem_mb'].append(mem)
            if util is not None:
                self.samples['gpu_util'].append(util)


def _cpu_times():
    """(idle, total) jiffies from /proc/stat, or None."""
    try:
        with open('/proc/stat') as f:
            vals = [int(x) for x in f.readline().split()[1:]]
        return vals[3] + vals[4], sum(vals)
    except (OSError, ValueError, IndexError):
        return None


def _cpu_mhz():
    try:
        with open('/proc/cpuinfo') as f:
            mhz = [float(line.split(':')[1]) for line in f if line.startswith('cpu MHz')]
        return float(np.mean(mhz)) if mhz else None
    except (OSError, ValueError):
        return None


def _gpu_usage():
    """(memory used in MB summed over GPUs, mean utilisation %) or (None, None)."""
    try:
        if shutil.which('nvidia-smi'):
            out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                  '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=5).stdout
            rows = [[float(x) for x in line.split(',')] for line in out.strip().splitlines() if line]
            if rows:
                return sum(r[0] for r in rows), float(np.mean([r[1] for r in rows]))
        if shutil.which('rocm-smi'):
            out = subprocess.run(['rocm-smi', '--showmeminfo', 'vram', '--showuse', '--json'],
                                 capture_output=True, text=True, timeout=5).stdout
            cards = json.loads(out or '{}')
            mem = [float(v) / 2**20 for c in cards.values() for k, v in c.items()
                   if 'VRAM Total Used Memory' in k]
            use = [float(v) for c in cards.values() for k, v in c.items() if 'GPU use' in k]
            if mem or use:
                return (sum(mem) if mem else None), (float(np.mean(use)) if use else None)
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    return None, None
