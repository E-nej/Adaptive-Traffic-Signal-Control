"""Benchmark summary for step 2I: our controllers next to the published numbers.

    python configs/v6/summarize.py > outputs/v6/benchmarks.md

Two protocols:
  held-out : the run's best (validation) policy on 10 held-out runs of the scenario (ours,
             strict), mean over training seeds; baselines on the same runs;
  RESCO    : the best *training* episode per seed (exploration included), averaged over the
             seeds, as in RESCO's Table 1.
Delay = time loss + departure delay incl. vehicles that never entered (RESCO's definition);
trip time = mean duration of the finished trips.
"""

import glob
import os

import numpy as np
import pandas as pd

MAPS = ['cologne1', 'cologne3', 'cologne8', 'ingolstadt1', 'ingolstadt7', 'ingolstadt21',
        'grid4x4', 'arterial4x4']
# published (delay, trip time): RESCO Table 1 (best episode, 5 seeds) and CoSLight (2024)
PUBLISHED = {
    'cologne1': {'IDQN': (26.05, 43.59), 'IPPO': (55.07, 67.7), 'MPLight': (28.74, 45.85),
                 'FMA2C': (30.12, 47.31)},
    'cologne3': {'IDQN': (23.99, 59.0), 'IPPO': (22.13, 57.45), 'MPLight': (83.65, 102.3),
                 'FMA2C': (25.37, 61.68)},
    'cologne8': {'IDQN': (22.06, 86.02), 'IPPO': (21.49, 85.54), 'MPLight': (60.42, 123.93),
                 'FMA2C': (33.28, 97.53), 'CoSLight': (24.79, 90.46),
                 'CoSLight table MaxPressure': (31.93, 95.96), 'CoSLight table FRAP': (27.5, 88.61),
                 'CoSLight table CoLight': (25.56, 89.72)},
    'ingolstadt1': {'IDQN': (21.48, 35.29), 'IPPO': (19.85, 34.19), 'MPLight': (28.31, 41.07),
                    'FMA2C': (25.36, 39.4)},
    'ingolstadt7': {'IDQN': (31.19, 68.69), 'IPPO': (30.7, 68.34), 'MPLight': (48.21, 76.58),
                    'FMA2C': (48.99, 85.03)},
    'ingolstadt21': {'IDQN': (59.64, 197.23), 'IPPO': (67.65, 205.44), 'MPLight': (78.16, 215.72),
                     'FMA2C': (90.42, 226.5)},
    'grid4x4': {'CoSLight': (46.11, 159.1), 'CoSLight table MaxPressure': (64.01, 175.97),
                'CoSLight table FRAP': (50.02, 161.58), 'CoSLight table CoLight': (51.58, 163.52),
                'CoSLight table IPPO': (56.38, 167.62)},
    'arterial4x4': {'CoSLight': (744.98, 364.21), 'CoSLight table MaxPressure': (952.53, 686.12),
                    'CoSLight table FRAP': (794.13, 383.71), 'CoSLight table CoLight': (776.61, 409.93)},
}
OURS = {'rl2nf': 'rl2 (per-junction DQN), noise-free', 'rl2f': 'phase-wise network, noise-free'}
BASELINES = {'fixed': 'fixed time', 'maxpressure_det': 'max-pressure (detectors)',
             'advanced_mp': 'Advanced-MP', 'maxpressure': 'max-pressure (exact counts)'}


def held_out(name):
    frames = [pd.read_csv(f) for f in glob.glob(f'outputs/eval/v6_bench_{name}*/results.csv')]
    if not frames:
        return {}
    df = pd.concat(frames)
    out = {}
    df['config'] = df.config.str.replace(f'_{name}', '', regex=False)
    for cfg, g in df.groupby('config'):
        per_seed = g.groupby('train_seed', dropna=False)[['delay', 'trip_time']].mean()
        out[cfg] = (per_seed.delay.mean(), per_seed.trip_time.mean(), len(per_seed))
    return out


def resco_protocol(kind, name):
    best = []
    for log in sorted(glob.glob(f'outputs/train/{kind}_{name}/s*/train_log.csv')):
        df = pd.read_csv(log)
        row = df.loc[df.delay.idxmin()]
        best.append((row.delay, row.trip_time))
    if not best:
        return None
    a = np.array(best)
    return a[:, 0].mean(), a[:, 1].mean(), len(best)


def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print('# Step 2I benchmarks: ours vs published\n')
    print('Delay / trip time [s]. Held-out: best-validation policy on 10 held-out runs, mean over '
          'training seeds. RESCO protocol: best training episode per seed, mean over seeds '
          '(how RESCO Table 1 reports).\n')
    for name in MAPS:
        print(f'## {name}\n')
        print('| controller | protocol | delay | trip time | seeds |')
        print('|---|---|---|---|---|')
        ho = held_out(name)
        for kind, label in OURS.items():
            if kind in ho:
                d, t, n = ho[kind]
                print(f'| **{label}** | held-out | {d:.2f} | {t:.2f} | {n} |')
            rp = resco_protocol(kind, name)
            if rp:
                print(f'| **{label}** | RESCO (best episode) | {rp[0]:.2f} | {rp[1]:.2f} | {rp[2]} |')
        for cfg, label in BASELINES.items():
            if cfg in ho:
                d, t, _ = ho[cfg]
                print(f'| {label} | held-out | {d:.2f} | {t:.2f} | – |')
        for label, (d, t) in PUBLISHED.get(name, {}).items():
            src = 'CoSLight (2024)' if label.startswith('CoSLight') else 'RESCO Table 1'
            print(f'| {label} (published) | {src} | {d:.2f} | {t:.2f} | – |')
        print()


if __name__ == '__main__':
    main()
