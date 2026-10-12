"""Plots and a summary table for `tsc evaluate` output.

    tsc plot outputs/eval/baselines/results.csv outputs/eval/step1/results.csv --out outputs/plots/step1

RL configs are first averaged over evaluation seeds per training seed; the
confidence interval is then taken over training seeds (the variance that
matters for "does this method work").  Baselines have no training seed, so
their interval is over evaluation seeds.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

METRICS = {
    'delay': 'Delay per vehicle [s] (lower is better)',
    'waiting_time': 'Waiting time per vehicle [s]',
    'stops': 'Stops per vehicle',
    'throughput': 'Throughput (arrived / generated)',
    'co2_g_per_veh': 'CO$_2$ per vehicle [g]',
    'queue': 'Mean halting vehicles per junction',
}
LABELS = {
    'fixed': 'Fixed time (40 s)', 'webster': 'Fixed time (Webster)', 'actuated': 'Actuated',
    'maxpressure': 'Max-pressure', 'base': 'DQN (paper EA)', 'full': 'VA full (paper VA)',
    'full_nocomm': 'VA − comm', 'queue': 'VA, queue reward',
    'det_queue': 'VA, detector queue reward', 'det_queue_nocomm': 'VA, detector queue − comm',
}


def ci95(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def aggregate(df, metric):
    df = df.copy()
    df['train_seed'] = df['train_seed'].fillna('').astype(str)
    df['group'] = np.where(df['train_seed'] == '', df['controller'], df['config'])
    out = []
    for (group, level), g in df.groupby(['group', 'level']):
        if (g['train_seed'] != '').any():
            per_run = g.groupby('train_seed')[metric].mean()
            out.append((group, level, per_run.mean(), ci95(per_run), len(per_run)))
        else:
            out.append((group, level, g[metric].mean(), ci95(g[metric]), len(g)))
    return pd.DataFrame(out, columns=['group', 'level', 'mean', 'ci', 'n'])


def plot(csv_files, out):
    """Metric-vs-demand plots (95 % CI) and summary.csv / summary.md in `out`."""
    os.makedirs(out, exist_ok=True)
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    table = []
    for metric, ylabel in METRICS.items():
        agg = aggregate(df, metric)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for group, g in agg.groupby('group'):
            g = g.sort_values('level')
            ax.errorbar(g['level'], g['mean'], yerr=g['ci'], marker='o', capsize=3,
                        label=f"{LABELS.get(group, group)} (n={g['n'].iloc[0]})")
        ax.set_xlabel('Peak demand [veh/h, whole network]')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'{metric}_vs_demand.png'), dpi=150)
        plt.close(fig)
        agg['metric'] = metric
        table.append(agg)

    table = pd.concat(table)
    table['value'] = table.apply(lambda r: f"{r['mean']:.2f} ± {r['ci']:.2f}", axis=1)
    wide = table.pivot_table(index=['metric', 'group'], columns='level', values='value', aggfunc='first')
    wide.to_csv(os.path.join(out, 'summary.csv'))
    with open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write(wide.to_markdown())
    print(wide.loc['delay'].to_string())
    return wide
