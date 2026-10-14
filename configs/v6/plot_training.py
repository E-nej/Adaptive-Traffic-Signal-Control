"""Training curves: reward, episode delay and validation delay against the episode.

    python configs/v6/plot_training.py                                # outputs/train/*
    python configs/v6/plot_training.py "outputs_archive/*/outputs/train/*" --out outputs_archive/plots

One figure per run (a run = a directory of s0, s1, ... seeds with train_log.csv): the mean over
seeds, a min-max band across them, and the demand level / epsilon schedule. Writes
<out>/<group>/<run>.png, where <group> is the directory two levels above the run (e.g. the
archived worktree), and an index.html showing every figure.
"""

import argparse
import glob
import html
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PANELS = [('mean_reward', 'mean reward per step'),
          ('delay', 'training-episode delay [s]'),
          ('val_delay', 'validation delay [s]')]


def load(run):
    logs = {}
    for seed in sorted(glob.glob(os.path.join(run, 's*'))):
        path = os.path.join(seed, 'train_log.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            if len(df):
                logs[os.path.basename(seed)] = df.set_index('episode')
    return logs


def band(ax, logs, col, smooth):
    series = {s: df[col].dropna() for s, df in logs.items() if col in df and df[col].notna().any()}
    if not series:
        ax.text(0.5, 0.5, 'not logged', ha='center', va='center', transform=ax.transAxes, color='grey')
        return
    if smooth > 1 and col != 'val_delay':   # validation is sparse already
        series = {s: v.rolling(smooth, min_periods=1).mean() for s, v in series.items()}
    wide = pd.DataFrame(series)
    for s, v in series.items():
        ax.plot(v.index, v.values, lw=0.6, alpha=0.35)
    marker = 'o' if col == 'val_delay' else None
    ax.plot(wide.index, wide.mean(axis=1), color='black', lw=1.6, marker=marker, ms=3,
            label=f'mean of {len(series)} seed{"s" * (len(series) > 1)}')
    if len(series) > 1:
        ax.fill_between(wide.index, wide.min(axis=1), wide.max(axis=1), color='grey', alpha=0.2,
                        label='min-max over seeds')
    if col == 'val_delay':
        best = wide.mean(axis=1).idxmin()
        ax.axvline(best, color='tab:green', ls='--', lw=1, label=f'best mean val (ep {best})')


def plot_run(run, logs, out, smooth):
    fig, axes = plt.subplots(1, len(PANELS) + 1, figsize=(17, 3.6))
    for ax, (col, title) in zip(axes, PANELS):
        band(ax, logs, col, smooth)
        ax.set_title(title + (f' ({smooth}-ep mean)' if smooth > 1 and col != 'val_delay' else ''),
                     fontsize=9)
        ax.set_xlabel('episode')
        ax.grid(alpha=0.3)
    axes[2].legend(fontsize=7, loc='upper right')
    ax = axes[-1]
    df = next(iter(logs.values()))
    ax.plot(df.index, df['epsilon'], color='tab:blue', label='epsilon')
    ax.set_ylabel('epsilon', color='tab:blue')
    ax.set_xlabel('episode')
    ax.set_title('schedule (seed ' + next(iter(logs)) + ')', fontsize=9)
    if 'level' in df:
        ax2 = ax.twinx()
        ax2.step(df.index, df['level'], color='tab:orange', where='post', label='demand level')
        ax2.set_ylabel('demand level', color='tab:orange')
    fig.suptitle(os.path.basename(run), fontsize=11, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('runs', nargs='?', default='outputs/train/*', help='glob of run directories')
    p.add_argument('--out', default='outputs/plots/training')
    p.add_argument('--smooth', type=int, default=5, help='rolling mean over episodes (1 = off)')
    args = p.parse_args(argv)

    groups = {}
    for run in sorted(glob.glob(args.runs)):
        logs = load(run)
        if not logs:
            continue
        group = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(run)))) or 'runs'
        name = os.path.basename(run)
        os.makedirs(os.path.join(args.out, group), exist_ok=True)
        plot_run(run, logs, os.path.join(args.out, group, name + '.png'), args.smooth)
        groups.setdefault(group, []).append(name)
    if not groups:
        sys.exit(f'no train_log.csv under {args.runs}')

    parts = ['<!doctype html><meta charset="utf-8"><title>Training curves</title>'
             '<style>body{font-family:sans-serif;margin:16px}img{max-width:100%;border:1px solid #ddd;'
             'margin:4px 0 16px}a{margin-right:10px}</style><h1>Training curves</h1>'
             '<p>' + ' '.join(f'<a href="#{g}">{g}</a>' for g in groups) + '</p>']
    for g, names in groups.items():
        parts.append(f'<h2 id="{g}">{g} ({len(names)} runs)</h2>')
        parts += [f'<img src="{g}/{html.escape(n)}.png" alt="{html.escape(n)}" loading="lazy">'
                  for n in names]
    with open(os.path.join(args.out, 'index.html'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))
    print(f'{sum(map(len, groups.values()))} runs -> {args.out}/index.html')


if __name__ == '__main__':
    main()
