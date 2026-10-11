"""Paired comparisons of evaluation results.

All controllers run on the same scenarios (level × evaluation seed), so the
right test is paired: per scenario, the difference to a reference controller.
RL configs are first averaged over their training seeds per scenario.
"""

import numpy as np
import pandas as pd
from scipy import stats


def per_scenario(df, metric='delay'):
    """metric per (group, level, eval_seed); group = config for RL runs, controller otherwise."""
    df = df.copy()
    df['train_seed'] = df['train_seed'].fillna('').astype(str)
    df['group'] = np.where(df['train_seed'] == '', df['controller'], df['config'])
    return df.groupby(['group', 'level', 'eval_seed'])[metric].mean().reset_index()


def paired(df, ref, metric='delay'):
    """Relative difference to `ref` per group and level.

    Returns a long DataFrame (group, level, rel_mean [%], rel_ci [95 %, %], p [Wilcoxon], n).
    """
    s = per_scenario(df, metric)
    base = s[s['group'] == ref].set_index(['level', 'eval_seed'])[metric]
    if base.empty:
        raise ValueError(f'Reference {ref!r} not found; groups: {sorted(s["group"].unique())}')
    rows = []
    for (group, level), g in s[s['group'] != ref].groupby(['group', 'level']):
        g = g.set_index(['level', 'eval_seed'])[metric]
        common = g.index.intersection(base.index)
        if len(common) < 2:
            continue
        diff = g.loc[common].values - base.loc[common].values
        rel = diff / base.loc[common].values * 100
        ci = stats.t.ppf(0.975, len(rel) - 1) * rel.std(ddof=1) / np.sqrt(len(rel))
        p = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
        rows.append((group, level, rel.mean(), ci, p, len(rel)))
    return pd.DataFrame(rows, columns=['group', 'level', 'rel_mean', 'rel_ci', 'p', 'n'])


def paired_table(df, ref, metric='delay', alpha=0.05):
    """Wide table of '+3.1 % ± 2.0 *' per group × level (* = Wilcoxon p < alpha)."""
    p = paired(df, ref, metric)
    p['value'] = [f"{m:+.1f} % ± {c:.1f}{' *' if pv < alpha else ''}"
                  for m, c, pv in zip(p['rel_mean'], p['rel_ci'], p['p'])]
    return p.pivot(index='group', columns='level', values='value')
