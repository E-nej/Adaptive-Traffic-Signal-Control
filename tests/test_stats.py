"""Paired comparison against a reference controller."""

import pandas as pd

from tsc.experiments.stats import paired, paired_table


def frame():
    rows = []
    for seed in range(6):
        rows.append({'controller': 'ref', 'config': 'ref', 'train_seed': None,
                     'level': 100, 'eval_seed': seed, 'delay': 10.0 + seed})
        for run in (0, 1):   # RL config: two training seeds, 20 % worse on average
            rows.append({'controller': f'rl_s{run}', 'config': 'rl', 'train_seed': run,
                         'level': 100, 'eval_seed': seed,
                         'delay': (10.0 + seed) * (1.1 if run == 0 else 1.3)})
    return pd.DataFrame(rows)


def test_paired_relative_difference():
    p = paired(frame(), 'ref')
    assert list(p['group']) == ['rl']
    assert abs(p['rel_mean'].iloc[0] - 20.0) < 1e-9     # mean over runs first: 1.2 × ref
    assert p['n'].iloc[0] == 6


def test_paired_table_marks_significance():
    table = paired_table(frame(), 'ref')
    assert table.loc['rl', 100].startswith('+20.0 %')
