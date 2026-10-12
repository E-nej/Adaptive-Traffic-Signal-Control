"""End to end: train one tiny run, evaluate it next to a baseline, plot."""

import os

import pandas as pd
import pytest
from conftest import TEST5

from tsc import config
from tsc.experiments.evaluate import evaluate
from tsc.experiments.plot import plot
from tsc.experiments.train import train

pytestmark = pytest.mark.sumo


def test_train_evaluate_plot(tmp_path):
    cfg = config.from_dict({
        'name': 'tiny', 'output_dir': str(tmp_path), 'workers': 1,
        'network': {'path': TEST5},
        'demand': {'duration': 60, 'profile': 'flat', 'min_distance': 50.0},
        'agent': {'hidden': [16, 16], 'batch_size': 8, 'learn_start': 8},
        'train': {'episodes': 2, 'val_every': 2, 'val_levels': [1800.0]},
        'evaluate': {'baselines': ['maxpressure'], 'levels': [1800.0], 'n_seeds': 1},
    })
    (run_dir,) = train(cfg, seeds=[1])
    assert os.path.exists(os.path.join(run_dir, 'config.toml'))
    assert os.path.exists(os.path.join(run_dir, 'run_info.json'))
    assert config.load(os.path.join(run_dir, 'config.toml')).train.seeds == [1]
    log = pd.read_csv(os.path.join(run_dir, 'train_log.csv'))
    assert len(log) == 2 and log['val_delay'].notna().iloc[-1]

    cfg = config.replace(cfg, evaluate={'runs': [os.path.join(str(tmp_path), 'train', 'tiny', 's*')]})
    csv_path = evaluate(cfg)
    df = pd.read_csv(csv_path)
    assert sorted(df['controller']) == ['maxpressure', 'tiny_s1']

    wide = plot([csv_path], str(tmp_path / 'plots'))
    assert os.path.exists(tmp_path / 'plots' / 'delay_vs_demand.png')
    assert ('delay', 'tiny') in wide.index
