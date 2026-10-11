import glob
import os

import pytest
from conftest import ROOT

from tsc import config, seeds
from tsc.experiments.runs import legacy_config

CONFIGS = sorted(glob.glob(os.path.join(ROOT, 'configs', '**', '*.toml'), recursive=True))


@pytest.mark.parametrize('path', CONFIGS, ids=lambda p: os.path.relpath(p, ROOT))
def test_shipped_configs_load(path):
    cfg = config.load(path)
    assert cfg.name


def test_unknown_key_is_an_error(tmp_path):
    p = tmp_path / 'c.toml'
    p.write_text('[observation]\nmod = "grid"\n')
    with pytest.raises(ValueError, match='mod'):
        config.load(p)


def test_overrides_and_round_trip(tmp_path):
    p = tmp_path / 'c.toml'
    p.write_text('name = "x"\n')
    cfg = config.load(p, ['observation.noise=0.1', 'observation.mode=detector',
                          'signals.max_green=60', 'train.seeds=[1, 2]'])
    assert cfg.observation.noise == 0.1 and cfg.observation.mode == 'detector'
    assert cfg.signals.max_green == 60 and cfg.train.seeds == [1, 2]
    out = tmp_path / 'resolved.toml'
    config.save(cfg, out)
    assert config.load(out) == cfg


def test_none_values_are_omitted_and_restored(tmp_path):
    cfg = config.Config()
    assert cfg.signals.max_green is None
    out = tmp_path / 'r.toml'
    config.save(cfg, out)
    assert config.load(out) == cfg


def test_seed_ranges_do_not_overlap():
    for s in range(seeds.MAX_TRAIN_SEED + 1):
        assert seeds.train(s, seeds.TRAIN_STRIDE - 1) < seeds.VALIDATION
    with pytest.raises(ValueError):
        seeds.train(7, 0)
    assert seeds.validation(0) == 700001 and seeds.calibration(0) == 800000
    assert seeds.evaluation(9) == 900009


def test_legacy_step2_run_config():
    meta = {'config': 'det_queue', 'seed': 2, 'obs': 'detector', 'noise': 0.05, 'dropout': 0.02,
            'max_green': 60, 'max_red': 90, 'episodes': 150}
    cfg = legacy_config(meta)
    assert cfg.name == 'det_queue' and cfg.train.seeds == [2]
    assert cfg.observation.mode == 'detector' and cfg.observation.comm
    assert cfg.observation.noise == 0.05 and cfg.observation.dropout == 0.02
    assert cfg.signals.max_green == 60 and cfg.signals.max_red == 90
    assert cfg.reward.kind == 'det_queue' and cfg.agent.double


def test_legacy_step1_run_config_defaults():
    cfg = legacy_config({'config': 'base', 'seed': 0, 'obs': 'grid', 'max_green': None})
    assert cfg.observation.mode == 'grid' and not cfg.observation.comm
    assert cfg.observation.noise == 0.0 and cfg.signals.max_red is None
    assert cfg.agent.target_update == 'hard' and not cfg.agent.double and cfg.reward.kind == 'wait'
