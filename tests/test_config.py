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
    assert seeds.VALIDATION + 1000 < seeds.acceptance(0) < seeds.acceptance(999) < seeds.CALIBRATION


def test_retired_keys_are_dropped(tmp_path):
    """Keys of removed features (still in earlier runs' config.toml) load and are ignored,
    elsewhere they are still unknown keys."""
    p = tmp_path / 'c.toml'
    p.write_text('[signals]\nselect_interval = 90\n')
    assert config.load(p) == config.Config()
    p.write_text('[observation]\nselect_interval = 90\n')
    with pytest.raises(ValueError, match='select_interval'):
        config.load(p)


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


def test_file_lock_is_exclusive_between_processes(tmp_path):
    """Two processes building the same shared file: the second waits for the first."""
    import subprocess
    import sys
    import time
    from tsc.sumo_env import file_lock
    lock = tmp_path / 'x.lock'
    code = ('import sys, time; from tsc.sumo_env import file_lock\n'
            f'with file_lock(r"{lock}", poll=0.05):\n'
            f'    open(r"{tmp_path / "order.txt"}", "a").write("child\\n")\n')
    with file_lock(str(lock)):
        child = subprocess.Popen([sys.executable, '-c', code])
        time.sleep(1.0)
        (tmp_path / 'order.txt').open('a').write('parent\n')
    assert child.wait(timeout=60) == 0
    assert (tmp_path / 'order.txt').read_text().split() == ['parent', 'child']
    assert not lock.exists()


def test_file_demand_keeps_evaluation_files_apart(tmp_path):
    """generator 'file': one file is always used; with several, evaluation scenarios only
    use the last `eval_files` (in natural order: 2 before 10) and nothing else does."""
    from tsc.demand import route_file_for
    for i in range(1, 13):
        (tmp_path / f'g_{i}.rou.xml').write_text('<routes/>')
    d = config.DemandConfig(generator='file', route_files=[str(tmp_path / 'g_*.rou.xml')],
                            eval_files=3)
    names = lambda seeds_: {os.path.basename(route_file_for(d, s)) for s in seeds_}
    assert names(seeds.evaluation(i) for i in range(10)) == {'g_10.rou.xml', 'g_11.rou.xml',
                                                             'g_12.rou.xml'}
    used = names([seeds.train(s, e) for s in range(3) for e in range(50)]
                 + [seeds.validation(i) for i in range(5)] + [seeds.calibration(i) for i in range(5)])
    assert used == {f'g_{i}.rou.xml' for i in range(1, 10)}
    one = config.DemandConfig(generator='file', route_files=[str(tmp_path / 'g_1.rou.xml')])
    assert route_file_for(one, seeds.evaluation(0)) == route_file_for(one, 5)
    with pytest.raises(ValueError):
        route_file_for(config.DemandConfig(generator='file', route_files=d.route_files), 0)
    with pytest.raises(FileNotFoundError):
        route_file_for(config.DemandConfig(generator='file', route_files=['nope/*.xml']), 0)
