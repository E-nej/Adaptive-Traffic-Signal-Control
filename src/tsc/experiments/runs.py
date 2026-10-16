"""Trained run directories: their config and their policy.

A run directory holds `config.toml` (the resolved config of that run, with
exactly one training seed) and `{best,final}_<tl>.pt` (PyTorch state dicts; runs trained
with TensorFlow, `*.weights.h5`, cannot be loaded any more).

Runs trained with the pre-package scripts (v0.2-era `train.py`) hold a
`config.json` instead: the argparse values plus one of the named ablation
configs below.  They are converted to a Config on load, so they can be
evaluated with the current code.
"""

import json
import os

from tsc import config as config_mod
from tsc.config import Config

# The ablation configs of the pre-package train.py (CONFIGS there).
LEGACY_CONFIGS = {
    'base': dict(comm=False, double=False, target_update='hard', reward='wait'),
    'full': dict(comm=True, double=True, target_update='soft', reward='multi'),
    'full_nocomm': dict(comm=False, double=True, target_update='soft', reward='multi'),
    'queue': dict(comm=True, double=True, target_update='soft', reward='queue'),
    'det_queue': dict(comm=True, double=True, target_update='soft', reward='det_queue'),
    'det_queue_nocomm': dict(comm=False, double=True, target_update='soft', reward='det_queue'),
}

# Observer noise of an evaluated policy is seeded with train seed + this offset,
# so evaluation noise differs from training noise (as in the pre-package code).
EVAL_NOISE_SEED_OFFSET = 12345


def is_run_dir(path):
    return any(os.path.exists(os.path.join(path, f)) for f in ('config.toml', 'config.json'))


def load_run_config(run_dir):
    toml_path = os.path.join(run_dir, 'config.toml')
    if os.path.exists(toml_path):
        return config_mod.load(toml_path)
    with open(os.path.join(run_dir, 'config.json')) as f:
        return legacy_config(json.load(f))


def legacy_config(meta):
    """Config of a run trained by the pre-package train.py (from its config.json)."""
    named = LEGACY_CONFIGS[meta['config']]
    cfg = Config(name=meta['config'])
    return config_mod.replace(
        cfg,
        demand={'profile': meta.get('profile', 'peak'), 'duration': meta.get('duration', 3600)},
        signals={'max_green': meta.get('max_green'), 'max_red': meta.get('max_red')},
        observation={'mode': meta.get('obs', 'grid'), 'comm': named['comm'],
                     'noise': meta.get('noise', 0.0), 'dropout': meta.get('dropout', 0.0)},
        reward={'kind': named['reward']},
        agent={'double': named['double'], 'target_update': named['target_update']},
        train={'seeds': [meta['seed']], 'episodes': meta.get('episodes', 150),
               'level_min': meta.get('level_min', 600.0), 'level_max': meta.get('level_max', 3000.0),
               'eps_min': meta.get('eps_min', 0.05), 'eps_fraction': meta.get('eps_fraction', 0.6),
               'val_every': meta.get('val_every', 10)},
    )


def make_observer(network, cfg, seed):
    from tsc.observation import Observer
    o = cfg.observation
    return Observer(network, mode=o.mode, comm=o.comm, noise=o.noise, dropout=o.dropout,
                    seed=seed, e2_length=o.e2_length, message=o.message,
                    approach_length=o.approach_length, layout=o.layout)


def load_policy(run_dir, network, which='best', device=None):
    """(greedy RLController, run Config) of a trained run, on `network`.

    The policy runs on `device` whatever device it was trained on; evaluation passes
    evaluate.device ('cpu'), so its results do not depend on the GPU a run was trained on
    (docs/GPU.md).  None: the device this process already runs on (a policy loaded inside a
    training run: RL experts, expert workers, acceptance), else the CPU.
    """
    from tsc import devices
    devices.setup(device)
    cfg = load_run_config(run_dir)
    seed = cfg.train.seeds[0]
    observer = make_observer(network, cfg, seed + EVAL_NOISE_SEED_OFFSET)
    return make_controller(network, observer, cfg).load(run_dir, which), cfg


def make_controller(network, observer, cfg, epsilon=0.0):
    """RLController with fresh agents for a Config (action space, shared agent)."""
    from tsc.controllers.rl import RLController, make_actions, make_agents
    actions = make_actions(network, observer, cfg)
    agents = make_agents(network, observer, cfg.agent, actions)
    return RLController(agents, observer, epsilon, actions=actions,
                        shared=cfg.agent.shared and not cfg.agent.phase_net)
