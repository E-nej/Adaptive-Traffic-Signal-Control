"""Parity harness: fingerprints of a short training run, recorded and compared.

    tsc parity record configs/step1/full.toml            # -> <output_dir>/parity/<host>/full/golden/
    tsc parity check configs/step1/full.toml             # rerun as 'check', compare with 'golden'
    tsc parity compare <dir or manifest> <dir or manifest>

A run is cut to `episodes` episodes (validation every `val_every`) and after every
episode the manifest records the full train_log row, whether validation improved the
best weights, SHA-256 hashes of every agent's online, target and optimizer variables,
hashes of all RNG states (Python, NumPy global, trainer, plan, observer) and of the route
file, plus the software stack.  Two manifests are equal when every recorded value is
equal, except IGNORED_FIELDS of the row.

Parity levels (CLAUDE.md): an optimisation must reproduce the golden manifest on the
same machine's CPU (level 1); repeated runs on one device must match each other
(level 2); different machines or devices are compared for diagnosis only (level 3).
"""

import hashlib
import json
import os
import random
import re
import socket
import time

import numpy as np

from tsc import config as config_mod
from tsc.experiments.common import network_for, software_stack

# Not compared: timing, and SUMO's CO2 metric columns (the emissions device, which
# only fills them, is off in training episodes; the reward's CO2 term is unaffected).
IGNORED_FIELDS = ('seconds', 'co2_g_per_veh', 'co2_kg_total')
MANIFEST = 'manifest.json'
XML_COMMENT = re.compile(rb'<!--.*?-->', re.DOTALL)


def parity_root(cfg):
    host = socket.gethostname().replace('--', '-') or 'host'
    return os.path.join(cfg.output_dir, 'parity', host, cfg.name)


def short_config(cfg, episodes=6, val_every=2, seed=0):
    return config_mod.replace(cfg, train={'episodes': episodes, 'val_every': val_every, 'seeds': [seed]})


def record(cfg, out_dir):
    """Train the (already shortened) cfg in-process into out_dir; write and return its manifest."""
    from tsc.experiments.train import train_one
    os.environ.setdefault('TSC_THREADS', '1')      # as run_parallel does for `tsc train`
    network_for(cfg)                                # as train() does before starting the run
    episodes = []

    def on_episode_end(ep, info):
        episodes.append(fingerprint(ep, info))

    t0 = time.time()
    train_one(cfg, os.path.join(out_dir, 'run'), on_episode_end=on_episode_end)
    manifest = {
        'config': config_mod.to_dict(cfg),
        'stack': software_stack(),
        'seconds_total': round(time.time() - t0, 1),
        'episodes': episodes,
    }
    with open(os.path.join(out_dir, MANIFEST), 'w') as f:
        json.dump(manifest, f, indent=1)
    return manifest


def fingerprint(ep, info):
    """Everything about one finished episode that must be reproduced exactly (no RNG used)."""
    with open(info['route'], 'rb') as f:
        # without the generators' header comments (timestamp, output paths)
        route = hashlib.sha256(XML_COMMENT.sub(b'', f.read())).hexdigest()
    return {
        'episode': ep,
        'row': {k: _plain(v) for k, v in info['row'].items()},
        'improved': info['improved'],
        'route': route,
        'rng': {
            'python': _sha_text(repr(random.getstate())),
            'numpy_global': _sha_arrays([np.asarray(x) for x in np.random.get_state()[1:]]
                                        + [np.frombuffer(np.random.get_state()[0].encode(), np.uint8)]),
            **{name: _sha_text(json.dumps(g.bit_generator.state, sort_keys=True, default=str))
               for name, g in info['rngs'].items()},
        },
        'agents': {
            tl_id: {part: _sha_arrays(arrays) for part, arrays in agent.state_arrays().items()}
            for tl_id, agent in info['agents'].items()
        },
    }


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, MANIFEST)
    with open(path) as f:
        return json.load(f)


def compare(a, b):
    """Differences between two manifests as (episode, field, value_a, value_b); [] if equal."""
    diffs = []
    ea, eb = a['episodes'], b['episodes']
    if len(ea) != len(eb):
        diffs.append((None, 'number of episodes', len(ea), len(eb)))
    for x, y in zip(ea, eb):
        ep = x['episode']
        for key in sorted(set(x['row']) | set(y['row'])):
            if key in IGNORED_FIELDS:
                continue
            va, vb = x['row'].get(key), y['row'].get(key)
            if json.dumps(va) != json.dumps(vb):
                diffs.append((ep, f'row.{key}', va, vb))
        for key in ('improved', 'route'):
            if x[key] != y[key]:
                diffs.append((ep, key, x[key], y[key]))
        for name in sorted(set(x['rng']) | set(y['rng'])):
            if x['rng'].get(name) != y['rng'].get(name):
                diffs.append((ep, f'rng.{name}', x['rng'].get(name), y['rng'].get(name)))
        for tl_id in sorted(set(x['agents']) | set(y['agents'])):
            for part in ('online', 'target', 'optimizer'):
                va = x['agents'].get(tl_id, {}).get(part)
                vb = y['agents'].get(tl_id, {}).get(part)
                if va != vb:
                    diffs.append((ep, f'agents.{tl_id}.{part}', va, vb))
    return diffs


def report(diffs, label_a='a', label_b='b', limit=20):
    """Human-readable comparison result; the first differing episode is what matters."""
    if not diffs:
        return 'IDENTICAL: every compared field matches.'
    first = diffs[0][0]
    lines = [f'DIFFERENT: {len(diffs)} differing field(s); first differing episode: {first}']
    for ep, field, va, vb in diffs[:limit]:
        lines.append(f'  ep {ep}: {field}: {label_a}={_short(va)} {label_b}={_short(vb)}')
    if len(diffs) > limit:
        lines.append(f'  ... {len(diffs) - limit} more')
    return '\n'.join(lines)


# ----------------------------------------------------------------------

def _plain(v):
    if isinstance(v, np.generic):
        return v.item()
    return v


def _sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _sha_arrays(arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f'{a.dtype}{a.shape}'.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _short(v):
    s = json.dumps(v)
    return s[:16] if isinstance(v, str) and len(v) == 64 else s
