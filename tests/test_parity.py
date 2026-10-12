"""Manifest comparison of the parity harness (no SUMO)."""

import copy

from tsc.experiments.parity import compare, report


def manifest():
    ep = {'episode': 0, 'row': {'episode': 0, 'delay': 12.5, 'mean_loss': float('nan'),
                                'seconds': 10.0, 'co2_g_per_veh': 120.0},
          'improved': True, 'route': 'r' * 64,
          'rng': {'python': 'p' * 64, 'trainer': 't' * 64},
          'agents': {'J0': {'online': 'a' * 64, 'target': 'b' * 64, 'optimizer': 'c' * 64}}}
    ep2 = copy.deepcopy(ep)
    ep2['episode'] = 1
    ep2['row']['episode'] = 1
    return {'episodes': [ep, ep2]}


def test_identical_including_nan():
    assert compare(manifest(), manifest()) == []
    assert report([]).startswith('IDENTICAL')


def test_ignored_fields():
    b = manifest()
    b['episodes'][0]['row']['seconds'] = 99.0
    b['episodes'][0]['row']['co2_g_per_veh'] = 0.0
    assert compare(manifest(), b) == []


def test_reports_first_differing_episode():
    b = manifest()
    b['episodes'][1]['agents']['J0']['target'] = 'x' * 64
    b['episodes'][1]['rng']['python'] = 'y' * 64
    diffs = compare(manifest(), b)
    assert {d[1] for d in diffs} == {'agents.J0.target', 'rng.python'}
    assert all(d[0] == 1 for d in diffs)
    assert 'first differing episode: 1' in report(diffs)


def test_episode_count_and_row_values():
    b = manifest()
    b['episodes'][0]['row']['delay'] = 12.500000001
    del b['episodes'][1]
    diffs = compare(manifest(), b)
    assert (None, 'number of episodes', 2, 1) in diffs
    assert (0, 'row.delay', 12.5, 12.500000001) in diffs
