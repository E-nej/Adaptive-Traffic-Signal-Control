"""The Maribor Titova cesta network from OpenStreetMap (step 3)."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARIBOR = os.path.join(ROOT, 'networks', 'maribor', 'titova.net.xml')


def test_maribor_network(work_root):
    from tests.conftest import make_network
    from tsc.config import DemandConfig
    from tsc.demand import boundary_points, generate_routes
    net = make_network(MARIBOR, work_root)
    assert len(net.tl_ids) == 21
    assert max(len(v) for v in net.neighbours.values()) <= 6       # NEIGHBOUR_RANGE
    pts = boundary_points(net)
    assert any(p[2] is None for p in pts) and any(p[3] is None for p in pts)   # one-way ends
    demand = DemandConfig(generator='od', major_share=0.2, tidal_share=0.7,
                          tidal_axis=[0.0, 1.0], min_distance=300.0, duration=300)
    route = generate_routes(net, demand, os.path.join(work_root, 'mb.rou.xml'), 1, 3000)
    with open(route) as f:
        assert f.read().count('<vehicle') > 100


def test_short_approaches_get_full_detectors(work_root):
    """OSM splits roads near intersections: short controlled lanes get an area detector that
    continues upstream (multi-lane E2), and SUMO accepts the detector file."""
    from tests.conftest import make_network
    from tsc.config import DemandConfig
    from tsc.controllers import baseline
    from tsc.demand import generate_routes
    from tsc.detectors import e2_span
    from tsc.simulation import run_episode
    net = make_network(MARIBOR, work_root)
    spans = {l: e2_span(net, l) for t in net.tl_ids for l in net.in_lanes[t]}
    assert any(len(lanes) > 1 for lanes, _, _ in spans.values())
    short = [l for l, (_, _, covered) in spans.items() if net.lane_length[l] < 10]
    assert short and all(spans[l][2] >= 40 for l in short if len(spans[l][0]) > 1)
    demand = DemandConfig(generator='od', major_share=0.2, tidal_share=0.7,
                          tidal_axis=[0.0, 1.0], min_distance=300.0, duration=300)
    route = generate_routes(net, demand, os.path.join(work_root, 'mb_det.rou.xml'), 2, 2000)
    tls, ctrl = baseline('maxpressure_det')
    m = run_episode(net, ctrl, route, 300, os.path.join(work_root, 'mb_det'), seed=2, tls=tls)
    assert m['inserted'] > 50 and m['arrived'] > 20     # 300 s: long trips still running
