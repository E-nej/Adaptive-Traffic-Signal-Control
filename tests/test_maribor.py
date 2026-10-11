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
