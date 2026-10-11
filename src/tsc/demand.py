"""Traffic demand: route files for one scenario (demand level + seed).

generator 'random_trips': Poisson insertions between random origin/destination
edges (randomTrips.py), routed by duarouter.  This is the demand for the closed
test network (F9).

generator 'od': Poisson trips between the network's boundary points (dead-end
nodes), for networks with open boundaries (the corridor).  A share `major_share`
of all trips runs between boundary points of the highest-priority road (through
traffic on the arterial); the rest follows a gravity model whose weight of a
boundary point is the capacity of its road (lanes × speed).  `tidal_share` > 0.5
makes trips that travel along `tidal_axis` more likely than trips against it
(e.g. a morning inbound peak); with `tidal_random` the direction is drawn per
scenario.  Trips are routed by duarouter.
"""

import xml.etree.ElementTree as ET

import numpy as np

from tsc import sumo_env

# Demand multipliers over the episode (piecewise constant, equal-length slices)
PROFILES = {
    'flat': [1.0],
    'peak': [0.5, 0.75, 1.0, 1.0, 0.75, 0.5],
}


def generate_routes(network, demand, route_file, seed, rate_vph, flip=None):
    """Write the route file of one scenario and return its path.

    rate_vph is the network-wide insertion rate (veh/h) at profile factor 1.0;
    `demand` is a config.DemandConfig.  `flip` forces the tidal direction of the
    'od' generator (None: drawn from the seed if demand.tidal_random).
    """
    if demand.generator == 'random_trips':
        return random_trips(network, route_file, seed, rate_vph, demand.duration, demand.profile,
                            demand.min_distance)
    if demand.generator == 'od':
        return od_trips(network, demand, route_file, seed, rate_vph, flip)
    raise ValueError(f'Unknown demand generator: {demand.generator!r}')


def random_trips(network, route_file, seed, rate_vph, duration, profile='flat', min_distance=150.0):
    rates = [max(1.0, rate_vph * f) for f in PROFILES[profile]]
    trips_file = route_file.replace('.rou.xml', '.trips.xml')
    sumo_env.run_tool(
        'randomTrips.py',
        '-n', network.net_file, '-o', trips_file, '-r', route_file,
        '-b', '0', '-e', str(duration),
        '--insertion-rate', *[f'{r:.2f}' for r in rates],
        '--poisson', '--seed', str(int(seed)),
        '--min-distance', f'{min_distance:g}',
        '--random-departpos', '--random-arrivalpos',
        '--trip-attributes', 'departLane="best" departSpeed="max"',
        '--prefix', f's{int(seed)}_',
    )
    return route_file


# ----------------------------------------------------------------------
# O/D demand between boundary points
# ----------------------------------------------------------------------

def boundary_points(network):
    """[(node id, (x, y), source edge, sink edge, road priority, capacity weight)].

    A boundary point is a dead-end node with one road leaving it (source) and/or one
    entering it (sink).  Where a one-way road or one carriageway of a dual carriageway
    is cut (real networks), the node has only a source or only a sink; the other is None.
    """
    net = network.sumo_net
    points = []
    for node in net.getNodes():
        out = [e for e in node.getOutgoing() if e.allows('passenger')]
        inc = [e for e in node.getIncoming() if e.allows('passenger')]
        if node.getType() != 'dead_end' or len(out) > 1 or len(inc) > 1 or not (out or inc):
            continue
        src, sink = (out or [None])[0], (inc or [None])[0]
        road = src or sink
        weight = road.getLaneNumber() * road.getSpeed()
        points.append((node.getID(), node.getCoord(), src and src.getID(), sink and sink.getID(),
                       road.getPriority(), weight))
    return sorted(points)


def od_matrix(network, demand, flip=False):
    """(pairs, probabilities): pairs are (origin point, destination point) indices."""
    pts = boundary_points(network)
    if len(pts) < 2:
        raise ValueError('The od generator needs a network with at least 2 boundary points')
    top = max(p[4] for p in pts)
    axis = np.asarray(demand.tidal_axis, dtype=float)
    axis = axis / np.linalg.norm(axis) * (-1.0 if flip else 1.0)
    major, other = [], []
    for i, o in enumerate(pts):
        for j, d in enumerate(pts):
            if i == j or o[2] is None or d[3] is None:     # origin needs a source, destination a sink
                continue
            delta = np.subtract(d[1], o[1])
            if np.hypot(*delta) < demand.min_distance:
                continue
            along = float(delta @ axis)
            tidal = demand.tidal_share if along > 1e-6 else (
                1.0 - demand.tidal_share if along < -1e-6 else 0.5)
            if o[4] == top and d[4] == top:
                major.append(((i, j), tidal))
            else:
                other.append(((i, j), o[5] * d[5] * tidal))
    share = demand.major_share if major and other else (1.0 if major else 0.0)
    pairs, probs = [], []
    for group, total in ((major, share), (other, 1.0 - share)):
        w = sum(x for _, x in group)
        for pair, x in group:
            pairs.append(pair)
            probs.append(total * x / w)
    return pts, pairs, np.asarray(probs) / sum(probs)


def is_directional(demand):
    """True if scenarios have a tidal direction (so fixed-time plans are made per direction)."""
    return demand.generator == 'od' and demand.tidal_share != 0.5


def scenario_flip(demand, seed):
    """Tidal direction of scenario `seed`: False = along tidal_axis, True = against it."""
    return bool(demand.tidal_random and np.random.default_rng(seed).random() < 0.5)


def od_trips(network, demand, route_file, seed, rate_vph, flip=None):
    rng = np.random.default_rng(seed)
    draw = rng.random()                      # always drawn, so `flip` does not shift the stream
    if flip is None:
        flip = bool(demand.tidal_random and draw < 0.5)
    pts, pairs, probs = od_matrix(network, demand, flip)

    factors = PROFILES[demand.profile]
    slice_len = demand.duration / len(factors)
    departs = []
    for k, f in enumerate(factors):
        n = rng.poisson(max(0.0, rate_vph * f) * slice_len / 3600.0)
        departs.extend(rng.uniform(k * slice_len, (k + 1) * slice_len, n))
    departs = np.sort(np.asarray(departs))
    chosen = rng.choice(len(pairs), size=len(departs), p=probs)

    root = ET.Element('routes')
    for n, (t, c) in enumerate(zip(departs, chosen)):
        o, d = pairs[c]
        ET.SubElement(root, 'trip', id=f's{int(seed)}_{n}', depart=f'{t:.2f}',
                      **{'from': pts[o][2], 'to': pts[d][3]},
                      departLane='best', departSpeed='max')
    trips_file = route_file.replace('.rou.xml', '.trips.xml')
    ET.indent(root)
    ET.ElementTree(root).write(trips_file, encoding='UTF-8', xml_declaration=True)
    sumo_env.run([
        'duarouter', '-n', network.net_file, '--route-files', trips_file, '-o', route_file,
        '--seed', str(int(seed)), '--no-step-log', '--no-warnings', '--ignore-errors',
    ])
    return route_file
