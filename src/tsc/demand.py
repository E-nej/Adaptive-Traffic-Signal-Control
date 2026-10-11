"""Traffic demand: route files for one scenario (demand level + seed).

generator 'random_trips': Poisson insertions between random origin/destination
edges (randomTrips.py), routed by duarouter.  This is the demand for the closed
test network (F9); a corridor with real entry/exit roads and counts will add a
generator here (e.g. routeSampler).
"""

from tsc import sumo_env

# Demand multipliers over the episode (piecewise constant, equal-length slices)
PROFILES = {
    'flat': [1.0],
    'peak': [0.5, 0.75, 1.0, 1.0, 0.75, 0.5],
}


def generate_routes(network, demand, route_file, seed, rate_vph):
    """Write the route file of one scenario and return its path.

    rate_vph is the network-wide insertion rate (veh/h) at profile factor 1.0;
    `demand` is a config.DemandConfig.
    """
    if demand.generator == 'random_trips':
        return random_trips(network, route_file, seed, rate_vph, demand.duration, demand.profile,
                            demand.min_distance)
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
