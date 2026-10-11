"""Build the Maribor Titova cesta network from OpenStreetMap (step 3).

    uv run python networks/maribor/build.py           # convert titova.osm
    uv run python networks/maribor/build.py --info    # also list the signalised junctions

titova.osm is an Overpass API extract (© OpenStreetMap contributors, ODbL) of the
roads around Titova cesta, the north-south arterial east of Maribor's centre:

    bbox 46.538,15.636,46.565,15.666  (lat/lon, south-west / north-east)
    highway = motorway..living_street (+ _link), and traffic-signal nodes

Only the main road network is kept (residential and smaller streets are
removed, so their junctions with the arterial disappear), and only the largest
connected component.  Junction clusters of the dual carriageways are joined, so
one intersection is one signal.  OSM has no signal timings: the experiments
rebuild every plan with netconvert, as for the synthetic networks.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))

from tsc import sumo_env  # noqa: E402

OSM = os.path.join(HERE, 'titova.osm')
NET = os.path.join(HERE, 'titova.net.xml')

REMOVED_TYPES = ['highway.residential', 'highway.living_street', 'highway.unclassified',
                 'highway.service']
# Overpass returns whole ways, so some reach far outside the extract (a road running 3 km
# north).  Edges with geometry beyond this many metres north of the network's southern
# edge are removed (the extract is ~3.0 km tall).
MAX_NORTH = 3200.0


def netconvert(out, removed=()):
    sumo_env.run([
        'netconvert', '--osm-files', OSM, '-o', out,
        '--keep-edges.by-vclass', 'passenger',
        '--remove-edges.by-type', ','.join(REMOVED_TYPES),
        *(['--remove-edges.explicit', ','.join(sorted(removed))] if removed else []),
        '--keep-edges.components', '1',
        '--remove-edges.isolated',
        '--geometry.remove', '--roundabouts.guess', '--ramps.guess', '--edges.join',
        '--junctions.join', '--junctions.join-dist', '20',
        '--tls.guess-signals', '--tls.discard-simple', '--tls.join', '--tls.join-dist', '30',
        '--tls.default-type', 'static',
        '--no-turnarounds',
        '--output.street-names',
        '--no-warnings',
    ])


def build():
    import sumolib  # type: ignore
    netconvert(NET)
    net = sumolib.net.readNet(NET)
    far = {e.getID() for e in net.getEdges() if max(y for _, y in e.getShape()) > MAX_NORTH}
    if far:
        netconvert(NET, far)
    return NET


def info(path=NET):
    import sumolib  # type: ignore
    net = sumolib.net.readNet(path, withPrograms=True)
    print(f'{len(net.getEdges())} edges, {len(net.getNodes())} nodes, '
          f'{len(net.getTrafficLights())} signals')
    rows = []
    for tl in net.getTrafficLights():
        conns = tl.getConnections()
        if not conns:
            continue
        x, y = conns[0][0].getEdge().getToNode().getCoord()     # m east / north
        names = sorted({c[0].getEdge().getName() or '?' for c in conns})
        rows.append((y, x, tl.getID(), len(conns), names))
    for y, x, tl_id, n, names in sorted(rows, reverse=True):   # north to south
        print(f'y {y:7.1f} x {x:7.1f}  {tl_id[:30]:30s} {n:3d} links  {", ".join(names)}')


if __name__ == '__main__':
    print('wrote', build())
    if '--info' in sys.argv:
        info()
