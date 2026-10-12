"""Build the synthetic arterial corridor (step 2E.1).

    uv run python networks/corridor/build.py

Writes the plain-XML sources (corridor.nod/edg/con.xml) and corridor.net.xml
next to this file.  The network is fully determined by the constants below,
so rebuilding with the same SUMO version gives the same network.

Layout (x grows eastwards, the arterial runs along y = 0):

    - 7 signalised intersections S0..S6 on the arterial, irregular spacing
      (220–380 m), so a green wave is not trivial
    - arterial (primary, priority 3): 2 lanes per direction, 50 km/h, a 75 m
      left-turn bay (3rd lane) before every signalised intersection
    - cross streets (secondary, priority 2): 1 lane per direction, 40 km/h,
      300 m to the network boundary; S2 has only a north arm, S4 only a south arm
    - residential streets (priority 1): 1 lane per direction, 30 km/h, 150 m,
      joining the arterial at unsignalised priority junctions R0..R2 (they yield)
    - open boundaries: every street ends in a dead-end node where traffic enters/leaves
"""

import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))

from tsc import sumo_env  # noqa: E402

SIGNALS = [0, 280, 520, 900, 1150, 1500, 1720]          # x of S0..S6
ARMS = {i: ('n', 's') for i in range(len(SIGNALS))}       # cross-street arms per signal
ARMS[2] = ('n',)
ARMS[4] = ('s',)
RESIDENTIAL = [(400, ('n',)), (720, ('n', 's')), (1320, ('s',))]   # x of R0..R2, arms
WEST, EAST = -300, 2020                                   # arterial boundary x
CROSS_LEN, RES_LEN = 300, 150
BAY = 75.0                                                # m, left-turn bay length

ARTERIAL = dict(priority=3, numLanes=2, speed=13.89, type='primary')
CROSS = dict(priority=2, numLanes=1, speed=11.11, type='secondary')
RESID = dict(priority=1, numLanes=1, speed=8.33, type='residential')


def build_plain():
    nodes = [('W', WEST, 0, 'dead_end'), ('E', EAST, 0, 'dead_end')]
    arterial = []                                         # (x, node id, signalised?)
    for i, x in enumerate(SIGNALS):
        nodes.append((f'S{i}', x, 0, 'traffic_light'))
        arterial.append((x, f'S{i}', True))
        for arm in ARMS[i]:
            nodes.append((f'S{i}{arm}', x, CROSS_LEN if arm == 'n' else -CROSS_LEN, 'dead_end'))
    for j, (x, arms) in enumerate(RESIDENTIAL):
        nodes.append((f'R{j}', x, 0, 'priority'))
        arterial.append((x, f'R{j}', False))
        for arm in arms:
            nodes.append((f'R{j}{arm}', x, RES_LEN if arm == 'n' else -RES_LEN, 'dead_end'))
    arterial = [(WEST, 'W', False)] + sorted(arterial) + [(EAST, 'E', False)]

    edges, conns = [], []

    def edge(frm, to, attrs, bay=False):
        eid = f'{frm}_{to}'
        edges.append((eid, frm, to, attrs, bay))
        return eid

    # arterial, both directions; a bay where the edge ends at a signal
    for (x0, a, sa), (x1, b, sb) in zip(arterial, arterial[1:]):
        edge(a, b, ARTERIAL, bay=sb)
        edge(b, a, ARTERIAL, bay=sa)
    for i in range(len(SIGNALS)):
        for arm in ARMS[i]:
            edge(f'S{i}{arm}', f'S{i}', CROSS)
            edge(f'S{i}', f'S{i}{arm}', CROSS)
    for j, (_, arms) in enumerate(RESIDENTIAL):
        for arm in arms:
            edge(f'R{j}{arm}', f'R{j}', RESID)
            edge(f'R{j}', f'R{j}{arm}', RESID)

    # lane use on arterial approaches to signals: lane 0 right + straight, lane 1 straight,
    # lane 2 (bay) left only
    out_edges = {}
    for eid, frm, to, attrs, _ in edges:
        out_edges.setdefault(frm, []).append((eid, to, attrs))
    xy = {n: (x, y) for n, x, y, _ in nodes}
    for eid, frm, to, attrs, bay in edges:
        if not bay:
            continue
        eid = f'{eid}_bay'                                 # the split-off last part
        for oid, onode, oattrs in out_edges[to]:
            if onode == frm:
                continue                                   # no U-turns
            turn = _turn(xy[frm], xy[to], xy[onode])
            if turn == 's':
                conns += [(eid, oid, 0, 0), (eid, oid, 1, 1)]
            elif turn == 'r':
                conns.append((eid, oid, 0, 0))
            else:
                conns.append((eid, oid, 2, oattrs['numLanes'] - 1))
    return nodes, edges, conns


def _turn(a, b, c):
    """'l' / 's' / 'r' for the movement a → b → c (y up)."""
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    if dot > 0 and abs(cross) < 1e-6 * (abs(dot) + 1):
        return 's'
    return 'l' if cross > 0 else 'r'


def write_plain(nodes, edges, conns, stem):
    root = ET.Element('nodes')
    for nid, x, y, kind in nodes:
        ET.SubElement(root, 'node', id=nid, x=f'{x:.2f}', y=f'{y:.2f}', type=kind)
    _write(root, f'{stem}.nod.xml')

    root = ET.Element('edges')
    for eid, frm, to, attrs, bay in edges:
        e = ET.SubElement(root, 'edge', id=eid, **{'from': frm, 'to': to},
                          **{k: str(v) for k, v in attrs.items()})
        if bay:
            # split BAY metres before the end; the last part (<id>_bay) gets an extra left lane
            n = attrs['numLanes']
            ET.SubElement(e, 'split', pos=f'-{BAY:g}', lanes=' '.join(str(k) for k in range(n + 1)),
                          idBefore=eid, idAfter=f'{eid}_bay')
    _write(root, f'{stem}.edg.xml')

    root = ET.Element('connections')
    for frm, to, fl, tl in conns:
        ET.SubElement(root, 'connection', **{'from': frm, 'to': to},
                      fromLane=str(fl), toLane=str(tl))
    _write(root, f'{stem}.con.xml')


def _write(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='UTF-8', xml_declaration=True)


def main():
    stem = os.path.join(HERE, 'corridor')
    nodes, edges, conns = build_plain()
    write_plain(nodes, edges, conns, stem)
    sumo_env.run([
        'netconvert',
        '-n', f'{stem}.nod.xml', '-e', f'{stem}.edg.xml', '-x', f'{stem}.con.xml',
        '-o', f'{stem}.net.xml',
        '--no-turnarounds', '--tls.layout', 'opposites',
        '--no-warnings',
    ])
    print(f'wrote {stem}.net.xml')


if __name__ == '__main__':
    main()
