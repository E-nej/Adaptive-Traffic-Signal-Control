"""
Auto-configure multi-agent TL setup from a SUMO .net.xml file.

Usage:
    parser = NetworkParser("intersection_2TL/2TL.net.xml")
    parser.generate_tl_programs("intersection_2TL/tl_programs.add.xml")
    # Then use parser.tl_config, parser.neighbours, parser.max_neighbours
"""

import math
import os
import xml.etree.ElementTree as ET


class NetworkParser:
    def __init__(self, net_file):
        self.net_file = net_file
        root = ET.parse(net_file).getroot()

        self._nodes       = self._parse_nodes(root)
        self._connections = self._parse_connections(root)  # tl_id -> [(link_idx, from_edge, dir)]
        self._edges       = self._parse_edges(root)        # edge_id -> (from_node, to_node)

        self.tl_ids        = []
        self.tl_config     = {}
        self.neighbours    = {}
        self.max_neighbours = 0

        self._parse_junctions(root)
        self._build_neighbours()

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_nodes(self, root):
        nodes = {}
        for j in root.findall('junction'):
            nodes[j.get('id')] = (float(j.get('x')), float(j.get('y')))
        return nodes

    def _parse_connections(self, root):
        conns = {}
        for c in root.findall('connection'):
            tl = c.get('tl')
            if tl is None:
                continue
            entry = (int(c.get('linkIndex')), c.get('from'), c.get('dir', 's'))
            conns.setdefault(tl, []).append(entry)
        for tl in conns:
            conns[tl].sort()
        return conns

    def _parse_edges(self, root):
        edges = {}
        for e in root.findall('edge'):
            if e.get('function') == 'internal':
                continue
            edges[e.get('id')] = (e.get('from'), e.get('to'))
        return edges

    def _parse_junctions(self, root):
        for j in root.findall('junction'):
            if j.get('type') != 'traffic_light':
                continue
            tl_id = j.get('id')
            self.tl_ids.append(tl_id)

            # Incoming external edges (preserve incLanes order for lane_groups)
            inc_lanes = j.get('incLanes', '').split()
            seen, roads = set(), []
            for lane in inc_lanes:
                edge = lane.rsplit('_', 1)[0]
                if not edge.startswith(':') and edge not in seen:
                    roads.append(edge)
                    seen.add(edge)

            self.tl_config[tl_id] = {
                'incoming_roads': roads,
                # Each road → base group 0,2,4,6 … (lanes 0-2 = group, lane 3 = group+1)
                'lane_groups': {r: i * 2 for i, r in enumerate(roads)},
            }

    def _build_neighbours(self):
        tl_set = set(self.tl_ids)
        self.neighbours = {t: [] for t in self.tl_ids}
        seen = set()

        for edge_id, (from_node, to_node) in self._edges.items():
            if from_node in tl_set and to_node in tl_set:
                key = tuple(sorted([from_node, to_node]))
                if key not in seen:
                    seen.add(key)
                    a, b = key
                    if b not in self.neighbours[a]:
                        self.neighbours[a].append(b)
                    if a not in self.neighbours[b]:
                        self.neighbours[b].append(a)

        self.max_neighbours = max(
            (len(v) for v in self.neighbours.values()), default=0
        )

    # ------------------------------------------------------------------
    # Opposing-pair detection (by geometry)
    # ------------------------------------------------------------------

    def _opposing_pairs(self, tl_id):
        """
        Return two pairs of incoming roads that face each other
        (angle difference closest to 180°).
        """
        cx, cy = self._nodes[tl_id]
        roads = self.tl_config[tl_id]['incoming_roads']

        angles = {}
        for road in roads:
            if road in self._edges:
                from_node, _ = self._edges[road]
                nx, ny = self._nodes.get(from_node, (cx, cy))
                angles[road] = math.atan2(ny - cy, nx - cx)

        if len(roads) < 4:
            # Fewer than 4 approaches — just split in half
            mid = len(roads) // 2
            return tuple(roads[:mid]), tuple(roads[mid:])

        # Find the pair with angle difference closest to π
        best, best_diff = None, float('inf')
        for i in range(len(roads)):
            for j in range(i + 1, len(roads)):
                a1 = angles.get(roads[i], 0)
                a2 = angles.get(roads[j], 0)
                diff = abs(abs(a1 - a2) - math.pi)
                if diff < best_diff:
                    best_diff = diff
                    best = (roads[i], roads[j])

        pair1 = best
        pair2 = tuple(r for r in roads if r not in pair1)
        return pair1, pair2

    # ------------------------------------------------------------------
    # Phase-string builder
    # ------------------------------------------------------------------

    def _phase_string(self, tl_id, active_roads, left_only=False):
        """
        Build a signal-state string for one phase.
        active_roads : roads that are allowed to move
        left_only    : True  → only left-turn links get G
                       False → straight + right links get G, left stays r
        """
        conns = self._connections.get(tl_id, [])
        if not conns:
            return ''
        n_links = conns[-1][0] + 1
        state = ['r'] * n_links

        for link_idx, from_edge, direction in conns:
            if from_edge not in active_roads:
                continue
            if left_only:
                if direction in ('l', 'L'):
                    state[link_idx] = 'G'
            else:
                if direction in ('r', 'R', 's'):
                    state[link_idx] = 'G'

        return ''.join(state)

    @staticmethod
    def _yellow(phase_str):
        return phase_str.replace('G', 'y')

    # ------------------------------------------------------------------
    # Public: generate the additional file
    # ------------------------------------------------------------------

    def generate_sumocfg(self, output_file, route_file='trips.xml',
                         tl_add_file='tl_programs.add.xml'):
        """Write a .sumocfg that references the net, route, and TL-program files."""
        net_name = os.path.basename(self.net_file)
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
            ' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">',
            '    <input>',
            f'        <net-file value="{net_name}"/>',
            f'        <route-files value="{route_file}"/>',
            f'        <additional-files value="{tl_add_file}"/>',
            '    </input>',
            '    <report>',
            '        <verbose value="true"/>',
            '    </report>',
            '    <time>',
            '        <begin value="0"/>',
            '    </time>',
            '    <processing>',
            '        <time-to-teleport value="-1"/>',
            '    </processing>',
            '</configuration>',
        ]
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'SUMO config written to {output_file}')

    def generate_tl_programs(self, output_file, green_dur=33, left_dur=6, yellow_dur=4):
        """
        Write a SUMO additional file with 8-phase DQN programs for every TL.

        Phase layout (same as single-agent setup):
            0 pair1 green (straight + right)     action 0
            1 pair1 yellow
            2 pair1 left-turn green               action 1
            3 pair1 left-turn yellow
            4 pair2 green (straight + right)      action 2
            5 pair2 yellow
            6 pair2 left-turn green               action 3
            7 pair2 left-turn yellow
        """
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<additional>']

        for tl_id in self.tl_ids:
            pair1, pair2 = self._opposing_pairs(tl_id)
            p1g   = self._phase_string(tl_id, pair1, left_only=False)
            p1l   = self._phase_string(tl_id, pair1, left_only=True)
            p2g   = self._phase_string(tl_id, pair2, left_only=False)
            p2l   = self._phase_string(tl_id, pair2, left_only=True)

            phases = [
                (green_dur,  p1g),
                (yellow_dur, self._yellow(p1g)),
                (left_dur,   p1l),
                (yellow_dur, self._yellow(p1l)),
                (green_dur,  p2g),
                (yellow_dur, self._yellow(p2g)),
                (left_dur,   p2l),
                (yellow_dur, self._yellow(p2l)),
            ]

            lines.append(f'    <tlLogic id="{tl_id}" type="static" programID="dqn" offset="0">')
            for dur, state in phases:
                lines.append(f'        <phase duration="{dur}" state="{state}"/>')
            lines.append('    </tlLogic>')

        lines.append('</additional>')
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'TL programs written to {output_file} for: {self.tl_ids}')
