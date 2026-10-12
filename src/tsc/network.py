"""Prepared road network: signal plans, green stages, lanes and neighbours.

By default the source network is rebuilt with netconvert so every junction
gets a conflict-free signal plan with yellow + all-red intergreens (the plan a
traffic engineer would start from).  All controllers then switch between the
*green stages* of that plan, so they are compared on equal terms.

Nothing here assumes a particular network: junctions may have any number of
incoming lanes and green stages; observations pad to the largest junction.
"""

import hashlib
import heapq
import json
import os

from tsc import sumo_env
import sumolib  # type: ignore  # noqa: E402  (sumo_env puts SUMO's tools on sys.path)

# Signals further apart than this (m of road, through unsignalised junctions) are not
# neighbours.  Synthetic networks are far below it; in real networks with sparse signals
# it keeps a junction's neighbours to the ones traffic actually travels between.
NEIGHBOUR_RANGE = 1000.0


class Network:
    """A network prepared for the experiments.

    Generated files (rebuilt networks, detector files) go to
    `<work_root>/<net name>-<hash>/`; the hash covers the source file and the
    signal settings, so a changed setting never reuses a stale network.

    Attributes
    ----------
    net_file, net_actuated : str
        Network with fixed-time plans / with SUMO actuated plans.
    tl_ids : list[str]
        Signalised junctions (sorted).
    greens : dict[str, list[str]]
        Signal states of the green stages of every junction.
    links : dict[str, list[(int, str, str)]]
        (link index, incoming lane, outgoing lane) per junction.
    in_lanes, out_lanes : dict[str, list[str]]
    lane_length, lane_speed : dict[str, float]
    neighbours : dict[str, list[str]]
        Signalised junctions directly connected (through unsignalised nodes).
    upstream_edges, downstream_edges : dict[str, list[str]]
        Edge chains before a controlled incoming edge / after an outgoing edge, joined by
        pass-through nodes (lane splits such as left-turn bays); empty for most networks.
    approach_edges : dict[str, list[str]]
        All upstream-chain edges of a junction's approaches: queues there belong to the
        junction even though those edges are not controlled.
    toward, arrival : dict[(str, str), list]
        toward[(a, b)]: link indices of a leading to neighbour b; arrival[(a, b)]: b's
        controlled incoming lanes on the road from a (for directional messages).
    """

    def __init__(self, cfg, work_root):
        self.cfg = cfg
        self.src_net = cfg.path
        self.yellow = cfg.yellow
        self.all_red = cfg.all_red
        self.min_green = cfg.min_green
        self.work_dir = os.path.join(work_root, f'{_stem(cfg.path)}-{self._fingerprint()}')
        os.makedirs(self.work_dir, exist_ok=True)

        if cfg.rebuild_tls:
            self.net_file = os.path.join(self.work_dir, 'static.net.xml')
            self.net_actuated = os.path.join(self.work_dir, 'actuated.net.xml')
            self._build_net(self.net_file, 'static')
            self._build_net(self.net_actuated, 'actuated')
        else:
            # Real signal plans are used as they are (step 3); no actuated variant yet.
            self.net_file = self.src_net
            self.net_actuated = None

        self.sumo_net = sumolib.net.readNet(self.net_file, withPrograms=True)
        self.tl_ids = sorted(tl.getID() for tl in self.sumo_net.getTrafficLights())
        self.greens = {tl_id: self._green_stages(tl_id) for tl_id in self.tl_ids}
        self.links = {tl_id: self._links(tl_id) for tl_id in self.tl_ids}
        self.in_lanes = {
            tl_id: sorted({l[1] for l in self.links[tl_id]}) for tl_id in self.tl_ids
        }
        self.out_lanes = {
            tl_id: sorted({l[2] for l in self.links[tl_id]}) for tl_id in self.tl_ids
        }
        self.lane_length = {
            lane: self.sumo_net.getLane(lane).getLength()
            for tl_id in self.tl_ids for lane in self.in_lanes[tl_id] + self.out_lanes[tl_id]
        }
        self.lane_speed = {
            lane: self.sumo_net.getLane(lane).getSpeed()
            for tl_id in self.tl_ids for lane in self.in_lanes[tl_id]
        }
        # Approaches longer than the controlled lanes: a lane split (e.g. a left-turn bay)
        # leaves an edge chain joined by pass-through nodes before the controlled edge.
        self.upstream_edges = {}      # controlled incoming edge -> chain before it, nearest first
        self.downstream_edges = {}    # outgoing edge -> chain after it, nearest first
        for tl_id in self.tl_ids:
            for _, in_lane, out_lane in self.links[tl_id]:
                in_edge = self.sumo_net.getLane(in_lane).getEdge()
                out_edge = self.sumo_net.getLane(out_lane).getEdge()
                if in_edge.getID() not in self.upstream_edges:
                    self.upstream_edges[in_edge.getID()] = self._chain(in_edge, upstream=True)
                if out_edge.getID() not in self.downstream_edges:
                    self.downstream_edges[out_edge.getID()] = self._chain(out_edge, upstream=False)
        self.approach_edges = {
            tl_id: sorted({e for l in self.in_lanes[tl_id] for e in self.upstream_edges[lane_edge(l)]})
            for tl_id in self.tl_ids
        }
        self.neighbours = self._neighbours()
        self.toward, self.arrival = self._directions()
        self.max_neighbours = max((len(v) for v in self.neighbours.values()), default=0)
        self.max_greens = max(len(g) for g in self.greens.values())
        self.max_in_lanes = max(len(l) for l in self.in_lanes.values())

    # ------------------------------------------------------------------

    def _netconvert_args(self, tls_type):
        c = self.cfg
        return [
            '--tls.rebuild', '--tls.layout', c.tls_layout,
            '--tls.yellow.time', str(c.yellow), '--tls.allred.time', str(c.all_red),
            '--tls.min-dur', str(c.min_green), '--tls.max-dur', str(c.max_green),
            '--tls.default-type', tls_type,
            '--no-warnings',
        ]

    def _fingerprint(self):
        h = hashlib.sha1()
        with open(self.src_net, 'rb') as f:
            h.update(f.read())
        h.update(json.dumps([self.cfg.rebuild_tls, self._netconvert_args('static'),
                             self._netconvert_args('actuated')]).encode())
        return h.hexdigest()[:8]

    def _build_net(self, out_file, tls_type):
        if os.path.exists(out_file):
            return
        # parallel runs (also separate commands) may need it at the same time: one builds it
        with sumo_env.file_lock(f'{out_file}.lock'):
            if os.path.exists(out_file):
                return
            tmp = f'{out_file}.{os.getpid()}.tmp.net.xml'
            sumo_env.run(['netconvert', '-s', self.src_net, '-o', tmp,
                          *self._netconvert_args(tls_type)])
            os.replace(tmp, out_file)

    def _green_stages(self, tl_id):
        """Signal states of the green stages (no yellow, at least one green)."""
        tl = self.sumo_net.getTLS(tl_id)
        program = next(iter(tl.getPrograms().values()))
        greens = []
        for phase in program.getPhases():
            s = phase.state
            if 'y' in s or not any(c in 'Gg' for c in s):
                continue
            if s not in greens:
                greens.append(s)
        return greens

    def _links(self, tl_id):
        """[(link_index, in_lane_id, out_lane_id)] for the junction."""
        return sorted(
            (idx, in_lane.getID(), out_lane.getID())
            for in_lane, out_lane, idx in self.sumo_net.getTLS(tl_id).getConnections()
        )

    @staticmethod
    def _pass_through(node):
        """A node that only joins two parts of one road (e.g. created by a lane split)."""
        if node.getType() in ('traffic_light', 'dead_end'):
            return False
        inc = [e for e in node.getIncoming() if e.allows('passenger')]
        out = [e for e in node.getOutgoing() if e.allows('passenger')]
        return len(inc) == 1 and len(out) == 1

    def _chain(self, edge, upstream):
        chain, seen = [], {edge.getID()}
        while True:
            node = edge.getFromNode() if upstream else edge.getToNode()
            if not self._pass_through(node):
                return chain
            edge = (node.getIncoming() if upstream else node.getOutgoing())[0]
            if edge.getID() in seen:
                return chain
            seen.add(edge.getID())
            chain.append(edge.getID())

    def _neighbours(self, max_dist=None, strict=False):
        """Signalised junctions reachable by following edges through unsignalised nodes
        (within max_dist m of road, default NEIGHBOUR_RANGE; strict: the first edge counts
        towards the limit too).

        A traffic light may control several nodes (joined junctions in OSM networks);
        its nodes are the ones its controlled lanes lead into.
        """
        max_dist = NEIGHBOUR_RANGE if max_dist is None else max_dist
        node_tl = {}
        for tl_id in self.tl_ids:
            for _, in_lane, _ in self.links[tl_id]:
                node_tl[self.sumo_net.getLane(in_lane).getEdge().getToNode().getID()] = tl_id
        neighbours = {t: set() for t in self.tl_ids}
        for node_id, tl_id in sorted(node_tl.items()):
            for edge in self.sumo_net.getNode(node_id).getOutgoing():
                for to_tl in _reach([edge], node_tl, tl_id, through_dead_ends=True,
                                    max_dist=max_dist, strict=strict):
                    neighbours[tl_id].add(to_tl)
                    neighbours[to_tl].add(tl_id)
        return {t: sorted(v) for t, v in neighbours.items()}

    def groups(self, max_dist):
        """Signals split into groups of junctions linked by roads of at most max_dist m
        (connected components), e.g. a coordinated arterial vs isolated junctions.
        Groups are sorted by their first junction id."""
        close = self._neighbours(max_dist, strict=True)
        seen, groups = set(), []
        for start in self.tl_ids:
            if start in seen:
                continue
            group, stack = [], [start]
            seen.add(start)
            while stack:
                t = stack.pop()
                group.append(t)
                for n in close[t]:
                    if n not in seen:
                        seen.add(n)
                        stack.append(n)
            groups.append(sorted(group))
        return groups

    def _directions(self):
        """Which movements of a junction lead to a neighbour, and where they arrive.

        Returns (toward, arrival):
            toward[(src, dst)]  : link indices of src whose outgoing road leads to dst
            arrival[(src, dst)] : dst's controlled incoming lanes on the road from src
        Roads are followed through unsignalised nodes, but not through dead ends
        (traffic leaves the network there).
        """
        node_tl = {}
        for tl_id in self.tl_ids:
            for _, in_lane, _ in self.links[tl_id]:
                node_tl[self.sumo_net.getLane(in_lane).getEdge().getToNode().getID()] = tl_id
        toward, arrival = {}, {}
        for src in self.tl_ids:
            reach = {}                                    # out edge -> {dst: arrival edges}
            for out_edge in sorted({lane_edge(l) for l in self.out_lanes[src]}):
                reach[out_edge] = _reach([self.sumo_net.getEdge(out_edge)], node_tl, src,
                                         through_dead_ends=False)
            for idx, in_lane, out_lane in self.links[src]:
                for dst, edges in reach[lane_edge(out_lane)].items():
                    toward.setdefault((src, dst), set()).add(idx)
                    lanes = {l for l in self.in_lanes[dst] if lane_edge(l) in edges}
                    arrival.setdefault((src, dst), set()).update(lanes)
        return ({k: sorted(v) for k, v in toward.items()},
                {k: sorted(v) for k, v in arrival.items()})


def _reach(start_edges, node_tl, src, through_dead_ends, max_dist=NEIGHBOUR_RANGE,
           strict=False):
    """{signal: edges arriving at it} for the signals (other than `src`) reachable from
    `start_edges` through unsignalised nodes within max_dist m (the start edges
    themselves always count, unless strict).  node_tl maps node id -> signal id."""
    found, done = {}, set()
    heap = [(e.getLength(), e.getID(), e) for e in start_edges
            if not strict or e.getLength() <= max_dist]
    heapq.heapify(heap)
    while heap:
        dist, eid, e = heapq.heappop(heap)
        if eid in done:
            continue
        done.add(eid)
        node = e.getToNode()
        dst = node_tl.get(node.getID())
        if dst is not None:
            if dst != src:
                found.setdefault(dst, set()).add(eid)
        elif through_dead_ends or node.getType() != 'dead_end':
            for nxt in node.getOutgoing():
                d = dist + nxt.getLength()
                if d <= max_dist and nxt.getID() not in done:
                    heapq.heappush(heap, (d, nxt.getID(), nxt))
    return found


def lane_edge(lane_id):
    """Edge id of a lane id ('<edge>_<index>')."""
    return lane_id.rsplit('_', 1)[0]


def _stem(path):
    name = os.path.basename(path)
    return name[:-len('.net.xml')] if name.endswith('.net.xml') else os.path.splitext(name)[0]
