"""Prepared road network: signal plans, green stages, lanes and neighbours.

By default the source network is rebuilt with netconvert so every junction
gets a conflict-free signal plan with yellow + all-red intergreens (the plan a
traffic engineer would start from).  All controllers then switch between the
*green stages* of that plan, so they are compared on equal terms.

Nothing here assumes a particular network: junctions may have any number of
incoming lanes and green stages; observations pad to the largest junction.
"""

import hashlib
import json
import os

from tsc import sumo_env
import sumolib  # type: ignore  # noqa: E402  (sumo_env puts SUMO's tools on sys.path)


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
        self.neighbours = self._neighbours()
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
        # parallel runs may build it at the same time: write to a temp file, then rename
        tmp = f'{out_file}.{os.getpid()}.tmp.net.xml'
        sumo_env.run(['netconvert', '-s', self.src_net, '-o', tmp, *self._netconvert_args(tls_type)])
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

    def _neighbours(self):
        """Signalised junctions reachable by following edges through unsignalised nodes.

        A traffic light may control several nodes (joined junctions in OSM networks);
        its nodes are the ones its controlled lanes lead into.
        """
        node_tl = {}
        for tl_id in self.tl_ids:
            for _, in_lane, _ in self.links[tl_id]:
                node_tl[self.sumo_net.getLane(in_lane).getEdge().getToNode().getID()] = tl_id
        neighbours = {t: set() for t in self.tl_ids}
        for node_id, tl_id in sorted(node_tl.items()):
            for edge in self.sumo_net.getNode(node_id).getOutgoing():
                seen = set()
                stack = [edge]
                while stack:
                    e = stack.pop()
                    if e.getID() in seen:
                        continue
                    seen.add(e.getID())
                    to_tl = node_tl.get(e.getToNode().getID())
                    if to_tl is not None:
                        if to_tl != tl_id:
                            neighbours[tl_id].add(to_tl)
                            neighbours[to_tl].add(tl_id)
                    else:
                        stack.extend(e.getToNode().getOutgoing())
        return {t: sorted(v) for t, v in neighbours.items()}


def _stem(path):
    name = os.path.basename(path)
    return name[:-len('.net.xml')] if name.endswith('.net.xml') else os.path.splitext(name)[0]
