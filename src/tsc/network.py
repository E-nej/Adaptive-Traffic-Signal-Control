"""Network preparation and traffic demand for the realism experiments.

The source network is rebuilt with netconvert so every junction gets a
conflict-free signal plan with yellow + all-red intergreens (the same plan a
traffic engineer would start from).  All controllers then switch between the
*green stages* of that plan, so they are compared on equal terms.
"""

import os
import subprocess
import sys

import helpers  # noqa: F401  (sets SUMO_HOME / PATH before sumolib is used)
import sumolib  # type: ignore

TOOLS = os.path.join(os.environ['SUMO_HOME'], 'tools')

YELLOW = 3      # s
ALL_RED = 2     # s
MIN_GREEN = 10  # s
MAX_GREEN = 60  # s (used by SUMO's actuated baseline)

# Demand multipliers over the episode (piecewise constant, equal-length slices)
PROFILES = {
    'flat': [1.0],
    'peak': [0.5, 0.75, 1.0, 1.0, 0.75, 0.5],
}


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return result


class Scenario:
    """A prepared network: rebuilt signal plans, green stages, lanes and neighbours.

    Parameters
    ----------
    src_net : str
        Original .net.xml.
    work_dir : str
        Where the rebuilt networks and generated demand are written.
    """

    def __init__(self, src_net, work_dir):
        self.src_net = src_net
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

        self.net_file = os.path.join(work_dir, 'static.net.xml')
        self.net_actuated = os.path.join(work_dir, 'actuated.net.xml')
        self._build_net(self.net_file, 'static')
        self._build_net(self.net_actuated, 'actuated')

        self.net = sumolib.net.readNet(self.net_file, withPrograms=True)
        self.tl_ids = sorted(tl.getID() for tl in self.net.getTrafficLights())
        self.greens = {tl_id: self._green_stages(tl_id) for tl_id in self.tl_ids}
        self.links = {tl_id: self._links(tl_id) for tl_id in self.tl_ids}
        self.in_lanes = {
            tl_id: sorted({l[1] for l in self.links[tl_id]}) for tl_id in self.tl_ids
        }
        self.out_lanes = {
            tl_id: sorted({l[2] for l in self.links[tl_id]}) for tl_id in self.tl_ids
        }
        self.lane_length = {
            lane: self.net.getLane(lane).getLength()
            for tl_id in self.tl_ids for lane in self.in_lanes[tl_id] + self.out_lanes[tl_id]
        }
        self.lane_speed = {
            lane: self.net.getLane(lane).getSpeed()
            for tl_id in self.tl_ids for lane in self.in_lanes[tl_id]
        }
        self.neighbours = self._neighbours()
        self.max_neighbours = max((len(v) for v in self.neighbours.values()), default=0)
        self.max_greens = max(len(g) for g in self.greens.values())
        self.max_in_lanes = max(len(l) for l in self.in_lanes.values())

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    def _build_net(self, out_file, tls_type):
        if os.path.exists(out_file) and os.path.getmtime(out_file) >= os.path.getmtime(self.src_net):
            return
        tmp = f'{out_file}.{os.getpid()}.tmp.net.xml'
        _run([
            'netconvert', '-s', self.src_net, '-o', tmp,
            '--tls.rebuild', '--tls.layout', 'opposites',
            '--tls.yellow.time', str(YELLOW), '--tls.allred.time', str(ALL_RED),
            '--tls.min-dur', str(MIN_GREEN), '--tls.max-dur', str(MAX_GREEN),
            '--tls.default-type', tls_type,
            '--no-warnings',
        ])
        os.replace(tmp, out_file)

    def _green_stages(self, tl_id):
        """Signal states of the green stages (no yellow, at least one green)."""
        tl = self.net.getTLS(tl_id)
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
            for in_lane, out_lane, idx in self.net.getTLS(tl_id).getConnections()
        )

    def _neighbours(self):
        """Signalised junctions reachable by following edges through unsignalised nodes."""
        tl_set = set(self.tl_ids)
        neighbours = {t: set() for t in self.tl_ids}
        for tl_id in self.tl_ids:
            node = self.net.getNode(tl_id)
            for edge in node.getOutgoing():
                seen = set()
                stack = [edge]
                while stack:
                    e = stack.pop()
                    if e.getID() in seen:
                        continue
                    seen.add(e.getID())
                    to = e.getToNode().getID()
                    if to in tl_set:
                        if to != tl_id:
                            neighbours[tl_id].add(to)
                            neighbours[to].add(tl_id)
                    else:
                        stack.extend(e.getToNode().getOutgoing())
        return {t: sorted(v) for t, v in neighbours.items()}

    # ------------------------------------------------------------------
    # Detectors (what a real intersection measures)
    # ------------------------------------------------------------------

    def detector_file(self, e2_length=50.0):
        """E2 (camera/radar-like) on the last e2_length m and an E1 loop at the stop line
        of every incoming lane.  Ids: e2_<lane>, e1_<lane>."""
        path = os.path.join(self.work_dir, f'detectors_{int(e2_length)}.add.xml')
        if os.path.exists(path):
            return path
        lines = ['<additional>']
        for tl_id in self.tl_ids:
            for lane in self.in_lanes[tl_id]:
                length = self.lane_length[lane]
                det_len = min(e2_length, length - 1.0)
                lines.append(
                    f'    <laneAreaDetector id="e2_{lane}" lane="{lane}" pos="{length - det_len:.2f}" '
                    f'endPos="{length:.2f}" freq="3600" file="NUL"/>')
                lines.append(
                    f'    <inductionLoop id="e1_{lane}" lane="{lane}" pos="{max(0.0, length - 1.0):.2f}" '
                    f'freq="3600" file="NUL"/>')
        lines.append('</additional>')
        tmp = f'{path}.{os.getpid()}.tmp'   # parallel runs may build it at the same time
        with open(tmp, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        os.replace(tmp, path)
        return path

    # ------------------------------------------------------------------
    # Demand
    # ------------------------------------------------------------------

    def generate_routes(self, route_file, seed, rate_vph, duration, profile='flat'):
        """Poisson demand between random origin/destination edges, routed by duarouter.

        rate_vph is the network-wide insertion rate (veh/h) at profile factor 1.0.
        """
        rates = [max(1.0, rate_vph * f) for f in PROFILES[profile]]
        trips_file = route_file.replace('.rou.xml', '.trips.xml')
        _run([
            sys.executable, os.path.join(TOOLS, 'randomTrips.py'),
            '-n', self.net_file, '-o', trips_file, '-r', route_file,
            '-b', '0', '-e', str(duration),
            '--insertion-rate', *[f'{r:.2f}' for r in rates],
            '--poisson', '--seed', str(int(seed)),
            '--min-distance', '150',
            '--random-departpos', '--random-arrivalpos',
            '--trip-attributes', 'departLane="best" departSpeed="max"',
            '--prefix', f's{int(seed)}_',
        ])
        return route_file

    # ------------------------------------------------------------------
    # Fixed-time baseline
    # ------------------------------------------------------------------

    def webster_programs(self, route_files, out_file):
        """Webster cycle/split optimisation for the given demand (SUMO tlsCycleAdaptation)."""
        _run([
            sys.executable, os.path.join(TOOLS, 'tlsCycleAdaptation.py'),
            '-n', self.net_file, '-r', ','.join(route_files), '-o', out_file,
            '-b', '0', '-y', str(YELLOW), '-a', str(ALL_RED),
            '-g', str(MIN_GREEN), '--min-cycle', '30', '--max-cycle', '120',
            '-p', 'webster',
        ])
        return out_file
