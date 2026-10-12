"""Roadside detectors: what a real intersection can measure.

Every incoming lane of a signalised junction gets an E2 area detector
(camera/radar-like) on its last `e2_length` metres and an E1 induction loop
at the stop line.  Detector ids: `e2_<lane>`, `e1_<lane>`.  Optionally a longer area
detector `e2a_<lane>` over the last `approach_length` metres (approaching vehicles).  A second loop `e1u_<lane>`
sits where SUMO's actuated control puts its loops (UPSTREAM_GAP s of travel at the
speed limit before the stop line, at most the lane's start).

A lane shorter than `e2_length` (common in OpenStreetMap networks, where a road
is split at every merge or lane change near an intersection) gets a multi-lane
area detector that continues upstream over the lanes feeding it, as a camera
watching the approach would, but not back through another signal.
"""

import os

VEH_SPACE = 7.5            # m per queued vehicle (length + gap)
UPSTREAM_GAP = 2.0         # s, SUMO's default 'detector-gap' of actuated signals


def e2_id(lane):
    return f'e2_{lane}'


def e2a_id(lane):
    return f'e2a_{lane}'


def e1_id(lane):
    return f'e1_{lane}'


def e1u_id(lane):
    return f'e1u_{lane}'


def e2_span(network, lane, e2_length=50.0):
    """(lanes from upstream to the stop line, start position on the first lane, metres covered)
    of the area detector of controlled lane `lane`."""
    length = network.lane_length[lane]
    if length >= e2_length:
        det_len = min(e2_length, length - 1.0)
        return [lane], length - det_len, det_len
    net = network.sumo_net
    signal_nodes = {net.getLane(l).getEdge().getToNode().getID()
                    for t in network.tl_ids for l in network.in_lanes[t]}
    lanes, covered, start = [lane], length, 0.0
    cur = net.getLane(lane)
    while covered < e2_length:
        if cur.getEdge().getFromNode().getID() in signal_nodes:
            break                                   # not back through another signal
        preds = sorted((p for p in cur.getIncoming() if p.allows('passenger')
                        and p.getID() not in lanes),
                       key=lambda p: (-p.getLength(), p.getID()))
        if not preds:
            break
        cur = preds[0]
        lanes.insert(0, cur.getID())
        need = e2_length - covered
        if cur.getLength() >= need:
            start = cur.getLength() - need
            covered = e2_length
        else:
            covered += cur.getLength()
    return lanes, start, covered


def e2_capacity(network, lane, e2_length=50.0):
    """Vehicles that fit in the area detector of `lane` (at least 1)."""
    length = network.lane_length[lane]
    covered = min(e2_length, length) if length >= e2_length else e2_span(network, lane, e2_length)[2]
    return max(1.0, covered / VEH_SPACE)


def detector_file(network, e2_length=50.0, approach_length=0.0):
    """Write (once) and return the additional file with the detectors of `network`."""
    tag = f'_a{int(approach_length)}' if approach_length else ''
    path = os.path.join(network.work_dir, f'detectors_{int(e2_length)}{tag}_v2.add.xml')
    if os.path.exists(path):
        return path
    lines = ['<additional>']
    for tl_id in network.tl_ids:
        for lane in network.in_lanes[tl_id]:
            length = network.lane_length[lane]
            lanes, start, _ = e2_span(network, lane, e2_length)
            # file="NUL": SUMO's null device (on every platform); values are read through the API
            where = f'lane="{lane}"' if len(lanes) == 1 else f'lanes="{" ".join(lanes)}"'
            lines.append(
                f'    <laneAreaDetector id="{e2_id(lane)}" {where} pos="{start:.2f}" '
                f'endPos="{length:.2f}" freq="3600" file="NUL"/>')
            lines.append(
                f'    <inductionLoop id="{e1_id(lane)}" lane="{lane}" pos="{max(0.0, length - 1.0):.2f}" '
                f'freq="3600" file="NUL"/>')
            if approach_length:
                a_lanes, a_start, _ = e2_span(network, lane, approach_length)
                a_where = (f'lane="{lane}"' if len(a_lanes) == 1
                           else f'lanes="{" ".join(a_lanes)}"')
                lines.append(
                    f'    <laneAreaDetector id="{e2a_id(lane)}" {a_where} pos="{a_start:.2f}" '
                    f'endPos="{length:.2f}" freq="3600" file="NUL"/>')
            upstream = max(0.0, length - UPSTREAM_GAP * network.lane_speed[lane])
            lines.append(
                f'    <inductionLoop id="{e1u_id(lane)}" lane="{lane}" pos="{upstream:.2f}" '
                f'freq="3600" file="NUL"/>')
    lines.append('</additional>')
    tmp = f'{path}.{os.getpid()}.tmp'   # parallel runs may build it at the same time
    with open(tmp, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    try:
        os.replace(tmp, path)
    except PermissionError:             # Windows: another run wrote it and has it open
        if not os.path.exists(path):
            raise
        os.remove(tmp)
    return path
