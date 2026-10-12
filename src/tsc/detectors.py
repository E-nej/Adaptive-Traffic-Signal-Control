"""Roadside detectors: what a real intersection can measure.

Every incoming lane of a signalised junction gets an E2 area detector
(camera/radar-like) on its last `e2_length` metres and an E1 induction loop
at the stop line.  Detector ids: `e2_<lane>`, `e1_<lane>`.
"""

import os


def e2_id(lane):
    return f'e2_{lane}'


def e1_id(lane):
    return f'e1_{lane}'


def detector_file(network, e2_length=50.0):
    """Write (once) and return the additional file with the detectors of `network`."""
    path = os.path.join(network.work_dir, f'detectors_{int(e2_length)}.add.xml')
    if os.path.exists(path):
        return path
    lines = ['<additional>']
    for tl_id in network.tl_ids:
        for lane in network.in_lanes[tl_id]:
            length = network.lane_length[lane]
            det_len = min(e2_length, length - 1.0)
            # file="NUL": SUMO's null device (on every platform); values are read through the API
            lines.append(
                f'    <laneAreaDetector id="{e2_id(lane)}" lane="{lane}" pos="{length - det_len:.2f}" '
                f'endPos="{length:.2f}" freq="3600" file="NUL"/>')
            lines.append(
                f'    <inductionLoop id="{e1_id(lane)}" lane="{lane}" pos="{max(0.0, length - 1.0):.2f}" '
                f'freq="3600" file="NUL"/>')
    lines.append('</additional>')
    tmp = f'{path}.{os.getpid()}.tmp'   # parallel runs may build it at the same time
    with open(tmp, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    os.replace(tmp, path)
    return path
