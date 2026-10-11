"""Fixed-time plans for a given demand: Webster (isolated) and a coordinated green wave."""

import math
import os
import xml.etree.ElementTree as ET

from tsc import sumo_env


def required_cycle(network, tl_id=None):
    """Shortest cycle that gives every stage its minimum green (s): of one junction, or the
    longest over all junctions.

    tlsCycleAdaptation gives some stage less than its minimum green, down to a zero or
    negative green, when its cycle is shorter than this (low demand at junctions with many
    stages), so such plans are recomputed with this as the minimum cycle.
    """
    longest = 0.0
    for tl in ([tl_id] if tl_id else network.tl_ids):
        program = next(iter(network.sumo_net.getTLS(tl).getPrograms().values()))
        intergreen = sum(p.duration for p in program.getPhases()
                         if 'y' in p.state or not any(c in 'Gg' for c in p.state))
        longest = max(longest, intergreen + len(network.greens[tl]) * network.min_green)
    return longest


def _too_short(network, plan_file):
    """True if some junction's cycle in the plan is shorter than required_cycle of it."""
    for tl in ET.parse(plan_file).getroot().iter('tlLogic'):
        if tl.get('id') not in network.greens:
            continue
        cycle = sum(float(p.get('duration')) for p in tl.iter('phase'))
        if cycle < required_cycle(network, tl.get('id')) or                 any(float(p.get('duration')) <= 0 for p in tl.iter('phase')):
            return True
    return False


def webster_programs(network, route_files, out_file, min_cycle=30, max_cycle=120,
                     unified=False, program_id='webster'):
    """Webster cycle/split optimisation (SUMO tlsCycleAdaptation) → additional file.

    unified: one common cycle length for all junctions (needed for coordination).
    If a junction's cycle comes out too short for its stages' minimum greens (low demand),
    the plan is recomputed with the minimum cycle raised to required_cycle(network).
    """
    def run(cycle_min):
        sumo_env.run_tool(
            'tlsCycleAdaptation.py',
            '-n', network.net_file, '-r', ','.join(route_files), '-o', out_file,
            '-b', '0', '-y', str(network.yellow), '-a', str(network.all_red),
            '-g', str(network.min_green), '--min-cycle', str(cycle_min),
            '--max-cycle', str(max_cycle), '-p', program_id, *(['-u'] if unified else []),
        )
    run(min_cycle)
    if _too_short(network, out_file):
        run(max(min_cycle, math.ceil(required_cycle(network))))
    return out_file


def green_wave_programs(network, route_files, out_file, min_cycle=30, max_cycle=120):
    """Coordinated fixed time: Webster splits with a common cycle, plus offsets from SUMO's
    tlsCoordinator (green waves along the main flows of the given demand)."""
    base = out_file.replace('.add.xml', '.base.add.xml')
    offsets = out_file.replace('.add.xml', '.offsets.add.xml')
    webster_programs(network, route_files, base, min_cycle, max_cycle,
                     unified=True, program_id='greenwave')
    # tlsCoordinator reads one route file; the first calibration scenario is representative
    sumo_env.run_tool('tlsCoordinator.py', '-n', network.net_file, '-r', route_files[0],
                      '-a', base, '-o', offsets)
    offset = {tl.get('id'): tl.get('offset') for tl in ET.parse(offsets).getroot().iter('tlLogic')}
    tree = ET.parse(base)
    for tl in tree.getroot().iter('tlLogic'):
        tl.set('offset', offset.get(tl.get('id'), '0'))
    tmp = f'{out_file}.{os.getpid()}.tmp'
    tree.write(tmp, encoding='UTF-8', xml_declaration=True)
    os.replace(tmp, out_file)
    return out_file
