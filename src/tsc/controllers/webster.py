"""Fixed-time plans for a given demand: Webster (isolated) and a coordinated green wave."""

import os
import xml.etree.ElementTree as ET

from tsc import sumo_env


def webster_programs(network, route_files, out_file, min_cycle=30, max_cycle=120,
                     unified=False, program_id='webster'):
    """Webster cycle/split optimisation (SUMO tlsCycleAdaptation) → additional file.

    unified: one common cycle length for all junctions (needed for coordination).
    """
    sumo_env.run_tool(
        'tlsCycleAdaptation.py',
        '-n', network.net_file, '-r', ','.join(route_files), '-o', out_file,
        '-b', '0', '-y', str(network.yellow), '-a', str(network.all_red),
        '-g', str(network.min_green), '--min-cycle', str(min_cycle), '--max-cycle', str(max_cycle),
        '-p', program_id, *(['-u'] if unified else []),
    )
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
