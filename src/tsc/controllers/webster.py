"""Fixed-time plans optimised with Webster's method for a given demand."""

from tsc import sumo_env


def webster_programs(network, route_files, out_file, min_cycle=30, max_cycle=120):
    """Webster cycle/split optimisation (SUMO tlsCycleAdaptation) → additional file."""
    sumo_env.run_tool(
        'tlsCycleAdaptation.py',
        '-n', network.net_file, '-r', ','.join(route_files), '-o', out_file,
        '-b', '0', '-y', str(network.yellow), '-a', str(network.all_red),
        '-g', str(network.min_green), '--min-cycle', str(min_cycle), '--max-cycle', str(max_cycle),
        '-p', 'webster',
    )
    return out_file
