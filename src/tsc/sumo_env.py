"""Locate SUMO and run its binaries and tools.

SUMO comes from the pip `eclipse-sumo` package (a project dependency) unless
SUMO_HOME is already set, in which case that installation is used.  Importing
this module puts SUMO's `bin/` on PATH and `tools/` on sys.path, so `sumolib`,
`traci` and the tool scripts work afterwards.

Environment variables
---------------------
SUMO_HOME   : use this SUMO installation instead of the pip package
TSC_TRACI=1 : use TraCI instead of libsumo even without the GUI (slower, for debugging)
"""

import contextlib
import os
import subprocess
import sys
import time


def _setup():
    if not os.environ.get('SUMO_HOME'):
        try:
            import sumo  # type: ignore  (the eclipse-sumo package)
            os.environ['SUMO_HOME'] = sumo.SUMO_HOME
        except ImportError as e:
            raise RuntimeError('SUMO not found: install eclipse-sumo or set SUMO_HOME') from e
    home = os.environ['SUMO_HOME']
    bin_dir = os.path.join(home, 'bin')
    if os.path.isdir(bin_dir) and bin_dir not in os.environ.get('PATH', '').split(os.pathsep):
        os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')
    tools = os.path.join(home, 'tools')
    if os.path.isdir(tools) and tools not in sys.path:
        sys.path.append(tools)
    return home, tools


SUMO_HOME, TOOLS = _setup()


def run(cmd):
    """Run a command, raising with its output if it fails."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return result


def run_tool(script, *args):
    """Run one of SUMO's Python tools (e.g. randomTrips.py) with this interpreter."""
    return run([sys.executable, os.path.join(TOOLS, script), *args])


def api(gui=False):
    """The simulation API: libsumo (fast, in-process) or TraCI (needed for the GUI)."""
    if gui or os.environ.get('TSC_TRACI') == '1':
        import traci  # type: ignore
        return traci
    import libsumo  # type: ignore
    return libsumo


@contextlib.contextmanager
def file_lock(path, stale=3600.0, poll=1.0):
    """Exclusive lock between processes (also separate `tsc` commands): a lock file created
    atomically.  Used where jobs build the same shared files (routes, plans).  A lock older
    than `stale` s is taken to be left by a crashed process and removed."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(path) > stale:
                    os.remove(path)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(poll)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
