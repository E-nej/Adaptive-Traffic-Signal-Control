"""Helpers shared by the experiments."""

import datetime
import json
import multiprocessing
import os
import platform
import socket
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import tsc
from tsc import config as config_mod
from tsc.sumo_env import file_lock  # noqa: F401  (re-exported for the experiments)
from tsc.network import Network


def network_for(cfg):
    """The prepared network of an experiment (generated files under <output_dir>/networks)."""
    return Network(cfg.network, os.path.join(cfg.output_dir, 'networks'))


# control modes that run on a coordinated plan (SignalHead ready every second)
PLAN_CONTROLS = ('plan', 'hierarchy')


def sim_kwargs(cfg):
    """Simulation options (Python-controlled junctions, start time, which trips count)."""
    # plan control switches on the plan's second; the policy still decides every interval
    interval = 1 if cfg.signals.control in PLAN_CONTROLS else cfg.signals.decision_interval
    return dict(decision_interval=interval, max_green=cfg.signals.max_green,
                max_red=cfg.signals.max_red, red_call=cfg.signals.red_call,
                e2_length=cfg.observation.e2_length,
                begin=cfg.demand.begin, undeparted=cfg.demand.count_undeparted,
                approach_length=cfg.observation.approach_length)


def save_run_info(cfg, out_dir):
    """Write the resolved config and provenance (versions, commit, time) next to the results."""
    os.makedirs(out_dir, exist_ok=True)
    config_mod.save(cfg, os.path.join(out_dir, 'config.toml'))
    info = {
        'tsc_version': tsc.__version__,
        'git_commit': _git_commit(),
        'sumo_version': _sumo_version(),
        'started': datetime.datetime.now().isoformat(timespec='seconds'),
        'stack': software_stack(),
    }
    with open(os.path.join(out_dir, 'run_info.json'), 'w') as f:
        json.dump(info, f, indent=2)


def update_run_info(out_dir, **fields):
    """Add fields to run_info.json (e.g. the device, known only once TensorFlow is imported)."""
    path = os.path.join(out_dir, 'run_info.json')
    with open(path) as f:
        info = json.load(f)
    info.update(fields)
    if 'tensorflow' in sys.modules:
        info['stack'] = software_stack()       # now including TensorFlow's build info
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)


def software_stack():
    """Versions and hardware that can change numerical results (recorded with parity manifests).

    TensorFlow's build info is only included when TensorFlow is already imported: importing it
    here could shift the RNG streams of a run that has not seeded yet.
    """
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for pkg in ('tensorflow', 'tensorflow-rocm', 'keras', 'numpy', 'eclipse-sumo', 'libsumo'):
        try:
            packages[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    info = {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'cpu': _cpu_model(),
        'cpu_flags': _cpu_flags(),
        'packages': packages,
        'env': {k: os.environ[k] for k in ('TSC_TF_THREADS', 'TF_ENABLE_ONEDNN_OPTS', 'ONEDNN_MAX_CPU_ISA',
                                           'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES')
                if k in os.environ},
    }
    if 'tensorflow' in sys.modules:
        tf = sys.modules['tensorflow']
        info['tf_build'] = {k: str(v) for k, v in tf.sysconfig.get_build_info().items()}
    return info


def physical_cores():
    """Number of physical CPU cores (not hardware threads)."""
    cores = set()
    try:
        with open('/proc/cpuinfo') as f:
            phys = core = None
            for line in f:
                if line.startswith('physical id'):
                    phys = line.split(':')[1].strip()
                elif line.startswith('core id'):
                    core = line.split(':')[1].strip()
                elif not line.strip() and core is not None:
                    cores.add((phys, core))
                    phys = core = None
    except OSError:
        pass
    return len(cores) or max(1, (os.cpu_count() or 2) // 2)


def _cpuinfo_field(name):
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith(name):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return None


def _cpu_model():
    return _cpuinfo_field('model name') or platform.processor() or None


def _cpu_flags():
    """The SIMD extensions that decide which oneDNN/Eigen kernels run."""
    flags = set((_cpuinfo_field('flags') or '').split())
    return sorted(f for f in ('sse4_2', 'avx', 'avx2', 'fma', 'avx512f', 'avx512_bf16', 'avx512_vnni',
                              'amx_tile') if f in flags)


def _git_commit():
    try:
        out = subprocess.run(['git', 'describe', '--always', '--dirty', '--tags'],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _sumo_version():
    try:
        from importlib.metadata import version
        return version('eclipse-sumo')
    except Exception:
        return None


def run_parallel(fn, tasks, workers, fresh=False):
    """Yield fn(task) results as they finish; in-process when workers <= 1 (easier debugging).

    fresh=True runs every task in a new process (needed when tasks use different TensorFlow
    devices, which a process cannot switch once TensorFlow is initialised).
    """
    os.environ.setdefault('TSC_TF_THREADS', '1')   # inherited by the worker processes
    if not fresh and (workers <= 1 or len(tasks) <= 1):
        for t in tasks:
            yield fn(t)
        return
    kwargs = {}
    if fresh:
        kwargs = dict(mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1)
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks))), **kwargs) as pool:
        futures = [pool.submit(_call, fn, t) for t in tasks]
        for fut in as_completed(futures):
            yield fut.result()


def _call(fn, task):
    """Run fn(task) in a worker; errors come back as RuntimeError with the worker's traceback
    (some exceptions, e.g. from libsumo, cannot be sent between processes)."""
    try:
        return fn(task)
    except Exception as e:
        import traceback
        raise RuntimeError(f'{type(e).__name__}: {e}\n{traceback.format_exc()}') from None
