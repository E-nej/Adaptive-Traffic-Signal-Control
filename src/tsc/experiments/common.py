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
    """Add fields to run_info.json (e.g. the device, known only once PyTorch is imported)."""
    path = os.path.join(out_dir, 'run_info.json')
    with open(path) as f:
        info = json.load(f)
    info.update(fields)
    if 'torch' in sys.modules:
        info['stack'] = software_stack()       # now including PyTorch's build info
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)


def software_stack():
    """Versions and hardware that can change numerical results (recorded with parity manifests).

    PyTorch's build info is only included when PyTorch is already imported (a run imports it
    before seeding; importing it here would only cost time).
    """
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for pkg in ('torch', 'numpy', 'eclipse-sumo', 'libsumo'):
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
        'env': {k: os.environ[k] for k in STACK_ENV if k in os.environ},
    }
    if 'torch' in sys.modules:
        torch = sys.modules['torch']
        info['torch_build'] = {
            'version': torch.__version__,
            'cuda': torch.version.cuda,
            'hip': torch.version.hip,
            'cudnn': (torch.backends.cudnn.version()
                      if torch.cuda.is_available() and torch.backends.cudnn.is_available() else None),
            # the SIMD level PyTorch's CPU kernels use (summation order: AVX2 vs AVX-512)
            'cpu_capability': torch.backends.cpu.get_cpu_capability(),
            'mkldnn': torch.backends.mkldnn.is_available(),
            'threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(),
        }
    return info


# Environment variables that can change numerical results or the device used.
STACK_ENV = ('TSC_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'ATEN_CPU_CAPABILITY',
             'CUBLAS_WORKSPACE_CONFIG', 'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES')


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

    fresh=True runs every task in a new process (needed when tasks use different devices,
    which a process cannot switch once PyTorch has set one up).  Workers are spawned, not
    forked, once this process has imported PyTorch: forking after its thread pools started
    can deadlock.
    """
    os.environ.setdefault('TSC_THREADS', '1')      # inherited by the worker processes
    if not fresh and (workers <= 1 or len(tasks) <= 1):
        for t in tasks:
            yield fn(t)
        return
    kwargs = {}
    if fresh:
        kwargs = dict(mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1)
    elif 'torch' in sys.modules:
        kwargs = dict(mp_context=multiprocessing.get_context('spawn'))
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
