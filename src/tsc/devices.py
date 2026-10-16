"""Device for the Q-networks: the CPU (default, the parity reference) or one GPU (CUDA or ROCm).

TensorFlow must be imported before a run is seeded (CLAUDE.md), so the setup is split:

    prepare(device)                before importing TensorFlow: device 'cpu' hides all GPUs
    configure(device, gpu_index)   after importing it, before seeding: picks the GPU and makes
                                   it reproducible (memory growth, op determinism, no TF32)

Neither consumes RNG.  GPU results are not bit-identical to CPU results (parity level 3
in CLAUDE.md); repeated runs on the same GPU and software stack are (level 2).  See
docs/GPU.md for installation (NVIDIA: `uv sync --extra cuda`; AMD: tensorflow-rocm) and
for which GPUs have been verified.

TensorFlow exposes CUDA and ROCm devices alike as 'GPU', so everything here is
backend-agnostic except the compatibility check (`verdict`).
"""

import os
import re
import sys

DEVICES = ('cpu', 'gpu')
HIDE_VARS = ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES')

# GPU models that passed docs/GPU.md's checklist (smoke test, same-device repeatability,
# end-to-end speedup); everything else that should work is reported as 'unverified'.
VERIFIED = {}

# AMD integrated GPUs (APUs): ROCm may enumerate them next to the discrete card, which is
# known to crash or hang; TensorFlow ROCm builds do not support them.
AMD_INTEGRATED = {'gfx1035', 'gfx1036', 'gfx1037', 'gfx1103', 'gfx1150', 'gfx1151', 'gfx1152'}

_configured = None   # the device this process was configured for


def check(device):
    if device not in DEVICES:
        raise ValueError(f'Unknown device {device!r}; choose from {DEVICES}')
    return device


def prepare(device):
    """Call before TensorFlow is imported.  'cpu': make sure TensorFlow cannot see any GPU."""
    check(device)
    if device == 'cpu':
        for var in HIDE_VARS:
            os.environ[var] = ''


def configure(device, gpu_index=None):
    """Call after importing TensorFlow and before seeding.  Returns a description of the device."""
    global _configured
    check(device)
    import tensorflow as tf
    if _configured is not None:
        if _configured['device'] != device or (device == 'gpu' and gpu_index is not None
                                               and gpu_index != _configured['index']):
            raise RuntimeError(f'This process already runs on {_configured}; one process cannot '
                               f'switch to {device!r} (use separate processes, e.g. --gpu-workers)')
        return _configured
    if device == 'cpu':
        if tf.config.list_physical_devices('GPU'):   # TensorFlow was imported before prepare()
            tf.config.set_visible_devices([], 'GPU')
        _configured = {'device': 'cpu'}
        return _configured

    gpus = list_gpus(tf)
    if not gpus:
        raise RuntimeError(
            f'device = "gpu" but TensorFlow sees no GPU ({backend(tf)} build). Run `tsc devices` '
            f'and see docs/GPU.md, or use device = "cpu".')
    chosen = choose(gpus, gpu_index)
    phys = tf.config.list_physical_devices('GPU')[chosen['index']]
    tf.config.set_visible_devices([phys], 'GPU')
    tf.config.experimental.set_memory_growth(phys, True)
    if tf.sysconfig.get_build_info().get('is_cuda_build'):
        tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.experimental.enable_op_determinism()
    _configured = {'device': 'gpu', 'index': chosen['index'], 'name': chosen['name'],
                   'arch': chosen['arch'], 'status': chosen['status'], 'backend': backend(tf)}
    return _configured


def choose(gpus, gpu_index=None):
    """The GPU to use: gpu_index, or else the first one that is not unsupported."""
    if gpu_index is None:
        usable = [g for g in gpus if g['status'] != 'unsupported']
        if not usable:
            raise RuntimeError('No supported GPU: ' + '; '.join(f"{g['name']}: {g['reason']}" for g in gpus))
        chosen = usable[0]
        if len(gpus) > 1:
            print(f"[devices] {len(gpus)} GPUs visible, using #{chosen['index']} {chosen['name']} "
                  f"(set train.gpu_index to choose)", file=sys.stderr, flush=True)
        return chosen
    if not 0 <= gpu_index < len(gpus):
        raise RuntimeError(f'gpu_index {gpu_index} out of range: {len(gpus)} GPU(s) visible')
    chosen = gpus[gpu_index]
    if chosen['status'] == 'unsupported':
        raise RuntimeError(f"GPU #{gpu_index} {chosen['name']} is not supported: {chosen['reason']}")
    return chosen


def setup(device=None, gpu_index=None):
    """prepare + import TensorFlow + configure, for code that has no RNG to protect (evaluation).

    device None: keep the device this process is already configured for, else 'cpu'.
    """
    if device is None:
        if _configured is not None:
            return _configured
        device = 'cpu'
    prepare(device)
    import tensorflow  # noqa: F401
    return configure(device, gpu_index)


# ----------------------------------------------------------------------
# What TensorFlow sees
# ----------------------------------------------------------------------

def backend(tf):
    info = tf.sysconfig.get_build_info()
    if info.get('is_rocm_build'):
        return 'rocm'
    if info.get('is_cuda_build'):
        return 'cuda'
    return 'cpu-only'


def list_gpus(tf):
    """[{index, name, arch, status, reason}] for every GPU TensorFlow can see."""
    build = tf.sysconfig.get_build_info()
    out = []
    for i, dev in enumerate(tf.config.list_physical_devices('GPU')):
        details = tf.config.experimental.get_device_details(dev)
        status, reason, arch = verdict(details, build)
        name = details.get('device_name', dev.name)
        if status != 'unsupported' and name in VERIFIED:
            status, reason = 'verified', VERIFIED[name]
        out.append({'index': i, 'name': name, 'arch': arch, 'status': status, 'reason': reason})
    return out


def verdict(details, build):
    """(status, reason, arch) of one GPU: status is 'unverified' (should work) or 'unsupported'."""
    if build.get('is_rocm_build'):
        return _rocm_verdict(details)
    cc = details.get('compute_capability')
    if not cc:
        return 'unverified', 'compute capability unknown', None
    major, minor = int(cc[0]), int(cc[1])
    arch = f'sm_{major}{minor}'
    caps = list(build.get('cuda_compute_capabilities') or [])
    native = [c for c in caps if c.startswith('sm_')]
    match = [c for c in native if _cc(c)[0] == major and _cc(c)[1] <= minor]
    if match:
        return 'unverified', f'runs the native {match[-1]} kernels of this TensorFlow build', arch
    ptx = [c for c in caps if c.startswith('compute_') and _cc(c) <= (major, minor)]
    reason = (f'this TensorFlow build has no kernels for compute capability {major}.{minor} '
              f'(built for {", ".join(caps)})')
    if ptx:
        reason += ('; it would JIT-compile PTX, which fails or takes 30+ minutes on new '
                   'architectures (e.g. Blackwell / RTX 50xx)')
        if os.environ.get('TSC_ALLOW_PTX_JIT') == '1':
            return 'unverified', reason + ' (allowed by TSC_ALLOW_PTX_JIT=1, experimental)', arch
        reason += '. Use device = "cpu" (or TSC_ALLOW_PTX_JIT=1 to try anyway)'
        return 'unsupported', reason, arch
    return 'unsupported', reason + '. Use device = "cpu".', arch


def _rocm_verdict(details):
    arch = None
    for key in ('gcn_arch_name', 'compute_capability'):
        v = details.get(key)
        if isinstance(v, str) and v.startswith('gfx'):
            arch = v.split(':')[0]
            break
    name = details.get('device_name', '')
    if arch in AMD_INTEGRATED or (arch is None and name.strip() == 'AMD Radeon Graphics'):
        return ('unsupported', 'integrated GPU (APU); hide it with HIP_VISIBLE_DEVICES or disable '
                'it in the BIOS, see docs/GPU.md', arch)
    return 'unverified', f'ROCm device{" " + arch if arch else ""}', arch


def _cc(tag):
    """'sm_89' / 'compute_90' -> (8, 9) / (9, 0)."""
    digits = re.sub(r'\D', '', tag)
    return int(digits[:-1]), int(digits[-1])


def describe():
    """Everything `tsc devices` prints (imports TensorFlow)."""
    import tensorflow as tf
    from importlib.metadata import PackageNotFoundError, version
    pins = {}
    for pkg in ('tensorflow', 'tensorflow-rocm'):
        try:
            pins[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    info = tf.sysconfig.get_build_info()
    return {
        'tensorflow': tf.__version__,
        'packages': pins,
        'backend': backend(tf),
        'build': {k: str(v) for k, v in info.items() if k in (
            'cuda_version', 'cudnn_version', 'cuda_compute_capabilities', 'rocm_version',
            'is_cuda_build', 'is_rocm_build')},
        'gpus': list_gpus(tf),
        'hidden_by_env': {v: os.environ[v] for v in HIDE_VARS if v in os.environ},
    }
