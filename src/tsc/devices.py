"""Device for the Q-networks: the CPU (default, the parity reference) or one GPU (CUDA or ROCm).

PyTorch is imported before a run is seeded (CLAUDE.md), so the setup is split:

    prepare(device)                before importing PyTorch: device 'cpu' hides all GPUs;
                                   'gpu' sets CUBLAS_WORKSPACE_CONFIG (deterministic cuBLAS)
    configure(device, gpu_index)   after importing it, before seeding: picks the GPU and makes
                                   it reproducible (deterministic algorithms, no TF32)
    torch_device()                 the torch.device the agents put their networks on

None of them consumes RNG.  GPU results are not bit-identical to CPU results (parity level
3 in CLAUDE.md); repeated runs on the same GPU and software stack are (level 2).  See
docs/GPU.md for installation (`uv sync --extra cuda` or `--extra rocm`) and for which GPUs
have been verified.

PyTorch exposes CUDA and ROCm devices alike as 'cuda:<i>', so everything here is
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
# known to crash or hang; PyTorch ROCm builds do not support them.
AMD_INTEGRATED = {'gfx1035', 'gfx1036', 'gfx1037', 'gfx1103', 'gfx1150', 'gfx1151', 'gfx1152'}

_configured = None   # the device this process was configured for


def check(device):
    if device not in DEVICES:
        raise ValueError(f'Unknown device {device!r}; choose from {DEVICES}')
    return device


def prepare(device):
    """Call before PyTorch is imported.  'cpu': make sure PyTorch cannot see any GPU;
    'gpu': cuBLAS needs a fixed workspace to be deterministic (set before CUDA starts)."""
    check(device)
    if device == 'cpu':
        for var in HIDE_VARS:
            os.environ[var] = ''
    else:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def configure(device, gpu_index=None):
    """Call after importing PyTorch and before seeding.  Returns a description of the device."""
    global _configured
    check(device)
    import torch
    if _configured is not None:
        if _configured['device'] != device or (device == 'gpu' and gpu_index is not None
                                               and gpu_index != _configured['index']):
            raise RuntimeError(f'This process already runs on {_configured}; one process cannot '
                               f'switch to {device!r} (use separate processes, e.g. --gpu-workers)')
        return _configured
    if device == 'cpu':
        # The agents' networks live on torch_device() = the CPU, even if PyTorch was imported
        # before prepare() and can still see a GPU.
        _configured = {'device': 'cpu'}
        return _configured

    gpus = list_gpus(torch)
    if not gpus:
        raise RuntimeError(
            f'device = "gpu" but PyTorch sees no GPU ({backend(torch)} build). Run `tsc devices` '
            f'and see docs/GPU.md, or use device = "cpu".')
    chosen = choose(gpus, gpu_index)
    torch.cuda.set_device(chosen['index'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False      # NVIDIA: full float32 matmuls
    torch.backends.cudnn.allow_tf32 = False
    _configured = {'device': 'gpu', 'index': chosen['index'], 'name': chosen['name'],
                   'arch': chosen['arch'], 'status': chosen['status'], 'backend': backend(torch)}
    return _configured


def torch_device():
    """The torch.device for the Q-networks: the configured GPU, else the CPU."""
    import torch
    if _configured is not None and _configured['device'] == 'gpu':
        return torch.device('cuda', _configured['index'])
    return torch.device('cpu')


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
    """prepare + import PyTorch + configure, for code that has no RNG to protect (evaluation).

    device None: keep the device this process is already configured for, else 'cpu'.
    """
    if device is None:
        if _configured is not None:
            return _configured
        device = 'cpu'
    prepare(device)
    import torch  # noqa: F401
    return configure(device, gpu_index)


# ----------------------------------------------------------------------
# What PyTorch sees
# ----------------------------------------------------------------------

def backend(torch):
    if torch.version.hip:
        return 'rocm'
    if torch.version.cuda:
        return 'cuda'
    return 'cpu-only'


def build_info(torch):
    """The build facts `verdict` needs: backend flags and the GPU architectures compiled in."""
    available = torch.cuda.is_available()
    return {'is_rocm_build': bool(torch.version.hip),
            'is_cuda_build': bool(torch.version.cuda) and not torch.version.hip,
            'cuda_compute_capabilities': list(torch.cuda.get_arch_list()) if available else []}


def device_details(torch, i):
    p = torch.cuda.get_device_properties(i)
    return {'device_name': p.name, 'compute_capability': (p.major, p.minor),
            'gcn_arch_name': getattr(p, 'gcnArchName', None),
            'memory_gb': round(p.total_memory / 2 ** 30, 1)}


def list_gpus(torch):
    """[{index, name, arch, status, reason}] for every GPU PyTorch can see."""
    if not torch.cuda.is_available():
        return []
    build = build_info(torch)
    out = []
    for i in range(torch.cuda.device_count()):
        details = device_details(torch, i)
        status, reason, arch = verdict(details, build)
        name = details['device_name']
        if status != 'unsupported' and name in VERIFIED:
            status, reason = 'verified', VERIFIED[name]
        out.append({'index': i, 'name': name, 'arch': arch, 'status': status, 'reason': reason})
    return out


def verdict(details, build):
    """(status, reason, arch) of one GPU: status is 'unverified' (should work) or 'unsupported'."""
    if build.get('is_rocm_build'):
        return _rocm_verdict(details, build)
    cc = details.get('compute_capability')
    if not cc:
        return 'unverified', 'compute capability unknown', None
    major, minor = int(cc[0]), int(cc[1])
    arch = f'sm_{major}{minor}'
    caps = list(build.get('cuda_compute_capabilities') or [])
    native = [c for c in caps if c.startswith('sm_')]
    match = [c for c in native if _cc(c)[0] == major and _cc(c)[1] <= minor]
    if match:
        return 'unverified', f'runs the native {match[-1]} kernels of this PyTorch build', arch
    ptx = [c for c in caps if c.startswith('compute_') and _cc(c) <= (major, minor)]
    reason = (f'this PyTorch build has no kernels for compute capability {major}.{minor} '
              f'(built for {", ".join(caps)})')
    if ptx:
        reason += ('; it would JIT-compile PTX, which fails or takes 30+ minutes on new '
                   'architectures (e.g. Blackwell / RTX 50xx)')
        if os.environ.get('TSC_ALLOW_PTX_JIT') == '1':
            return 'unverified', reason + ' (allowed by TSC_ALLOW_PTX_JIT=1, experimental)', arch
        reason += '. Use device = "cpu" (or TSC_ALLOW_PTX_JIT=1 to try anyway)'
        return 'unsupported', reason, arch
    return 'unsupported', reason + '. Use device = "cpu".', arch


def _rocm_verdict(details, build=None):
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
    built = [c for c in (build or {}).get('cuda_compute_capabilities') or [] if c.startswith('gfx')]
    if arch and built and arch not in built:
        return ('unsupported', f'this PyTorch ROCm build has no kernels for {arch} '
                f'(built for {", ".join(built)}). Use device = "cpu".', arch)
    return 'unverified', f'ROCm device{" " + arch if arch else ""}', arch


def _cc(tag):
    """'sm_89' / 'compute_90' -> (8, 9) / (9, 0)."""
    digits = re.sub(r'\D', '', tag)
    return int(digits[:-1]), int(digits[-1])


def describe():
    """Everything `tsc devices` prints (imports PyTorch)."""
    import torch
    build = {'cuda': torch.version.cuda, 'hip': torch.version.hip,
             'arch_list': build_info(torch)['cuda_compute_capabilities']}
    if torch.cuda.is_available() and torch.backends.cudnn.is_available():
        build['cudnn'] = torch.backends.cudnn.version()
    return {
        'torch': torch.__version__,
        'backend': backend(torch),
        'build': build,
        'gpus': list_gpus(torch),
        'hidden_by_env': {v: os.environ[v] for v in HIDE_VARS if v in os.environ},
    }
