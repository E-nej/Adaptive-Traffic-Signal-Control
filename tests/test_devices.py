"""Device selection and GPU support verdicts, with mocked devices (no GPU needed)."""

import os

import pytest

from tsc import config, devices

# devices.build_info() of a CUDA build without Blackwell kernels, and of ROCm builds
CUDA_221 = {'is_cuda_build': True, 'is_rocm_build': False,
            'cuda_compute_capabilities': ['sm_60', 'sm_70', 'sm_80', 'sm_89', 'compute_90']}
ROCM = {'is_cuda_build': False, 'is_rocm_build': True}
ROCM_GFX11 = {'is_cuda_build': False, 'is_rocm_build': True,
              'cuda_compute_capabilities': ['gfx900', 'gfx90a', 'gfx1030', 'gfx1100', 'gfx1101']}


@pytest.mark.parametrize('cc, status', [
    ((6, 1), 'unverified'),     # Quadro P2000 (Pascal) runs sm_60 kernels
    ((7, 5), 'unverified'),     # Turing runs sm_70 kernels
    ((8, 6), 'unverified'),     # Ampere consumer runs sm_80 kernels
    ((8, 9), 'unverified'),     # RTX 40xx (Ada)
    ((12, 0), 'unsupported'),   # RTX 50xx (Blackwell): only PTX, needs CUDA >= 12.8
    ((5, 2), 'unsupported'),    # Maxwell: older than anything in the build
])
def test_cuda_verdict(cc, status, monkeypatch):
    monkeypatch.delenv('TSC_ALLOW_PTX_JIT', raising=False)
    got, reason, arch = devices.verdict({'device_name': 'x', 'compute_capability': cc}, CUDA_221)
    assert got == status, reason
    assert arch == f'sm_{cc[0]}{cc[1]}'
    if cc == (12, 0):
        assert 'Blackwell' in reason and 'device = "cpu"' in reason


def test_ptx_jit_opt_in(monkeypatch):
    monkeypatch.setenv('TSC_ALLOW_PTX_JIT', '1')
    details = {'device_name': 'x', 'compute_capability': (12, 0)}
    assert devices.verdict(details, CUDA_221)[0] == 'unverified'
    # no PTX at all for Maxwell: the opt-in does not help
    assert devices.verdict({'device_name': 'x', 'compute_capability': (5, 2)}, CUDA_221)[0] == 'unsupported'


def test_rocm_verdict():
    assert devices.verdict({'device_name': 'AMD Radeon RX 7900 XT', 'gcn_arch_name': 'gfx1100'},
                           ROCM)[0] == 'unverified'
    status, reason, arch = devices.verdict({'device_name': 'AMD Radeon Graphics',
                                            'gcn_arch_name': 'gfx1036:xnack-'}, ROCM)
    assert (status, arch) == ('unsupported', 'gfx1036') and 'HIP_VISIBLE_DEVICES' in reason
    assert devices.verdict({'device_name': 'AMD Radeon Graphics'}, ROCM)[0] == 'unsupported'


def test_rocm_build_without_the_arch():
    assert devices.verdict({'device_name': 'AMD Radeon RX 7900 XT', 'gcn_arch_name': 'gfx1100'},
                           ROCM_GFX11)[0] == 'unverified'
    status, reason, arch = devices.verdict({'device_name': 'AMD Radeon RX 9070',
                                            'gcn_arch_name': 'gfx1201'}, ROCM_GFX11)
    assert (status, arch) == ('unsupported', 'gfx1201') and 'no kernels' in reason


def test_torch_device(monkeypatch):
    import torch
    monkeypatch.setattr(devices, '_configured', None)
    assert devices.torch_device() == torch.device('cpu')
    monkeypatch.setattr(devices, '_configured', {'device': 'gpu', 'index': 1})
    assert devices.torch_device() == torch.device('cuda', 1)


def test_backend_of_builds():
    from types import SimpleNamespace
    build = lambda cuda, hip: SimpleNamespace(version=SimpleNamespace(cuda=cuda, hip=hip))  # noqa: E731
    assert devices.backend(build('13.0', None)) == 'cuda'
    assert devices.backend(build(None, '7.2.0')) == 'rocm'
    assert devices.backend(build(None, None)) == 'cpu-only'


def gpu(index, name, status):
    return {'index': index, 'name': name, 'arch': None, 'status': status, 'reason': status}


def test_choose_skips_integrated_gpu():
    gpus = [gpu(0, 'AMD Radeon Graphics', 'unsupported'), gpu(1, 'AMD Radeon RX 7900 XT', 'unverified')]
    assert devices.choose(gpus)['index'] == 1
    assert devices.choose(gpus, 1)['index'] == 1
    with pytest.raises(RuntimeError, match='not supported'):
        devices.choose(gpus, 0)
    with pytest.raises(RuntimeError, match='out of range'):
        devices.choose(gpus, 2)
    with pytest.raises(RuntimeError, match='No supported GPU'):
        devices.choose([gpus[0]])


def test_prepare_cpu_hides_gpus(monkeypatch):
    for var in devices.HIDE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    devices.prepare('gpu')
    assert not any(var in os.environ for var in devices.HIDE_VARS)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'     # deterministic cuBLAS
    devices.prepare('cpu')
    assert all(os.environ[var] == '' for var in devices.HIDE_VARS)
    with pytest.raises(ValueError):
        devices.prepare('tpu')


def test_setup_default_keeps_the_process_device(monkeypatch):
    """A policy loaded inside a GPU training run (RL experts, expert workers, acceptance)
    keeps the run's device: one process cannot switch from GPU to CPU."""
    for var in devices.HIDE_VARS:
        monkeypatch.delenv(var, raising=False)
    on_gpu = {'device': 'gpu', 'index': 0, 'name': 'x', 'arch': None, 'status': 'unverified',
              'backend': 'cuda'}
    monkeypatch.setattr(devices, '_configured', on_gpu)
    assert devices.setup() is on_gpu
    assert not any(var in os.environ for var in devices.HIDE_VARS)     # prepare('cpu') not run
    with pytest.raises(RuntimeError, match='cannot switch'):
        devices.setup('cpu')


def test_device_config_defaults():
    cfg = config.Config()
    assert cfg.train.device == 'cpu' and cfg.train.gpu_index is None
    assert cfg.evaluate.device == 'cpu'
    cfg = config.from_dict({'train': {'device': 'gpu', 'gpu_index': 1}})
    assert (cfg.train.device, cfg.train.gpu_index) == ('gpu', 1)


def test_parse_mix():
    from tsc.experiments.bench import mix_dir, mix_name, parse_mix
    assert parse_mix('gpu:2,cpu:4') == {'gpu': 2, 'cpu': 4}
    assert mix_dir(parse_mix('gpu:2,cpu:4')) == 'gpu2_cpu4'      # no ':' in paths (SUMO)
    assert mix_name(parse_mix('cpu:5')) == 'cpu:5'
    for bad in ('gpu2', 'tpu:1', 'cpu:0'):
        with pytest.raises(ValueError):
            parse_mix(bad)
