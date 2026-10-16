# Training on a GPU (opt-in)

Training runs on the **CPU by default**. The CPU is the bit-exact reference, as described in
CLAUDE.md (parity level 1). A GPU can train some or all seeds instead:

```bash
uv run tsc train configs/step1/full.toml --seeds 0-4 --set train.device=gpu   # all seeds on the GPU
uv run tsc train configs/step1/full.toml --seeds 0-6 --workers 8 --gpu-workers 2  # hybrid
uv run tsc devices          # what PyTorch sees, with a verdict per GPU
```

`--gpu-workers K` trains the first K seeds on the GPU and the others on the CPU.
Each run records its device in its `config.toml` and `run_info.json`.

## What changes with a GPU

- **Results differ from CPU results.** cuBLAS and rocBLAS sum in a different order than the CPU kernels.
  So a GPU run is not bit-identical to the CPU run of the same seed, nor to a run on another GPU model (parity level 3).
  Do not mix devices within one experiment without saying so.
- **Repeated runs on the same GPU are identical** (level 2). The device setup (`src/tsc/devices.py`) turns on PyTorch's
  deterministic algorithms (and `CUBLAS_WORKSPACE_CONFIG=:4096:8`) and turns TF32 off on NVIDIA.
  A GPU only counts as verified once this has been checked (see the checklist below).
- **Evaluation stays on the CPU** (`evaluate.device = "cpu"`, the default), whatever a policy was trained on.
  This is a reproducibility policy: evaluation results then do not depend on the GPU model.
  Trained policies loaded *inside* a training run follow that run's device, because one process cannot switch devices.
  This covers RL experts (`[signals] rl_experts`), expert workers and the acceptance test at the end of hierarchy runs.
  A standalone `tsc accept` runs on the CPU.
- **SUMO and the Python loop always run on the CPU.** A GPU only speeds up the Q-network updates (~60–70 % of an episode).
  Each GPU worker still needs roughly half a CPU core.

## Support status

| GPU | Status | Notes |
|---|---|---|
| AMD Radeon RX 7900 XT (gfx1100) | **priority target**, not yet verified | `--group rocm`; Linux only (see below) |
| NVIDIA RTX 20xx–50xx (sm_75–sm_120) | expected to work, not yet verified | `--group cuda` (CUDA 13.0; includes Blackwell / RTX 50xx) |
| NVIDIA Quadro P2000 / Pascal (sm_61) | detected (`tsc devices`: runs the sm_60 kernels); PyTorch runs not yet measured | `--group cuda126` (CUDA 13 dropped Pascal). With TensorFlow it was "works, not recommended", see "Measurements" |
| AMD integrated GPUs (e.g. Ryzen 7000 "Radeon Graphics", gfx1036) | unsupported | must be hidden on ROCm (see below) |

`tsc devices` refuses a GPU whose architecture the installed PyTorch build has no kernels for
(e.g. an RTX 50xx with the `cuda126` build); install the matching group instead.

A GPU moves to **verified** (`devices.VERIFIED`) only after the checklist below passes on it.

## Installing a GPU build

`uv sync` installs PyTorch's CPU build. A GPU build replaces it; the groups exclude each other:

```bash
uv sync --no-group cpu --group cuda       # NVIDIA RTX 20xx and newer (CUDA 13.0, incl. RTX 50xx)
uv sync --no-group cpu --group cuda126    # older NVIDIA, e.g. Pascal / Quadro P2000 (CUDA 12.6)
uv sync --no-group cpu --group rocm       # AMD, Linux only (ROCm 7.2)
export UV_NO_SYNC=1                       # so `uv run` keeps that build (it would re-sync to the CPU one)
uv run tsc devices
```

The wheels bundle the CUDA / ROCm runtime libraries (several GB); only the GPU driver must be
installed on the system. To go back to the CPU build: `unset UV_NO_SYNC; uv sync`.

## NVIDIA (CUDA)

- You need an NVIDIA driver that supports the build's CUDA version (CUDA 13.0: driver ≥ 580; 12.6: ≥ 560).
- Native Windows works (PyTorch ships Windows CUDA wheels); no WSL2 needed.
- Each process holds a CUDA context of a few hundred MB, which limits GPU workers on small cards.

## AMD (ROCm): RX 7900 XT

**Requirements**:
- Linux. AMD supports Ubuntu 22.04/24.04 and RHEL; other distributions may work but are unofficial.
  PyTorch's ROCm wheels are Linux-only; on Windows an AMD GPU is not used (train on the CPU).
- The `amdgpu` kernel driver (the ROCm runtime libraries come with the wheel).
- Your user in the `render` and `video` groups.

```bash
uv sync --no-group cpu --group rocm
export UV_NO_SYNC=1
uv run tsc devices          # must say "rocm build" and list the 7900 XT (gfx1100)
```

### Integrated GPU (Ryzen 7000 desktop CPUs)

The 7800X3D has an integrated Radeon GPU (gfx1036).
- ROCm may enumerate it next to the 7900 XT, which is known to crash or hang ROCm programs.
- Hide it with `export HIP_VISIBLE_DEVICES=<index of the 7900 XT>`
  (check `rocminfo` or `rocm-smi`), or disable the iGPU in the BIOS.
- `tsc devices` lists every device with its gfx target and marks integrated ones as unsupported.
- `train.gpu_index` selects a GPU explicitly.

## Validation checklist (send the results back)

Run everything from the repository root on the branch under test, with the GPU build installed
(see above) and `UV_NO_SYNC=1` set.
The `<host>` below is the machine's hostname.

1. `uv run tsc devices` (copy the output)
2. `uv run python -m pytest -m "not sumo"` (must pass)
3. **CPU reference** (also lets us compare machines): `uv run tsc parity record configs/step1/full.toml`
4. **GPU repeatability**: record twice, then compare:
   ```bash
   uv run tsc parity record configs/step1/full.toml --set train.device=gpu --label gpu_a
   uv run tsc parity record configs/step1/full.toml --set train.device=gpu --label gpu_b
   uv run tsc parity compare outputs/parity/<host>/full/gpu_a outputs/parity/<host>/full/gpu_b
   ```
   This must print `IDENTICAL`. If it fails with an error saying an op has no deterministic GPU implementation,
   send the error text.
5. **Throughput** (full episodes; takes about 1–2 h):
   ```bash
   uv run tsc bench configs/step1/full.toml --mix cpu:8 --mix gpu:1 --mix gpu:2 --mix gpu:4 \
       --mix gpu:2,cpu:6 --mix gpu:4,cpu:6 --episodes 10 --repeats 3
   ```
   Adapt `cpu:8` to your number of physical cores (`tsc devices` suggests it).
6. Send back:
   - `outputs/bench/full/bench.json`;
   - `outputs/parity/<host>/full/*/manifest.json`;
   - `outputs/parity/<host>/full/gpu_a/run/run_info.json`;
   - the output of steps 1–2.

**Go/no-go per machine**: the GPU is worth using if the best mix reaches at least **1.3×** the
episodes/min of the best CPU-only mix (`S vs best CPU` in the bench summary) **and** step 4 is `IDENTICAL`.
If it passes, the GPU model goes into `devices.VERIFIED` and the recommended mix into the table above.

## Measurements

These were measured with TensorFlow 2.21 (tsc 0.2), before the PyTorch port; PyTorch GPU runs
are not measured yet.

**Quadro P2000** (laptop with i7-8850H, 2026-10-12, before the Tier 1 CPU speedups):
- Same-seed repeatability: IDENTICAL.
- First divergence from the CPU run: episode 1. Until then the replay buffers are below `learn_start`, so nothing has trained yet.
- `DQNAgent` micro-benchmark, GPU / CPU: `replay()` 3.15 / 4.26 ms; greedy `act()` 0.87 / 0.54 ms.
  The batch-100 MLP is launch- and sync-bound on the GPU.
- `tsc bench` (step1/full; only 4 episodes × 1 repeat, so indicative):

| mix | episodes/min | S vs best CPU |
|---|---|---|
| cpu:5 | 15.75 | 1.00× |
| gpu:1 | 5.83 | 0.37× |
| gpu:2 | 9.20 | 0.58× |
| gpu:2+cpu:4 | 16.47 | 1.04× |

A GPU only pays off where a train step is much cheaper than on a CPU core.
That needs low launch overhead (newer GPU, faster host); batching all junction agents into one step
(Tier 2 in PLAN_TRAINING_PERF.md) would help most.
