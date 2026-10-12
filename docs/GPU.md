# Training on a GPU (opt-in)

Training runs on the **CPU by default**. The CPU is the bit-exact reference, as described in
CLAUDE.md (parity level 1). A GPU can train some or all seeds instead:

```bash
uv run tsc train configs/step1/full.toml --seeds 0-4 --set train.device=gpu   # all seeds on the GPU
uv run tsc train configs/step1/full.toml --seeds 0-6 --workers 8 --gpu-workers 2  # hybrid
uv run tsc devices          # what TensorFlow sees, with a verdict per GPU
```

`--gpu-workers K` trains the first K seeds on the GPU and the others on the CPU.
Each run records its device in its `config.toml` and `run_info.json`.

## What changes with a GPU

- **Results differ from CPU results.** cuBLAS and rocBLAS sum in a different order than the CPU kernels.
  So a GPU run is not bit-identical to the CPU run of the same seed, nor to a run on another GPU model (parity level 3).
  Do not mix devices within one experiment without saying so.
- **Repeated runs on the same GPU are identical** (level 2). The device setup (`src/tsc/devices.py`) enables op determinism
  and turns TF32 off on NVIDIA. A GPU only counts as verified once this has been checked (see the checklist below).
- **Evaluation stays on the CPU** (`evaluate.device = "cpu"`, the default), whatever a policy was trained on.
  This is a reproducibility policy: evaluation results then do not depend on the GPU model.
- **SUMO and the Python loop always run on the CPU.** A GPU only speeds up the Q-network updates (~60–70 % of an episode).
  Each GPU worker still needs roughly half a CPU core.

## Support status

| GPU | Status | Notes |
|---|---|---|
| AMD Radeon RX 7900 XT (gfx1100) | **priority target**, not yet verified | ROCm; Linux only (see below) |
| NVIDIA RTX 40xx (sm_89) | expected to work, not yet verified | `uv sync --extra cuda` |
| NVIDIA Quadro P2000 / Pascal (sm_61) | **works, not recommended** | Repeatable: two same-seed runs were IDENTICAL. But `replay()` takes 3.15 ms vs 4.26 ms on a CPU core, and the first bench gave S ≈ 1.04 at best (gpu:2+cpu:4). See "Measurements" |
| NVIDIA RTX 50xx (Blackwell, sm_120) | **unsupported** with TensorFlow 2.21 | no sm_120 kernels in the build. `tsc devices` reports it, and `device=gpu` stops; `TSC_ALLOW_PTX_JIT=1` tries the PTX JIT at your own risk |
| AMD integrated GPUs (e.g. Ryzen 7000 "Radeon Graphics", gfx1036) | unsupported | must be hidden on ROCm (see below) |

A GPU moves to **verified** (`devices.VERIFIED`) only after the checklist below passes on it.

## NVIDIA (CUDA)

```bash
uv sync --extra cuda        # adds the CUDA 12 / cuDNN 9 runtime libraries (~3 GB) as pip packages
uv run tsc devices
```

- You need a Linux driver ≥ 525. No system CUDA toolkit is required.
- On Windows, use WSL2: TensorFlow has not supported GPUs on native Windows since 2.10.
- TensorFlow allocates GPU memory on demand (memory growth is on).
  Each process still holds ~0.3–0.5 GB of CUDA context, which limits GPU workers on small cards.

## AMD (ROCm): RX 7900 XT

**Requirements**:
- Linux. AMD supports Ubuntu 22.04/24.04 and RHEL; other distributions may work but are unofficial.
  There is no ROCm TensorFlow for Windows.
- Python 3.12.
- The ROCm stack installed (`amdgpu-install`, see AMD's ROCm install guide).
- Your user in the `render` and `video` groups.

### Option 1: project venv (preferred)

`tensorflow-rocm` replaces the stock `tensorflow` package, which the lockfile pins, so it is installed by hand.
After that, **uv must not re-sync the environment**; otherwise it puts the stock wheel back.

```bash
uv sync
uv pip uninstall tensorflow
# X.Y = your ROCm release; check that the index lists a 2.21 wheel, otherwise use 2.20.*
uv pip install "tensorflow-rocm==2.21.*" -f https://repo.radeon.com/rocm/manylinux/rocm-rel-X.Y/
export UV_NO_SYNC=1         # or pass --no-sync to every `uv run`
uv run tsc devices          # must say "rocm build" and list the 7900 XT
```

If only TF 2.20 is available for your ROCm release, that is acceptable for GPU runs. They are never
compared bit for bit with CPU runs, and the version is recorded in `run_info.json`; `tsc devices` warns about it.

### Option 2: AMD's Docker image

```bash
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video \
    -v "$PWD":/work -w /work rocm/tensorflow:rocm7.2.4-py3.12-tf2.20-dev
pip install --no-deps -e . && pip install eclipse-sumo==1.28.0 libsumo==1.28.0 traci==1.28.0 \
    sumolib==1.28.0 pandas scipy matplotlib tabulate tomli-w pytest
```

### Integrated GPU (Ryzen 7000 desktop CPUs)

The 7800X3D has an integrated Radeon GPU (gfx1036).
- ROCm may enumerate it next to the 7900 XT, which is known to crash or hang ROCm programs.
- Hide it with `export HIP_VISIBLE_DEVICES=<index of the 7900 XT>`
  (check `rocminfo` or `rocm-smi`), or disable the iGPU in the BIOS.
- `tsc devices` lists every device with its gfx target and marks integrated ones as unsupported.
- `train.gpu_index` selects a GPU explicitly.

## Validation checklist (send the results back)

Run everything from the repository root on the branch under test. On ROCm, keep `UV_NO_SYNC=1` set.
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
