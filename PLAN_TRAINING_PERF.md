# Training-pipeline performance review and plan (`tsc train`)

## Context
The user asked for a review of the training pipeline: find the bottlenecks, rank the fixes by
expected speedup against effort and risk, and flag anything that could change model results.
Follow-ups:
- Can training use the GPU, or the CPU and GPU together?
- Which setups must be supported?
  - **A**: the user's laptop: i7-8850H, Quadro P2000 4 GB.
  - **B**: a coworker's desktop: Ryzen 7 7800X3D, Radeon RX 7900 XT 20 GB.
  - **C**: NVIDIA RTX 4060 and newer.
- A reviewer's feedback on the first draft, incorporated here.

**Decisions made with the user:**
- Parity scope is **per machine, CPU**.
- **Milestone 1 = a parity harness, the CPU speedups that don't change results, and an opt-in GPU path.**
- **GPU priority: the RX 7900 XT (ROCm) first**, validated by the coworker. NVIDIA follows; the code is shared, only validation comes later.
- Everything goes on a new branch `perf/training-pipeline`, with one commit per step.

## Parity, defined (to be written into CLAUDE.md in M1)
1. **CPU reference parity (binding).** An optimization must reproduce the reference run exactly on
   **the same machine's CPU**. That covers the weights (online, target, optimizer state), the full
   training log, the checkpoint selection and the RNG streams. Every Tier 1 change must pass this.
2. **Same-device reproducibility (binding for any backend we support).** Repeated runs with the same
   seed on the same device and software stack give identical results. GPU backends must pass this before they are called supported.
3. **Cross-device / cross-machine equivalence (not guaranteed; diagnostic only).**
   This covers CPU vs GPU, NVIDIA vs AMD, and the laptop (AVX2) vs the 7800X3D (AVX-512).
   If two machines diverge, find the first diverging step and operation before concluding anything.
   Runs record the full software stack, so differences can be explained.

## What was measured (setup A)
- 6 cores, 12 threads; 5 single-threaded workers already almost fill the CPU.
  **Throughput therefore scales with CPU-seconds per run.**
- TF 2.21 is a CUDA build but runs CPU-only (the CUDA libraries are not installed).
- Method: cProfile of 2 training episodes plus 1 validation episode, run for step1/full, step2/full and corridor/dir_coop,
  plus `DQNAgent` micro-benchmarks (all outputs in the scratchpad).

Estimated cost of one step1/full training episode (~10.5 s, ~1800 `replay()` calls):

| Part | s / episode | Share |
|---|---|---|
| `replay()` TF compute (`_train_step` ≈ 3.4 ms, near compute-bound on one core) | ~6.1 | ~58 % |
| `replay()` overhead: 5× `convert_to_tensor`, separate `_soft_update` (0.4 ms), dispatch, deque sample + stacking | ~1.6 | ~15 % |
| `act()` greedy (0.54 ms/call; the predict itself is 0.15 ms) | ~0.3–0.9 | ~6 % |
| Python per-second loop (queue metric, `Reward.all`, `Observer.tick`, `observe`) | ~0.8 | ~8 % |
| SUMO `simulationStep` | ~0.6 (corridor at high demand: ~3.8) | 6 % (corridor ~25 %) |
| `randomTrips.py` subprocess | ~0.5 | ~5 % |
| Validation (amortised; routes regenerated each time) | ~0.6 | ~5 % |
| Checkpoints, CSV log, tripinfo parsing | <0.05 | ≈0 |

Measured facts behind the ranking:
- The emissions device costs 20–25 % of SUMO time.
  Without it, delay, arrivals, queue and the summed `multi` reward were identical.
- Fusing the soft update into the train step: bit-identical over 300 steps, ~2 % faster.
- Calling the concrete predict function: same output, 0.25 → 0.15 ms.
- `TSC_TF_THREADS=2`: bit-identical here; it only helps when cores are idle.
- XLA: no gain, and the weights differ. Rejected.
- Mixed precision / bf16: rejected.

All speedups below are **hypotheses until measured end-to-end**:
S = T_baseline / T_optimized, where both are full-episode times under the same conditions.

## Tier 1: no change to results (milestone 1)
Each item is its **own commit**, parity-checked on its own and then together with the others.

1. **Emissions device off in training and validation episodes**: an `emissions` kwarg on `Simulation`/`run_episode`;
   `evaluate` keeps it on. ~2 % (step1) to ~6 % (corridor).
   - **The agent's CO2 reward term is unaffected.** `multi` reads `lane/edge.getCO2Emission`, which SUMO computes from
     each vehicle's emission model with or without the device. Measured on test5: the summed reward was bit-identical.
   - The device only fills the per-vehicle tripinfo `<emissions>` record, i.e. the `co2_*` metric columns.
     So the training-log CO2 columns become 0, and the parity check ignores them by explicit rule.
     Nothing reads them: validation selects on delay.
   - Evaluation CO2 results (`results.csv`, plots, `tsc compare`) are unchanged, because evaluation keeps the device.
   - Extra check in this commit: the reward-sum comparison is repeated on the corridor network (it was only run on test5),
     and `mean_reward` stays covered by `tsc parity check` on all three configs.
2. **Faster `act()`**: `.numpy()[0]` instead of slicing the tensor, and the concrete predict function. ~4 %, more in validation and evaluate.
3. **Fused train step + soft update**, with numpy inputs (no 5× `convert_to_tensor`). The hard-update path is unchanged. ~3–5 %.
4. **Background route generation for the next episode** (a subprocess in a 1-thread executor, two alternating files).
   It must be proven not to touch Python's or NumPy's global RNG: the harness compares RNG-state hashes every episode.
   Never call randomTrips in-process. ~5 % (step1).
5. **Validation routes generated once** and reused. ~1 %.
6. **Halting counts computed once per second** and shared by `Simulation.step` and `Reward`, keeping the summation order. ~3–5 %.
7. *(Optional, last)* **Ring-buffer replay memory**, sampling with `random.sample(range(n), k)`. ~3–5 %, roughly 2× replay RAM.

Expected combined S ≈ 1.2–1.3× (hypothesis). The ceiling is set by the ~60 % that is real matmul work.

## Tier 2: changes model results (options only; the reference config stays `train_every=1` with the current network)
| Change | Expected S | Risk |
|---|---|---|
| Gradient step every k decisions | k=2 ≈ 1.6×, k=4 ≈ 2.2× | New learning dynamics. Biggest lever; needs a new baseline |
| Smaller network (256-256) | ~1.5–2× on the train step | Departs from the paper's architecture |
| Fewer episodes / early stopping; validating less often | Proportional / ~3–5 % | Changes `final_*` and the `best_*` selection |
| Numpy forward pass in `act()` | ~4 % | Argmax can flip on near-ties |
| Agents batched into one vectorised train step | Big on GPU | High effort; changes numerics |
| XLA, mixed precision, bf16, TF32 | None or small | Not recommended |

## GPU: hypotheses to validate (**RX 7900 XT is the priority target**)
Nothing in this table counts as "supported" until a training smoke test and a same-device repeatability test pass on real hardware.
The training code is backend-agnostic: TF exposes ROCm and CUDA devices alike as `GPU`.
So building for the 7900 XT first also covers NVIDIA in code; only the validation order changes.

| Hardware | Expected status | Basis | Main risk |
|---|---|---|---|
| **B: RX 7900 XT (gfx1100, 20 GB): PRIORITY** | Possible, validated by the coworker | AMD's ROCm matrix lists TF 2.21/2.20 + Python 3.12 for gfx1100 on Ubuntu 22.04/24.04 and RHEL; the validated Docker images stop at TF 2.20 | See the "RX 7900 XT specifics" section |
| A: Quadro P2000 (Pascal, CC 6.1, 4 GB) | Expected to work | The TF 2.21 build lists `sm_60` (which runs on CC 6.1); driver 580 | Old card, ~3.2 GB free (it also drives the desktop): ≤ 4 workers. Real kernels are untested |
| C: RTX 4060–4090 (Ada, CC 8.9) | Expected to work; **first NVIDIA target** | The build has native `sm_89` | TF32 on by default (turn it off); Windows only via WSL2 |
| C: RTX 50xx (Blackwell, CC 12.0) | Expected **not** to work with the stock 2.21 | No `sm_120` in the build; the bundled CUDA 12.5 predates Blackwell (needs ≥ 12.8); PTX-JIT failures reported | `tsc devices` must detect this and fall back to CPU |
| B: Ryzen 7800X3D (CPU) | Works | Stock wheel | CPU results may differ from the laptop's: diagnostic only (parity level 3) |

What the GPU can buy, end-to-end:
- On the laptop: if replay compute drops from 6.1 s to ~1 s and nothing else changes, an episode takes ~5.4 s.
  That is a **theoretical** S ≈ 1.9×; real-world will be lower (transfers, launches, contention).
- On the desktop the gain is smaller, because the stronger CPU already shrinks the network-compute share.
- Combining CPU and GPU means **hybrid scheduling**: whole seeds on the GPU, others on CPU cores.
  Splitting one small model across devices does not pay off.
  On B this means 8 Zen 4 cores plus a 20 GB card: memory allows many GPU workers.
  The limits are GPU time-slicing and each GPU worker's own CPU share (SUMO + Python, ~40 % of a core).

### RX 7900 XT specifics (setup B)
- **OS prerequisite.** ROCm TensorFlow is Linux-only and officially supports Ubuntu 22.04/24.04 and RHEL.
  There is no Windows path; WSL support for Radeon is limited and not planned for.
  **Open item: confirm the coworker's OS before they start.**
- **Install, option 1 (preferred): a venv.**
  1. Install the amdgpu driver and the ROCm userspace (`amdgpu-install`); the ROCm release must match the wheel's `rocm-rel-X.Y`.
  2. Add the user to the `render` and `video` groups.
  3. Run `uv sync`, then `uv pip uninstall tensorflow`, then
     `uv pip install "tensorflow-rocm==2.21.*" -f https://repo.radeon.com/rocm/manylinux/rocm-rel-X.Y/`.
  4. From then on, always run with `UV_NO_SYNC=1` / `uv run --no-sync`; otherwise uv reinstalls the stock CPU/CUDA wheel.
- **Install, option 2: Docker.** `rocm/tensorflow:rocm7.2.x-py3.12-tf2.20-dev` with
  `--device=/dev/kfd --device=/dev/dri --group-add video`, the repo mounted, the project installed `--no-deps`,
  and its other dependencies (`eclipse-sumo==1.28.0`, `libsumo==1.28.0`, …) installed by pip.
- **TF version.** If no 2.21 ROCm wheel exists, use 2.20.
  - GPU runs are never parity-compared to CPU runs (level 3), so this is acceptable.
  - The version goes into `run_info.json`.
  - `tsc devices` warns that the TF version differs from the pin.
- **The 7800X3D has an integrated GPU** (Raphael, gfx1036).
  ROCm can enumerate it next to the 7900 XT; this is known to cause crashes and hangs.
  - `devices.py` selects the GPU explicitly (`train.gpu_index`, default: the first *supported* gfx).
  - `tsc devices` lists every device it sees with its gfx target.
  - `docs/GPU.md` recommends `HIP_VISIBLE_DEVICES=<dGPU>` or disabling the iGPU in the BIOS.
- **Determinism on ROCm**, which the same-device repeatability level requires.
  - Some ops may have no deterministic ROCm kernel and then raise under `enable_op_determinism()`.
  - The likely one: the gradient of `tf.gather(..., batch_dims=1)` in `_train_step` (a scatter-add).
  - Contingency, only if that happens: replace it with `reduce_sum(q * one_hot(a), axis=1)`.
    That is exact in value on the CPU (one non-zero term per row), but it must still pass the
    CPU parity harness (level 1) before it is used.
- **Launch overhead.** ROCm kernel launches cost more than CUDA's, and the workload is launch-bound.
  The single-process S may therefore be modest; the payoff is mainly extra GPU workers next to 8 CPU workers.
  The full-episode bench decides.

## Milestones

### M1: parity harness → RX 7900 XT GPU path → Tier 1 (this implementation)
**Order rationale.**
- The 7900 XT is the priority, but it can only be validated on the coworker's machine, which adds round-trip latency.
- So the GPU path ships right after the parity harness (which its repeatability test needs).
- While the coworker tests, the Tier 1 CPU items are implemented here. They speed up GPU runs too (act, routes, SUMO, Python loop).
**Branch.** All M1 work goes on a new branch `perf/training-pipeline`, created from `main` at `9d69577`.
It was later rebased onto `main` `a1057e9` (2F–2I); see "Rebase onto 2F–2I" under Progress.
Nothing is committed to `main`.
- One commit per step below, so each optimization can be tested, bisected or reverted on its own.
- The untracked `Deep-Qlearning-TSC/` directory stays untouched and out of commits.
- No push or PR unless the user asks; when the user is happy with the test results, the branch can be pushed and merged with a PR.

**Step 0: the parity harness, committed before any optimization.**
- `tests/reference_dqn.py`: a verbatim frozen copy of today's `DQNAgent`, used only as a test oracle.
  After the rebase it is the copy of `main` `a1057e9`'s agent, with every 2F–2I option.
- `tests/test_dqn_parity.py` (unit, no SUMO):
  - The reference and new agents are built under identical seeding; their initial weights must be identical.
  - Both get the same transitions and then run 200 `replay()` steps in each mode: soft/double, hard/non-double, hard/double.
  - After **every** step it compares: the sampled batch, the `random.getstate()` and `np.random.get_state()` hashes,
    the loss, the online weights, the target weights, and the Adam slots and iteration counter.
  - It also compares ε-greedy `act()` actions and RNG states, and (if item 7 is done) the ring-buffer contents.
- `src/tsc/experiments/parity.py` and the CLI command `tsc parity record|check <config> [--episodes N]`:
  - It calls `train_one` through a new optional `on_episode_end` hook, which defaults to `None`, does not consume RNG and is otherwise inert.
  - Per episode it records:
    - the full `train_log` row (without `seconds`; CO2 excluded by explicit rule);
    - whether this episode's validation improved `best_*`;
    - SHA-256 hashes of every agent's online, target and optimizer variables;
    - hashes of the RNG states (`random`, `np.random`, the trainer rng, the observer rng);
    - the SHA-256 of the route file;
    - the timing, which is recorded but not compared.
  - It also records the software stack.
  - `check` reruns and reports the **first differing episode and field**.
- **Golden reference**: run `tsc parity record` on the hook-only commit and confirm it matches plain `main` (identical `train_log`).
  Store it under `outputs/parity/<hostname>/` (gitignored, machine-specific).
  Re-run it after any future TF or SUMO upgrade.
- Three configs cover the branches:
  - `step1/full`: random_trips, grid observations, `multi` reward, soft/double.
  - `step1/base`: hard/non-double, `wait` reward, no comm.
  - `corridor/dir_coop`: od demand, detector observations with noise/dropout, directional messages, cooperative reward.
  - Added after the rebase, for the 2F–2I training paths:
    - `plan/dir_shared`: plan control, the plan library, `plan_rng`, a shared agent, tidal od demand.
    - `hierarchy/workers/corridor_greenwave`: a fixed expert with residual workers, `delay` reward,
      the expert-alone validation before training, acceptance at the end.
  - Each runs 6 episodes with `val_every=2`.

**Step 1: opt-in GPU device path, ROCm first (default stays CPU, so CPU parity is unaffected).**
- [config.py](src/tsc/config.py): `TrainConfig.device = 'cpu'` (`'cpu' | 'gpu'`) and `TrainConfig.gpu_index: int | None = None`, both validated.
- New `src/tsc/devices.py`, called from `train_one`:
  - **Before the TF import**, with the device set to CPU: set `CUDA_VISIBLE_DEVICES=""` and `HIP_VISIBLE_DEVICES=""`.
    This guards against TF silently picking up a GPU.
  - **After the import and before seeding** (the import-before-seed order is kept), with the device set to GPU:
    - detect the backend (`is_rocm_build` / `is_cuda_build`);
    - pick the GPU: `gpu_index`, otherwise the first supported one (on ROCm, skip unsupported gfx targets such as the iGPU);
    - `set_visible_devices` to that GPU;
    - turn on memory growth and `enable_op_determinism()`;
    - on CUDA only: turn off TF32 and run the compute-capability check (the RTX 50xx message).
  - None of these calls consume RNG.
  - `info()` returns the backend and versions (ROCm/HIP or CUDA/cuDNN), GPU name and arch, TF version, CPU model and ISA,
    and Python/NumPy/Keras/SUMO versions. It goes into `save_run_info` in [common.py](src/tsc/experiments/common.py).
- [cli.py](src/tsc/cli.py):
  - `tsc devices`: the backend, every visible device with its arch, and a verdict per device (verified, unverified or unsupported, with the reason).
    It also warns about a TF version that differs from the pin, and an iGPU that is visible.
  - `tsc bench <config> --mix "gpu:2,cpu:6" [--episodes 10 --repeats 3]`: **full training episodes** under each worker mix.
    It reports episodes/min, the median and IQR of episode time, peak GPU memory (`rocm-smi` / `nvidia-smi` when present),
    CPU utilisation and clocks. It writes `bench.json` for sharing.
- `train --gpu-workers K`: the first K seed tasks get `device=gpu` (recorded in each run's `config.toml`), the rest run on the CPU.
  This is the hybrid scheduling; per-machine defaults come only after measurement.
- **Evaluation policy**: `evaluate` runs on the CPU as an **explicit, documented reproducibility policy** (standardised inference),
  configurable through `evaluate.device`. A test checks that weights save and load across devices.
- New `docs/GPU.md` contains:
  - the ROCm install for setup B (both options, the iGPU note, `UV_NO_SYNC`);
  - the CUDA install (`cuda` extra; WSL2 note; RTX 50xx unsupported);
  - **the coworker's validation checklist**, in this order:
    1. `tsc devices`
    2. `pytest -m "not sumo"`
    3. a CPU `tsc parity record` on step1/full (their CPU golden)
    4. a GPU `tsc parity record` run **twice**, then `tsc parity check` between the two (same-device repeatability)
    5. `tsc bench --mix` with gpu:1, gpu:2, gpu:4, cpu:8, gpu:2+cpu:6 and gpu:4+cpu:6
    6. send back `bench.json`, the manifests and `run_info.json`
- `pyproject.toml`: an optional `cuda` extra (NVIDIA). ROCm stays a documented manual install, because its wheel conflicts with the locked `tensorflow`.
- Unit tests (no GPU needed): device selection with mocked device lists (iGPU plus 7900 XT; RTX 50xx capability → unsupported);
  the CPU guard sets the environment variables before the import; config validation.
- **Hand-off**: when this commit is done, the branch is ready for the coworker. Push it only when the user says so.

**Steps 2–8: one Tier 1 item per commit** (items 1–7 above).
After each one, run `pytest -m "not sumo"` (including the parity unit test) and `tsc parity check` on all three configs.
Items 4 and 6 are verified alone first, then together.

**Step 9: documentation.**
- CLAUDE.md: the three parity levels, the parity-harness commands, the golden-run procedure, the `device` option and a pointer to `docs/GPU.md`.
- CHANGELOG: a performance and GPU section (not part of F1–F11) noting the CO2 training-log columns.

Files:
- [dqn.py](src/tsc/agents/dqn.py)
- [simulation.py](src/tsc/simulation.py)
- [reward.py](src/tsc/reward.py)
- [train.py](src/tsc/experiments/train.py)
- [evaluate.py](src/tsc/experiments/evaluate.py)
- [common.py](src/tsc/experiments/common.py)
- [config.py](src/tsc/config.py)
- [cli.py](src/tsc/cli.py)
- `pyproject.toml`
- new `src/tsc/devices.py`
- new `src/tsc/experiments/parity.py` and `src/tsc/experiments/bench.py`
- new `tests/reference_dqn.py`, `tests/test_dqn_parity.py` and `tests/test_devices.py`
- new `docs/GPU.md`
- CLAUDE.md
- CHANGELOG.md

### M2: RX 7900 XT validation and go/no-go (on the coworker's machine, with their results)
- **Accept** the 7900 XT as "verified" only if:
  - the smoke test passes;
  - two same-seed GPU runs give identical parity manifests (level 2);
  - the best `--mix` reaches **sustained end-to-end S ≥ 1.3×** over the best CPU-only worker count on the same machine.
- If determinism fails, apply the one-hot contingency (which must first pass CPU parity here), then re-test.
- If S < 1.3×, the 7900 XT stays "works but not recommended"; the code stays opt-in.
- If it passes, record the recommended mix (e.g. `--workers 8 --gpu-workers k`) in `docs/GPU.md`.

### M3: a robust CPU baseline across machines
- Run the cross-machine diagnostic: compare the laptop's and the desktop's CPU manifests.
- If they diverge, localise the first diverging step with the unit-test harness and try `ONEDNN_MAX_CPU_ISA=AVX2`.
  The outcome is documented, not enforced.
- Use `tsc bench` to tune the CPU-only `--workers` per machine (A: 4–6; B: 6–8).

### M4: NVIDIA verification
- An RTX 4060-class card, then the P2000 here: the same checklist as M2 (`uv sync --extra cuda`).
- The RTX 50xx stays "unsupported" until a TF wheel ships `sm_120` kernels.

## First actions after approval (before any code)
1. Copy this plan, verbatim, to the project root as `PLAN_TRAINING_PERF.md`.
2. `git switch -c perf/training-pipeline main`. The uncommitted plan file comes along to the new branch.
3. Commit `PLAN_TRAINING_PERF.md` alone as the branch's first commit ("Add training performance / GPU plan"),
   so the plan is versioned next to the changes it describes.
   Keep it updated as milestones complete, in the same way as `docs/PLAN_REALISM.md`.

## Verification (M1, all on branch `perf/training-pipeline`)
0. Check `git status` before each commit (only the intended files; never `Deep-Qlearning-TSC/` or `outputs/`).
1. On the hook-only commit, `tsc parity record` gives a `train_log` identical to plain `main` for all three configs.
2. After the GPU-path commit (step 1):
   - `tsc parity check` on all three configs passes with the default `device=cpu`, which proves the GPU path is inert;
   - `pytest -m "not sumo"` passes, including `test_devices.py`;
   - `tsc devices` runs here (it reports the P2000 as unverified, or CPU-only without the CUDA libraries);
   - `tsc bench --mix cpu:5 --episodes 2 --repeats 1` produces a valid `bench.json`.
3. After each Tier 1 commit:
   - `uv run pytest -m "not sumo"` passes, including the per-step `test_dqn_parity.py`;
   - `tsc parity check` passes on all three configs, with no differing field (except the declared CO2 exclusion).
4. After all commits:
   - the full `uv run pytest` passes;
   - the combined `tsc parity check` passes;
   - the timing is measured: ≥ 10 episodes × 3 repetitions, before vs after, on step1/full and corridor/dir_coop.
     Report the median S per item and combined; it is expected near 1.2–1.3× but reported as measured.

## Progress

- **Step 0 (parity harness): done.**
  - Recordings with the inert hook reproduce plain `main` exactly on step1/full, step1/base and corridor/dir_coop
    (train_log plus all `best_*`/`final_*` weights).
  - Route hashes ignore the generators' header comments, which hold a timestamp and paths.
  - Golden manifests live in `outputs/parity/<host>/*/golden`.
- **Step 1 (opt-in GPU path, `tsc devices`, `tsc bench`, `--gpu-workers`, `docs/GPU.md`): done.**
  - CPU parity check: IDENTICAL on all three configs.
  - On the P2000 (CUDA libraries as an overlay):
    - two same-seed GPU runs are IDENTICAL (level 2);
    - GPU vs CPU first diverge in episode 1 (level 3, as expected);
    - the first bench gave S ≈ 1.04 for the best mix, because `replay()` on the GPU is only 1.35× faster than on a CPU core (launch-bound).
  - So the P2000 is "works, not recommended".
  - RX 7900 XT: waiting for the coworker's checklist (`docs/GPU.md`).
- **Tier 1 items 1–6: done, one commit each.** Every `tsc parity check` was IDENTICAL on all three configs,
  every field: weights, optimizer, RNG and route hashes, train_log, checkpoint choice.
  - Checks were cumulative after each item.
  - Item 6 was also checked on its own (on top of items 1–3).
  - `tests/test_dqn_parity.py` passes step by step for items 2–3.
- **Item 7 (ring buffer): skipped.**
  - At a full 50k buffer, sampling drops 0.265 → 0.10 ms per step (~3.5 %).
  - But each agent adds ~360 transitions per episode, so the buffer only fills near episode 140 of 150: ~2 % on average.
  - The cost would be about 2× replay RAM.
- **Measured end-to-end S (step1/full, 10 episodes, `main` vs branch on the same seeds, concurrently):**

  | load | config | training episode, median | 10 episodes incl. validation |
  |---|---|---|---|
  | 6 concurrent runs | step1/full | **1.17×** (23.9 → 20.4 s) | 1.11× |
  | 6 concurrent runs | corridor/dir_coop | **1.10×** (37.9 → 34.3 s) | 1.09× |
  | 2 concurrent runs | step1/full | **1.18×** (12.55 → 10.60 s) | 1.14× |

  - Below the 1.2–1.3× hypothesis: the remaining time is dominated by real matmul work in the train step.
  - The totals understate the gain: the branch runs finish first, so `main`'s final episode + validation run on a less loaded machine.
  - Bigger gains need Tier 2 changes (e.g. `train_every`), which change results.
- **Rebase onto 2F–2I (`main` `a1057e9`, 2026-10-16): done; parity re-verified.**
  - The rebase applied cleanly in substance: `d` (the discount column) flows through the fused train step,
    `emissions=False` reaches the plan/hierarchy validation, and the prefetch peek stays valid
    because plan choice draws from its own `plan_rng`.
  - `tests/reference_dqn.py` is now the frozen copy of `a1057e9`'s agent.
    `tests/test_dqn_parity.py` also covers SMDP discounts, n-step (two keys plus `end_episode`), dueling,
    ref_action and phase_net. All 12 cases are bit-identical.
  - Parity harness: it also hashes `plan_rng`; runs without RL agents no longer crash the hook.
  - The old goldens (recorded on `9d69577`) are kept in `outputs/parity/<host>_pre_rebase_9d69577/`.
    New goldens were recorded on `97ebc77` (harness only, plus the hook fix above).
  - `tsc parity check` at the tip: IDENTICAL on step1/full, step1/base, corridor/dir_coop,
    plan/dir_shared and hierarchy/workers/corridor_greenwave (accept.json identical too).
  - Fixed: a policy loaded inside a training run (RL experts, expert workers, acceptance at the end
    of hierarchy runs) forced the CPU. A GPU run would have crashed there, because one process
    cannot switch devices. `load_policy(device=None)` now keeps the process's device.
  - `accept()` generates its acceptance routes once for both policies.
- **Open:**
  - M2: the coworker runs the RX 7900 XT checklist (`docs/GPU.md`).
  - M3: cross-machine CPU diagnostic.
  - M4: RTX 40xx check.

## PyTorch port (branch `torch-port`, from `perf/training-pipeline`; tsc 0.3.0)

Plan: switch the agents from TensorFlow to PyTorch, for GPU support (CUDA and ROCm wheels,
native Windows CUDA, RTX 50xx), speed, and a team requirement. Old weights are dropped; every
RL run is retrained with PyTorch. `torch-port` is merged back into `perf/training-pipeline`.

**Done** (one commit each):
- TensorFlow reference fixtures (`tests/fixtures/make_tf_reference.py`, `tf_reference/*.npz`),
  recorded on the TensorFlow agent: 8 modes, 60 gradient steps each.
- The PyTorch agent: the same API and the same numerics as Keras.
  - Glorot-uniform init with zero biases; Keras' Adam (ε added to √v) with per-weight clipnorm; Huber loss.
  - `tests/test_tf_equivalence.py` matches every mode at rtol 1e-4 / atol 1e-5.
    The correct agent stays below 0.43 of that tolerance; PyTorch's own ε placement exceeds it 18–870×, and global clipping 2600–8300×.
  - Weights are saved as `*.pt`; TensorFlow runs raise a clear error.
  - `tests/reference_dqn.py` is the frozen PyTorch agent.
- Device setup for PyTorch:
  - deterministic algorithms, TF32 off, `CUBLAS_WORKSPACE_CONFIG`;
  - `torch_device()`;
  - arch verdicts from `torch.cuda.get_arch_list()`.
- `run_info` records the PyTorch build (CPU kernel level, CUDA/HIP/cuDNN).
- Dependencies: TensorFlow is removed. `uv sync` installs the CPU build; the groups `cuda` (cu130), `cuda126` and `rocm` (rocm7.2) install the GPU builds.
- Speed: the first agent's `replay()` was 1.66× slower than TensorFlow's, because the op-by-op Adam took 2.9 ms.
  With fused Adam (ε/√(1−β₂ᵗ) per step) and a single `_foreach_lerp_` Polyak update:
  `replay()` 3.80 ms vs TensorFlow 4.34 ms; greedy `act()` 0.087 vs 0.254 ms
  (step1/full network, one thread, this laptop).
- The full test suite passes without TensorFlow installed (258 tests).
- `tsc devices` on the P2000 with `--group cuda126`: detected, and it runs the `sm_60` kernels.
- CPU same-seed reproducibility: two PyTorch `tsc parity record` runs are IDENTICAL on step1/full,
  step1/base, corridor/dir_coop, plan/dir_shared and hierarchy/workers/corridor_greenwave.
  The first recordings are the new goldens in `outputs/parity/<host>/`.
  The TensorFlow goldens of `a1057e9` are kept in `outputs/parity/<host>_tf_a1057e9/`.
- Indicative end-to-end speed: the parity recordings, 6 episodes each, with 5 recordings side by side for both frameworks.
  - step1/full: 141 s with TensorFlow vs 83 s with PyTorch (1.7×).
  - corridor/dir_coop: 285 s vs 179 s (1.6×).
  - The runs start from different initial weights and are short, so the proper comparison is item 2 below.

**Postponed verification** (about 4–5 h of compute; run it before relying on PyTorch results):
1. Learning quality, TensorFlow vs PyTorch, 3 seeds each, trained side by side.
   - This needs a TensorFlow environment: `git worktree add ../tsc-tf perf/training-pipeline`, then `cd ../tsc-tf && uv sync`.
   - In each tree (TensorFlow: `../tsc-tf`; PyTorch: this repo), with `<fw>` = `tf` or `torch`:
     ```bash
     .venv/bin/tsc train configs/step1/full.toml --seeds 0-2 --workers 3 --set name=cmp_<fw>_full --out <repo>/outputs/train/cmp_<fw>_full
     .venv/bin/tsc evaluate configs/step1/evaluate.toml --runs "<repo>/outputs/train/cmp_<fw>_full/s*" --no-baselines --workers 3 --out <repo>/outputs/eval/cmp_<fw>_full
     ```
   - The same for `configs/corridor/dir_coop.toml` with `configs/corridor/evaluate.toml`.
   - Then compare: `tsc compare outputs/eval/cmp_tf_full/results.csv outputs/eval/cmp_torch_full/results.csv --ref cmp_tf_full`.
     The expectation is no significant difference in delay.
   - Also compare the training curves (`configs/v6/plot_training.py`) and the episode times in `train_log.csv` (end-to-end speed under equal load).
2. Speed: `tsc bench configs/step1/full.toml --mix cpu:5` in both trees.
3. P2000 GPU (`uv sync --no-group cpu --group cuda126`, `UV_NO_SYNC=1`):
   - the agent microbenchmark (`replay`/`act` on GPU vs CPU);
   - two same-seed GPU `tsc parity record` runs, then `tsc parity compare` (they must be IDENTICAL);
   - `tsc bench --mix gpu:2,cpu:4`.
4. The coworker's RX 7900 XT checklist (`docs/GPU.md`, `--group rocm`).
