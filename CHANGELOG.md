# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Version 0.3.0.

### Breaking
- **The RL agents use PyTorch instead of TensorFlow/Keras** (`tsc.agents.dqn`, torch 2.14).
  The networks, initialisation (Glorot-uniform, zero biases), Keras' Adam update with
  per-weight gradient clipping, Huber loss and DQN targets are the same:
  `tests/test_tf_equivalence.py` replays recorded TensorFlow runs of every agent mode and
  matches them to float32 rounding. Results are still not bit-identical to TensorFlow runs
  (different random initial weights for a seed, different summation order), so **every
  RL run must be retrained**; earlier numbers stay valid as TensorFlow results.
- Weights are saved as PyTorch state dicts, `{best,final}_<tl>.pt`. Runs trained with
  TensorFlow (`*.weights.h5`, including the pre-package scripts' runs) cannot be loaded;
  `tsc` says so instead of failing obscurely.
- Dependencies: TensorFlow is gone. `uv sync` installs PyTorch's CPU build; GPU builds are
  dependency groups (`uv sync --no-group cpu --group cuda | cuda126 | rocm`, docs/GPU.md),
  which replaces the manual `tensorflow-rocm` install and the `cuda` extra.
- `TSC_THREADS` sets the threads per process (`TSC_TF_THREADS` is still read as a fallback).

### Added
- Step 2I (v6), results in `docs/V6_REPORT.md`:
  - RL controllers as hierarchy experts: `rl` / `rlf` in `[signals] experts`, run directories
    in `[signals] rl_experts` (`controllers.select.RLExpert`); an RL expert listed without a
    run is a placeholder (same worker input size), skipped by the table.
  - Phase-wise Q-network shared by all junctions (FRAP / MPLight style): `agent.phase_net`
    with `observation.layout = "phase"` (one feature row per green stage: served lanes,
    downstream queue, signal state).
  - Residual worker head `agent.ref_action`: Q(s, ref) = V(s), Q(s, a) = V(s) + A(s, a) with A
    starting at zero, so an untrained worker acts as its expert.
  - `train.eps_start` (first-episode ε) and `train.val_per_level` (validation scenarios per
    level).
  - `configs/v6/`: config generator (`make_configs.py`), job generator (`make_jobs.py`), job
    queue with dependencies, priorities and adjustable slots (`run_queue.py`), candidate
    workers for the table (`admit_candidates.py`), benchmark summary vs published numbers
    (`summarize.py`).
  - `docs/BENCHMARK_HANDOFF.md` (papers and improvement list behind 2I and the benchmarks);
    RESCO Table 1 in `docs/PLAN_REALISM.md`.
- Hierarchy (2H), manager side: `delay` reward (time lost on the approaches, the evaluated
  quantity); `[hierarchy] default_expert` (`"auto"`: best single expert on validation) with a
  Q-margin and a fallback checkpoint; `tsc accept` (best checkpoint vs its default on fresh
  scenarios, seeds 750000+, one-sided Wilcoxon) and `[hierarchy] accept_scenarios`.
- `[hierarchy] manager = "table"`: lookup-table manager built by running every expert at a grid
  of demand levels, choosing by measured network flow and jam per 5-min reading
  (`table_levels`, `table_scenarios`, `hysteresis`); configs in `configs/hierarchy/v3c/`.
- Hierarchy configs `configs/hierarchy/v2/` (DQN manager, delay reward, default expert) and
  `evaluate_corridor_opposite.toml`; results in `docs/HIERARCHY_REPORT.md`.
- `networks/benchmarks/fetch.py`: downloads the RESCO benchmark scenarios (not committed).
- `maxpressure_det` baseline: max-pressure on the E2 detectors only, with the experiment's
  detector noise / dropout (`configs/step2/baselines.toml`, with the safety shield).
- `tsc compare`: paired comparison against a reference controller (relative difference per
  scenario, 95 % CI, Wilcoxon signed-rank test).
- `tsc diagnose messages`: evaluates communicating policies with real, zeroed and
  wrong-neighbour messages (`Observer.set_message_mode`).
- Arterial corridor network (`networks/corridor/`, 7 signals, left-turn bays, cross and
  residential streets, open boundaries) with configs in `configs/corridor/`.
- Demand generator `od`: trips between boundary points with a through-traffic share,
  a gravity model and a tidal direction drawn per scenario.
- `greenwave` baseline (common-cycle Webster + tlsCoordinator offsets); Webster and
  green-wave plans are made per tidal direction.

- Directional messages (`observation.message = "directional"`): what each neighbour knows
  about the receiver (queue / releases toward it, whether it serves it, spillback from it).

- Cooperative reward (`reward.neighbour_weight`): r_i + α·Σ r_neighbours.

- Expert-specific workers (2H.5): `[hierarchy] worker = "expert"` with `worker_runs` (each
  expert's worker from its own run, `configs/hierarchy/workers/`), `table_alone` (the table
  decides per level whether the worker acts); configs `configs/hierarchy/v4/`, `v5/`.
- Benchmark scenarios (`configs/benchmarks/`): demand from route files (`demand.generator =
  "file"`, `route_files`, `begin`, `count_undeparted`), the scenario's own signal programs,
  `trip_time` metric; results in `docs/BENCHMARKS.md`.
- `advanced_mp` baseline (Advanced-MP, Zhang et al. 2022; `signals.amp_weight`, `amp_range`).
- `observation.approach_length`: a second, longer area detector; vehicles moving toward the
  stop line become a state feature.
- `agent.smdp` (time-based discounting, γ per second), `agent.dueling`, `agent.n_step`.
- Reward `queue_pressure` (`reward.pressure_weight`).
- `signals.red_call`: the max-red shield per signal link and only for links where a detector
  reports a waiting vehicle.
- `tsc accept` for worker runs; inter-process file lock for files parallel jobs share
  (`sumo_env.file_lock`).
- Parity harness: `tests/test_dqn_parity.py` checks the agent against a frozen copy of the
  agent before the speedups (every agent option) after every gradient step; `tsc parity record|check|compare` fingerprints a short
  training run (log rows, weight/optimizer/RNG/route hashes) and reports the first difference.
- Opt-in GPU training (`train.device = "gpu"`, CUDA or ROCm; `src/tsc/devices.py`, docs/GPU.md):
  deterministic algorithms, TF32 off; GPUs whose architecture the installed PyTorch build
  lacks and AMD integrated GPUs are refused. `tsc devices`, `tsc bench` (episodes/min under
  CPU/GPU worker mixes, peak GPU memory) and `tsc train --gpu-workers K` (hybrid
  scheduling). GPU runs are reproducible on the same GPU but not bit-identical to CPU runs.
- `run_info.json` records the software stack (packages, CPU and SIMD flags, the PyTorch
  build and its CPU kernel level) and the training device.

### Removed
- RL controller selection per junction (2G: `[signals] control = "select"`, `select_interval`,
  `configs/select/*_comm*`, `*_nocomm*`): it failed on the corridor and on Maribor and the
  hierarchy replaces it. The expert machinery (`controllers.select.ExpertPool`, renamed from
  `SelectorActions`) and the best fixed assignment (`tsc assign`) stay. Earlier runs'
  `select_interval` is ignored when loading (`config.RETIRED`).

### Fixed
- Directional messages on networks with gaps in SUMO's link indices (Ingolstadt 21).
- Parallel jobs no longer race on the evaluation routes / plans and the static network.

### Changed
- Rewards, the queue metric and exact-count max-pressure cover whole approaches, including
  the road before a lane split (e.g. a left-turn bay). No effect on networks without splits.
- Faster training with bit-identical results (parity-checked on step1/full, step1/base,
  corridor/dir_coop, plan/dir_shared and hierarchy/workers/corridor_greenwave): no SUMO emissions device in training/validation episodes (the
  `co2_*` columns of `train_log.csv` are now 0; the CO2 term of the `multi` reward is
  unaffected), a faster greedy `act()`, the soft target update fused into the gradient step,
  the next episode's routes generated in the background, validation routes generated once,
  and halting counts computed once per second. Evaluation output is unchanged.
- Trained policies are evaluated on the CPU (`evaluate.device`), whatever they were trained on.

## [0.2.0] - Unreleased

The codebase is now a single Python package (`tsc`) and the experiments are made
realistic (steps 1 and 2 of [docs/PLAN_REALISM.md](docs/PLAN_REALISM.md)).
**Results are not comparable with v0.1.0 or the paper.** The fixes below change
what the reported numbers mean (see *Fixed*).

### Added
- `tsc` package with `pyproject.toml` (uv, Python 3.12) and the `tsc` command:
  `tsc train`, `tsc evaluate`, `tsc plot`, parallel over seeds or evaluation tasks.
- One TOML config per experiment (`configs/`), validated (unknown keys are errors),
  with `--set section.key=value` overrides. Every run saves its resolved
  `config.toml` and `run_info.json` (package/SUMO version, git commit) next to the results.
- Signal plans rebuilt with netconvert into conflict-free green stages with 3 s
  yellow + 2 s all-red, the same for every controller; `SignalHead` enforces min
  green, intergreens and an optional max-green / max-red safety shield
  (overrides are counted).
- Realistic Poisson demand in veh/h (`flat` / `peak` profiles, one-hour episodes)
  and a demand sweep of 600–3600 veh/h.
- Baselines: fixed time, Webster-optimised fixed time, SUMO actuated, max-pressure.
- Standard metrics from SUMO tripinfo/statistics: delay per vehicle (time loss +
  insertion delay, unfinished vehicles included), waiting time, stops, travel time,
  throughput, CO₂, mean queue.
- Detector-based observation (E2 area detectors + stop-line loops) with Gaussian
  noise and detector dropout; detector-computable `det_queue` reward.
- Double DQN and soft/hard target-update switches for ablations (`base`, `full`,
  `full_nocomm`, `queue`, `det_queue`); action masking to the junction's valid stages.
- Training on randomised demand with validation-based model selection.
- Evaluation of baselines and trained runs on the same held-out scenarios; plots
  and summary tables with 95 % confidence intervals.
- Explicit, non-overlapping seed ranges for training / validation / calibration /
  evaluation scenarios; training seeds ≥ 7 are rejected because they would
  overlap the other ranges.
- Run directories of the pre-package `train.py` (`config.json`) can be evaluated directly.
- pytest suite: signal logic, observation shapes, metrics, configs, and short SUMO
  runs of every controller type.
- Networks in `networks/`: `test5` (the paper's 5-junction network) and
  `single_4l` (one junction with 4-lane arms). Nothing in the code assumes either.

### Changed
- SUMO comes from the pinned `eclipse-sumo` package (1.28.0) or `SUMO_HOME`; no
  hardcoded paths. libsumo is used by default (TraCI for the GUI or `TSC_TRACI=1`).
- Communication messages and observations use fixed physical normalisation
  constants (lane capacity, speed limit) instead of the number of vehicles.
- Generated files and results go to `outputs/` (gitignored); results are no longer committed.
- README rewritten; the plan moved to `docs/`.

### Removed
- The paper and single-agent code (`SumoEnv`, `TLAgent`, `TLAgentMA`,
  `MultiSumoEnv`, `NetworkParser`, `CommunicationModule`, `TrafficGenerator`,
  `Model`, `main.py`, `main_multiagent.py`, `utils`, `helpers`) and the committed
  results. They remain available in v0.1.0. The paper's 4-phase / grid setup is not
  reproducible with v0.2.0; use v0.1.0 for that.

### Fixed
Bugs and modelling problems of v0.1.0 that affect the paper's results
(numbering as in [docs/PLAN_REALISM.md](docs/PLAN_REALISM.md)):
- **F1** Occupancy grid almost empty: vehicle positions were measured against an
  assumed 500 m lane length, but the lanes are 81–128 m, so every vehicle fell into
  the last cell. The grid now uses the real lane length.
- **F2** Single-lane approaches: the left-turn lane group was never used and
  left-turners blocked straight traffic. Addressed by the stage plans (F11); multi-lane
  networks are supported throughout.
- **F3** Empty phase on 3-arm junctions: one of the 4 phases served nobody, wasting
  ≈ 25 % of the fixed-time baseline's cycle. Plans now contain only non-empty stages.
- **F4** "Delay" was the per-step change in accumulated edge waiting time (it could
  be negative). Delay is now SUMO's time loss per vehicle.
- **F5** State and messages were normalised by the episode's number of vehicles,
  leaking the demand level into the state. Fixed constants are used instead.
- **F6** Features no real junction can measure (episode progress, vehicles left in
  the simulation, CO₂ per approach, total waiting time) were removed from the state.
- **F7** Demand was ≈ 120 veh/h for the whole network; it is now 600–3600 veh/h.
- **F8** The ablation configurations (VA−X) used in the paper were not on any
  branch; they are rebuilt as configs.
- **F9** The 5-junction network is closed (no entry/exit roads). Kept for now; demand
  is expressed as network-wide veh/h between random internal origins/destinations.
  A real corridor (step 3) will fix this.
- **F10** The soft target update ran once per episode with τ = 0.005, so after
  250 episodes the target network was still ~29 % random initialisation. It is now
  applied after every gradient step.
- **F11** 4 phases with protected lefts on single-lane approaches (a waiting
  left-turner blocks the lane) replaced by netconvert's conflict-free 2-stage plans
  with permissive lefts and 3 s yellow + 2 s all-red.

## [0.1.0] - 2026-06-04

The code behind the paper (head of the `multiagent` branch): multi-agent DQN with
communication between neighbouring junctions on the 5-junction test network,
extending the single-intersection DQN controller. Results from this version are
affected by F1–F11 above.

[0.2.0]: https://github.com/E-nej/PSIS-seminarska/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/E-nej/PSIS-seminarska/releases/tag/v0.1.0
