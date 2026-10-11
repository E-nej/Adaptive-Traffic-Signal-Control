# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `maxpressure_det` baseline: max-pressure on the E2 detectors only, with the experiment's
  detector noise / dropout (`configs/step2/baselines.toml`, with the safety shield).
- `tsc compare`: paired comparison against a reference controller (relative difference per
  scenario, 95 % CI, Wilcoxon signed-rank test).
- `tsc diagnose messages`: evaluates communicating policies with real, zeroed and
  wrong-neighbour messages (`Observer.set_message_mode`).

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
