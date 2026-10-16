# Adaptive traffic-signal control with multi-agent deep RL

Deep Q-learning (DQN) agents control the traffic lights of a road network in
[SUMO](https://eclipse.dev/sumo/), one agent per signalised junction, optionally
exchanging short messages with their neighbours. They are compared with the
baselines a traffic engineer would use: fixed-time plans (including Webster
plans), SUMO's actuated control, and max-pressure.

This started as a university project (Univerza v Mariboru, FERI). We extended a
single-intersection DQN controller to a multi-agent system with communication
and published a paper on it (in Slovene). The paper's results came from a small,
unrealistic setup. The current work checks whether RL still beats strong
baselines under realistic demand and with inputs a real junction can actually
measure. The plan, findings and progress are in
[docs/PLAN_REALISM.md](docs/PLAN_REALISM.md).

| Version | What it is |
|---|---|
| [v0.1.0](../../tree/v0.1.0) | The code behind the paper: 4-phase plans, grid state, ~120 veh/h. Kept for reference. |
| v0.2.0 | This package: realistic demand, rebuilt signal plans with intergreens, strong baselines, standard metrics, detector-based observations, safety shield. Fixes bugs F1–F11 in v0.1.0 (see [CHANGELOG.md](CHANGELOG.md)). |
| Unreleased | Arterial corridor and the real Maribor (Titova cesta) district, plan-based RL, a hierarchy (a lookup table picks the best controller per demand level, RL fine-tunes it), RL controllers the table can pick, and the RESCO / CoSLight benchmark scenarios. |

## Results

Latest round: step 2I, [docs/V6_REPORT.md](docs/V6_REPORT.md). Mean delay per vehicle [s],
10 held-out scenarios per demand level, mean of 3 training seeds, **noise-free detectors**
(with sensor noise and detector failures the numbers are worse: see the earlier reports).

**Our networks.** The hierarchy (v6): a lookup table picks, per measured demand level, one of
six controllers: green wave, coordinated-actuated, max-pressure, actuated, and two RL
controllers. Each classic controller can run with or without an RL worker that fine-tunes it.

| network | best classic controller | previous best (2H) | **v6 hierarchy** |
|---|---|---|---|
| arterial corridor, 7 signals | green wave 73.8 | 67.7 (−8 %) | **48.4 (−34 %)** |
| corridor, stale (opposite-direction) plans | max-pressure 92.8 | 96.1 | **48.4 (−48 %)** |
| Maribor, Titova cesta, 21 signals | SUMO actuated 85.9 | 82.0 (−5 %) | **72.7 (−15 %)** |

Significantly better (paired Wilcoxon) at every demand level, with the same or higher
throughput and fewer stops. The gain comes from the RL controllers inside the table, above all
one phase-wise network shared by all junctions (FRAP / MPLight style).

**Published benchmarks** (the RESCO and CoSLight scenarios, delay [s]; ours held-out, the
published numbers as reported):

| map | **ours** | best published | best classic (our runs) |
|---|---|---|---|
| Cologne 1 | **16.9** | IDQN 26.05 | 20.4 |
| Cologne 3 | 18.9 | IPPO 22.13 | **17.4** |
| Cologne 8 | **17.2** | IPPO 21.49 (CoSLight 24.79) | 20.7 |
| Ingolstadt 1 | **12.2** | IPPO 19.85 | 14.1 |
| Ingolstadt 7 | **22.4** | IPPO 30.70 | 28.4 |
| Ingolstadt 21 | 62.3 (59.9 on RESCO's best-episode protocol) | **IDQN 59.64** | 65.1 |
| Grid 4×4 | **26.0** | CoSLight 46.11 | 46.9 |
| Avenue 4×4 | 793 (trip time **330**) | **CoSLight 745** (trip time 364) | 766 |

Lower delay than every published method on 6 of 8 maps, a tie on Ingolstadt 21. The setups
are close, not identical: RESCO's agents see every vehicle within 200 m and average 5 seeds;
ours use 50 m + 150 m area detectors and 3 seeds. Details and caveats (teleports, vehicles still
waiting to enter) in the report.

Earlier steps: [PLAN_REALISM.md](docs/PLAN_REALISM.md) (steps 1–3), [HIERARCHY_REPORT.md](docs/HIERARCHY_REPORT.md)
(2H), [BENCHMARKS.md](docs/BENCHMARKS.md) (benchmarks with noisy detectors and the handoff
improvements), [BENCHMARK_HANDOFF.md](docs/BENCHMARK_HANDOFF.md) (papers and improvement list).

## How it works

- **Network** (`tsc.network`): signal plans are rebuilt with `netconvert` into
  conflict-free *green stages* with 3 s yellow + 2 s all-red. Every controller
  only chooses the next green stage, so all of them are compared on the same plan.
  Nothing assumes a particular network: junctions may have any number of lanes and
  stages, and neighbours are found by following the roads.
- **Signals** (`tsc.signals.SignalHead`): what a real controller enforces for
  Python-controlled junctions: minimum green (10 s), yellow/all-red intergreens
  and an optional shield (maximum green, maximum red). A decision is made every
  5 s once min green has passed.
- **Demand** (`tsc.demand`): Poisson trips at a given network-wide rate (veh/h),
  with a `peak` profile (ramp up, peak, ramp down) over a one-hour episode.
- **Controllers** (`tsc.controllers`): `fixed` (netconvert's plan), `webster`
  (plan optimised for calibration demand), `actuated` (SUMO, gap-based),
  `maxpressure`, and trained RL policies.
- **Observation** (`tsc.observation`): `grid` (vehicle positions per incoming
  lane, perfect information) or `detector` (only E2 area detectors and stop-line
  loops, with optional noise and detector failures). Communication appends a
  4-value message from each neighbouring junction.
- **Reward** (`tsc.reward`): `wait` (change in waiting time, the paper's base),
  `multi` (the paper's multi-objective reward), `queue`, or `det_queue` (queue seen
  by the detectors, computable on a real junction).
  With `approach_length` a second, longer area detector adds the vehicles still moving
  toward the stop line; `layout = "phase"` describes every green stage instead of every
  lane (for the phase-wise network).
- **Agent** (`tsc.agents.dqn`): DQN with experience replay, optional Double DQN, a
  soft (Polyak, every gradient step) or hard target update, Huber loss and
  2 × 400 hidden units, as in the paper. Options: time-based (SMDP) discounting,
  dueling head, n-step returns, a residual head that starts as a reference action
  (`ref_action`), and a phase-wise network shared by all junctions (`phase_net`).
- **Control modes** (`[signals] control`): `stages` (RL picks the next stage), `plan`
  (RL adjusts a coordinated green-wave plan), `hierarchy` (a manager, e.g. a lookup
  table, picks an expert controller per demand level; optional RL workers fine-tune it).
- **Metrics** (`tsc.metrics`): from SUMO's tripinfo output: delay per vehicle
  (time loss + insertion delay, unfinished vehicles included), waiting time,
  stops, travel time, throughput, CO₂, mean queue.

## Install

You need [uv](https://docs.astral.sh/uv/) and Python 3.12. The RL agents use PyTorch;
`uv sync` installs its CPU build (the bit-exact reference). GPU builds (CUDA, ROCm) are
opt-in dependency groups, see [docs/GPU.md](docs/GPU.md).

```bash
uv sync
```

SUMO comes from the `eclipse-sumo` pip package, pinned to 1.28.0 because generated
networks and routes depend on the SUMO version. To use another SUMO installation,
set `SUMO_HOME`. The SUMO GUI (`sumo-gui`) is included in the package.

Check the installation:

```bash
uv run pytest
```

`uv run pytest -m "not sumo"` runs only the unit tests (about 10 s). The full
suite also runs short SUMO simulations of every controller type (about 1 min).

## Quickstart

Every experiment is one TOML file in `configs/`. A config holds only the values
that differ from the defaults in [`src/tsc/config.py`](src/tsc/config.py). Results go to
`outputs/` (gitignored), and every run saves its resolved `config.toml` and
`run_info.json` (versions, git commit) next to the results.

```bash
# baselines over the demand sweep (600–3600 veh/h, 10 scenarios each; ~5 min on 6 workers)
uv run tsc evaluate configs/baselines.toml

# train one RL configuration, 5 seeds in parallel (30–60 min per seed)
uv run tsc train configs/step1/full.toml --seeds 0-4

# evaluate the trained runs and plot everything with 95 % confidence intervals
uv run tsc evaluate configs/step1/evaluate.toml
uv run tsc plot outputs/eval/baselines/results.csv outputs/eval/step1/results.csv --out outputs/plots/step1
```

Any value can be overridden on the command line, e.g.
`--set observation.noise=0.1 --set name=full_noise`. `--workers N` sets the
number of parallel processes. PyTorch runs single-threaded in each process
(`TSC_THREADS`, default 1), so parallel runs do not slow each other down.

### Seeds

Every scenario (routes and SUMO's random numbers) has an explicit seed, and the
ranges never overlap ([`src/tsc/seeds.py`](src/tsc/seeds.py)):

| Use | Seeds |
|---|---|
| training, seed *s*, episode *e* | 100000·*s* + *e* (*s* = 0…6) |
| validation (model selection) | 700001 + *i* |
| acceptance (`tsc accept`: learned policy vs its fallback) | 750000 + *i* |
| calibration (Webster plans) | 800000 + *i* |
| evaluation | 900000 + *i* |

With the same seed, training reproduces the same run.

## Reproducing v0.2.0 results

Steps 1 and 2 of [the plan](docs/PLAN_REALISM.md):

```bash
uv run tsc evaluate configs/baselines.toml

# step 1: perfect-information grid state, ablations of the paper's extensions
for c in base full full_nocomm queue; do uv run tsc train configs/step1/$c.toml --seeds 0-4; done
uv run tsc evaluate configs/step1/evaluate.toml

# step 2: detector observations with noise / dropout, max-green 60 s / max-red 90 s shield
for c in full full_nocomm det_queue; do uv run tsc train configs/step2/$c.toml --seeds 0-4; done
uv run tsc evaluate configs/step2/evaluate.toml

uv run tsc plot outputs/eval/baselines/results.csv outputs/eval/step1/results.csv \
    outputs/eval/step2/results.csv --out outputs/plots/v0.2.0
```

The refactor was checked against the pre-package scripts: the baseline sweep
reproduces every metric of all 240 runs exactly, observations and rewards are
identical step by step, trained policies evaluate identically, and training with
the same seed gives an identical training log. That held up to version 0.2, which used
TensorFlow. Since 0.3 the agents use PyTorch: runs trained before (`*.weights.h5`, also
those of the pre-package scripts) cannot be loaded any more and must be retrained.

## Repository layout

```
configs/      experiment configs (TOML)
networks/     SUMO networks: test5 (5 junctions, the paper's network), single_4l (one junction,
              4-lane arms), corridor (7-signal arterial), maribor (Titova cesta, 21 signals),
              benchmarks (RESCO / CoSLight scenarios, downloaded by fetch.py)
src/tsc/      the package: network, demand, detectors, signals, simulation, controllers,
              observation, reward, agents, metrics, experiments (train / evaluate / plot), cli
tests/        pytest suite
docs/         plan and progress (PLAN_REALISM.md), reports (HIERARCHY_REPORT.md, BENCHMARKS.md,
              V6_REPORT.md), refactor checklist (REFACTOR.md)
configs/v6/   the latest round: config and job generators, the job queue (run_queue.py)
outputs/      generated networks, routes, runs, results (gitignored)
```

## Background: deep Q-learning for signal control

Each agent learns Q(s, a), the expected discounted future reward of choosing green
stage *a* in state *s*. A neural network approximates Q and is trained with the
TD target r + γ·max_a′ Q_target(s′, a′) on mini-batches drawn from a replay buffer
of past transitions. This breaks the correlation between consecutive samples and
reuses experience. A slowly updated target network stabilises the target. Double
DQN chooses a′ with the online network and evaluates it with the target network,
which reduces over-estimation. Exploration is ε-greedy, with ε decaying linearly
from 1 to 0.05 over the first 60 % of episodes. Agents only choose among the valid
stages of their junction (action masking).

## Credits

The project started as a fork of Ritu Pande's *Deep-Qlearning-TSC* (single
intersection, DQN, SUMO), which also provided the `single_4l` network. The
approach built on:

- J. Gao, Y. Shen, J. Liu, M. Ito, N. Shiratori: *Adaptive Traffic Signal Control: Deep Reinforcement Learning Algorithm with Experience Replay and Target Network*.
- A. Vidali, L. Crociani, G. Vizzari, S. Bandini: *A Deep Reinforcement Learning Approach to Adaptive Traffic Lights Management*.
- *Adaptive traffic signal control using deep reinforcement learning: A multi-objective approach for single and multi-intersection scenarios* ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0386111225000445)).
- P. Varaiya: *Max pressure control of a network of signalized intersections* (2013).
- R. S. Sutton, A. G. Barto: *Reinforcement Learning: An Introduction*.
