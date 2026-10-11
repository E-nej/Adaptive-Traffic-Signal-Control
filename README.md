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
- **Agent** (`tsc.agents.dqn`): DQN with experience replay, optional Double DQN, a
  soft (Polyak, every gradient step) or hard target update, Huber loss and
  2 × 400 hidden units, as in the paper.
- **Metrics** (`tsc.metrics`): from SUMO's tripinfo output: delay per vehicle
  (time loss + insertion delay, unfinished vehicles included), waiting time,
  stops, travel time, throughput, CO₂, mean queue.

## Install

You need [uv](https://docs.astral.sh/uv/). Python 3.12 is used because TensorFlow
does not support newer versions yet.

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
number of parallel processes. TensorFlow runs single-threaded in each process
(`TSC_TF_THREADS`, default 1), so parallel runs do not slow each other down.

### Seeds

Every scenario (routes and SUMO's random numbers) has an explicit seed, and the
ranges never overlap ([`src/tsc/seeds.py`](src/tsc/seeds.py)):

| Use | Seeds |
|---|---|
| training, seed *s*, episode *e* | 100000·*s* + *e* (*s* = 0…6) |
| validation (model selection) | 700001 + *i* |
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
the same seed gives an identical training log. Run directories written by those
scripts (`config.json` + `*.weights.h5`) can be evaluated directly, e.g.
`tsc evaluate configs/step1/evaluate.toml --runs "path/to/runs/train/*"`.

## Repository layout

```
configs/      experiment configs (TOML)
networks/     SUMO networks: test5 (5 junctions, the paper's network), single_4l (one junction, 4-lane arms)
src/tsc/      the package: network, demand, detectors, signals, simulation, controllers,
              observation, reward, agents, metrics, experiments (train / evaluate / plot), cli
tests/        pytest suite
docs/         plan and progress (PLAN_REALISM.md), refactor checklist (REFACTOR.md)
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
