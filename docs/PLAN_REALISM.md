# Plan: making the results realistic

Branch: `realism` (based on `origin/multiagent`, the code behind the paper).
Goal: find out whether the RL controllers still beat **realistic** baselines under
**realistic** traffic, using inputs a real intersection could actually measure.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` dropped

---

## Findings from reading the code (things that invalidate current results)

- **F1 — occupancy grid is almost empty of information.** `MultiSumoEnv._encode_state`
  computes `lane_pos = 500 - getLanePosition()`, but the lanes in
  `intersection/test.net.xml` are 81–128 m long (not 500 m as the paper says).
  So `lane_pos` is always ≥ 372 m → every vehicle falls into cell 9. The 80-bit
  grid only says "there is a car somewhere on this approach".
- **F2 — single-lane approaches.** Every edge has one lane, so the "left-turn lane group"
  is never used, and left-turners block straight traffic during protected phases.
- **F3 — empty phase on 3-arm junctions.** One of the 4 phases gives green to nobody;
  the fixed timer always serves it (≈ 25 % wasted time) → baseline is handicapped.
- **F4 — "delay" metric is the per-step change of accumulated edge waiting time.**
  It goes negative when cars leave the edge, which is why plots show "delay = -0.02 s".
  It is not a standard metric.
- **F5 — normalisation by `num_vehicles`** in state and messages leaks the demand
  level into the state and breaks when demand varies.
- **F6 — features that can't exist in reality:** `steps / max_steps`,
  `getMinExpectedNumber()` (vehicles left in the simulation), CO2 per approach,
  total waiting time.
- **F7 — demand is tiny:** 200 vehicles / 6000 s ≈ 120 veh/h for the whole network.
- **F8 — ablation code (3 seeds, VA−X configs) is not on any branch.** Decision: rebuild
  the ablation switches on top of `multiagent` (`train.py --config ...`).
- **F9 — the 5-junction network is closed.** No entry/exit roads (J5, J6 are corner nodes),
  so all trips start and end mid-link. Kept for steps 1–2; demand is expressed as
  network-wide veh/h between random internal origins/destinations. A real corridor (step 3)
  fixes this.
- **F10 — the "soft" target update ran once per episode with tau = 0.005.** After 250 episodes
  the target network still holds ~29 % of its random initialisation. New code applies it per
  gradient step (standard).
- **F11 — the paper's 4 phases with protected lefts on single-lane approaches** are unrealistic
  (a waiting left-turner blocks the whole lane). New code uses netconvert's conflict-free
  2-stage plans with permissive lefts, 3 s yellow + 2 s all-red, the same for every controller.

---

## Step 0 — Environment & harness

- [x] 0.1 Python 3.12 venv with `uv` (TensorFlow does not support 3.14), install
      `eclipse-sumo`, `traci`, `tensorflow`, `numpy`, `pandas`, `matplotlib`, `tqdm`.
- [x] 0.2 Make `SUMO_HOME` detection work with the pip `eclipse-sumo` package.
- [x] 0.3 Smoke test: run 1 short episode of current `main_multiagent.py` unchanged.
- [x] 0.4 Option to use `libsumo` instead of TraCI (much faster, no GUI).
- [x] 0.5 Measure wall-clock per episode → estimate the training budget.
      (SUMO: <1 s per simulated hour with libsumo; RL episode ≈ 10–20 s, dominated by
      TensorFlow; 150 episodes ≈ 30–60 min per run, runs in parallel.)

## Step 1 — Realistic demand + strong baselines (current network)

### 1A. Correct metrics (do this first, everything else depends on it)
- [x] 1A.1 Enable SUMO `--tripinfo-output` (+ `--duration-log.statistics`) per run.
- [x] 1A.2 `metrics.py`: parse tripinfo → per-run averages:
      travel time, **delay (timeLoss)**, waiting time, stops (`waitingCount`),
      throughput (arrived / departed), vehicles still in network at the end, CO2 (emissions output).
- [x] 1A.3 Keep per-step queue (halting vehicles) as a secondary metric.

### 1B. Demand generator
- [x] 1B.1 Poisson demand in **veh/h** via `randomTrips.py --insertion-rate --poisson`,
      routed with duarouter (`Scenario.generate_routes`). Random O/D (network is closed, F9).
- [x] 1B.2 Time profiles: `flat` and `peak` (ramp up → peak → ramp down).
- [x] 1B.3 Demand sweep (network-wide peak rate): 600 / 1200 / 1800 / 2400 / 3000 / 3600 veh/h.
      Saturation sets in between 2400 and 3000; at 3600 the network gridlocks.
      (The paper used ≈ 120 veh/h.)
- [x] 1B.4 Episode length 3600 s (a peak hour) instead of 6000 s; vehicles are
      not required to finish, the unfinished ones count in the metrics.
- [ ] 1B.5 Sanity check in GUI: queues build up at the highest level, no gridlock at the lowest.

### 1C. Fix the phase set
- [x] 1C.1 Signal plans rebuilt with netconvert (`--tls.layout opposites`), no empty phases
      (F3, F11). `signals.SignalHead` switches between green stages with proper intergreens.
- [x] 1C.2 All-red clearance (2 s) after yellow (3 s), like real controllers.

### 1D. Baselines
- [x] 1D.1 **Fixed time (Webster)** — cycle and green splits computed from the demand level,
      only valid phases.
- [x] 1D.2 **Actuated** — SUMO `type="actuated"` programs (min/max green, gap-out) generated
      by `NetworkParser`; SUMO places the detectors.
- [x] 1D.3 **Max-pressure** — TraCI controller, every 10 s pick the phase with the largest
      Σ(incoming queue − outgoing queue).
- [x] 1D.4 Keep the old "classical" fixed timer only for comparison with the paper.

### 1E. Unified controller interface + evaluation script
- [x] 1E.1 `controllers.py`: common interface `act(obs) -> phase` for fixed / actuated /
      max-pressure / RL agents, so a single loop runs everything.
- [x] 1E.2 `evaluate.py`: controller × demand level × N scenario seeds → `results/eval.csv`.
- [x] 1E.3 Fix F1 (use real lane length), F5 (fixed normalisation constants) in the RL state
      so RL is not crippled by bugs.

### 1F. RL training at realistic demand
- [x] 1F.1 Train with **randomised demand** (level sampled per episode) so one model
      covers the whole sweep.
- [x] 1F.2 Configs `base` / `full` / `full_nocomm` / `queue`, 5 training seeds each, 150 episodes.
- [x] 1F.3 Evaluate on ≥ 10 held-out scenario seeds per demand level, ε = 0.

### 1G. Report
- [x] 1G.1 Plot: delay per vehicle vs. demand level, one line per controller, 95 % CI.
- [x] 1G.2 Table: travel time, delay, stops, throughput, CO2 at each level.
- [x] 1G.3 Short write-up: does RL beat actuated / max-pressure? Does comm help near saturation?

## Step 2 — Realistic sensing & safety constraints

### 2A. Detector-based observation
- [x] 2A.1 Generate an additional file with **E2 detectors** (last 50 m of each incoming lane,
      camera/radar-like) and **E1 loops** at the stop line.
- [x] 2A.2 New state per junction: per incoming lane [halting count, occupancy, vehicles
      passed since last decision], one-hot current phase, time since phase start.
- [x] 2A.3 Remove unrealistic features (F6).
- [x] 2A.4 Sensor noise: Gaussian noise on counts + random detector dropout (p configurable).
- [x] 2A.5 Comm message built from the same detector data.
- [x] 2A.6 Reward computable from detectors (`det_queue`). Note: training happens in
      simulation, so a ground-truth reward is legitimate; the detector reward only matters
      for online fine-tuning on a real junction. Both are trained.

### 2B. Safety shield
- [x] 2B.1 Action masking: only valid phases (from 1C.1).
- [x] 2B.2 Min green 10 s, **max green** 60 s, **max red** 90 s — shield overrides the agent
      when violated; overrides are logged.
- [x] 2B.3 Count shield overrides as a metric (a good policy rarely needs them).

### 2C. Re-run
- [x] 2C.1 Retrain `full` / `full_nocomm` / `det_queue` with detector state (noise 0.05,
      dropout 0.02) + shield, 5 seeds.
- [x] 2C.2 Evaluate with the same harness; compare to step 1 ("cost of realism").

### 2D. Follow-ups found during evaluation
- [x] 2D.1 Max-pressure in step 2 read whole-lane vehicle counts (ground truth). Added
      `maxpressure_det` (E2 counts on incoming lanes, downstream E2 where it exists, with the
      experiment's detector noise/dropout): `tsc evaluate configs/step2/baselines.toml`.
- [x] 2D.2 Shielded max-pressure (60 s / 90 s) is part of the same config. The shield alone
      improves max-pressure by 3–19 %.
- [x] 2D.3 Paired comparison + Wilcoxon test: `tsc compare <csv...> --ref <controller>`.

## Step 2E — Make communication useful
Diagnostic (`tsc diagnose messages`, test-time only): replacing the real neighbour messages with
messages from *random other junctions* changes delay by < 1.2 % (n.s.), while zeroing them adds
+14–69 %. **The agents use messages only as a network-load signal, not as neighbour-specific
information.** With uniform random O/D, short links and randomised demand per episode, any
junction's message tells the same thing (how busy the network is). Standard check from now on:
communication only "works" if wrong-neighbour messages hurt.

- [~] 2E.1 Scenario where neighbours carry information a junction can't see itself.
  - [x] Arterial corridor `networks/corridor/` (built by `build.py`): 7 signals S0–S6 at
        irregular 220–380 m spacing, arterial 2+2 lanes at 50 km/h with 75 m left-turn bays,
        secondary cross streets (1+1, 40 km/h; T-junctions at S2/S4), 3 residential streets at
        unsignalised priority junctions, open boundaries. netconvert gives every signal
        3 stages: arterial through/right (+ permissive left), protected left, cross street.
  - [x] Demand generator `od`: Poisson trips between boundary points; 60 % through traffic
        on the arterial, the rest by a capacity-weighted gravity model; tidal 70/30 split,
        direction drawn per scenario (`demand.scenario_flip`).
  - [x] Baseline `greenwave`: Webster splits with a common cycle + offsets from SUMO's
        `tlsCoordinator.py`. Webster and green wave get one plan per tidal direction
        (time-of-day plans); each scenario runs the plan of its direction.
  - [x] Approach chains (`Network.upstream_edges` / `approach_edges`): a lane split makes the
        controlled edge only the last 64 m of an approach. Ground-truth rewards, the queue
        metric and exact-count max-pressure now cover the whole approach (test network
        unaffected: identical results).
  - [ ] Train `configs/corridor/{full,full_nocomm}.toml` (5 seeds) and evaluate vs baselines.
  - Baseline sweep (4 seeds per level, delay [s]): green wave 48 / 48 / 54 / 70 / 116 at
    1000–5000 veh/h, best everywhere; Webster 70 / 64 / 74 / 98 / 172; actuated
    69 / 69 / 76 / 104 / 201; detector max-pressure (shield) 48 / 59 / 71 / 173 / 271;
    exact-count max-pressure (shield) 59 / 93 / 178 / 369 / 456 — on 300 m links it counts
    moving vehicles far from the stop line and switches too often (6–11 stops per vehicle).
    Unshielded max-pressure starves the protected-left stage (bays overflow, teleports).
- [ ] 2E.2 Directional, lane-aligned messages: per incoming link, vehicles the upstream
      neighbour released toward me (its stop-line loop counts) + whether it currently serves
      me and for how long; per outgoing link, the downstream queue (spillback).
- [ ] 2E.3 Cooperative reward: r_i + α·Σ r_neighbours (spatial discount, Chu et al. 2019) or a
      pressure reward that includes downstream queues.
- [ ] 2E.4 Only if 2E.1–2E.3 show neighbour-specific use: parameter sharing + learned attention
      over neighbours (CoLight-style), CTDE / value decomposition (CVDMARL).
- Later: a small grid (e.g. 3×4) with a road hierarchy as a second scenario.

## Later (not now)
- Step 3: real Maribor corridor (OSM → netconvert, manual fixes, counts → routeSampler).
- Step 4: pedestrians at selected crossings.
- Graph-based / parameter-shared agents for larger networks.

---

## How to run

Since v0.2.0 everything runs through the `tsc` package (see the README):

```bash
uv run tsc evaluate configs/baselines.toml                       # baselines only
uv run tsc train configs/step1/full.toml --seeds 0-4             # step 1 training
uv run tsc train configs/step2/det_queue.toml --seeds 0-4        # step 2 training
uv run tsc evaluate configs/step1/evaluate.toml                  # trained runs
uv run tsc plot outputs/eval/baselines/results.csv outputs/eval/step1/results.csv --out outputs/plots/step1
```

Runs trained with the pre-package scripts (`Deep-Qlearning-TSC/runs/train*/…`) can be
evaluated with `--runs "<dir>/*"`; they are loaded from their `config.json`.

```bash
uv run tsc evaluate configs/step2/baselines.toml                 # shielded + detector max-pressure
uv run tsc compare outputs/eval/step2_baselines/results.csv outputs/eval/step2/results.csv \
       --ref maxpressure_det_shield                              # paired, Wilcoxon
uv run tsc diagnose messages configs/step2/evaluate.toml         # do agents use the messages?
```

## Log
- 2026-10-11: plan written, branch `realism` created from `origin/multiagent`.
- 2026-10-11: steps 0, 1A–1E, 2A–2B implemented. Baseline sweep done (10 seeds per level).
  Max-pressure is the strongest baseline near/over saturation; actuated ≈ max-pressure at low
  demand; Webster is worse than a 40 s fixed plan once oversaturated (spillback on ~100 m links).
  Step 1 and step 2 RL training launched (5 seeds × 4 + 5 seeds × 3 configs).
- 2026-10-11: refactor into the `tsc` package (v0.2.0, branch `refactor-v0.2`,
  see [REFACTOR.md](REFACTOR.md)). Baseline sweep, observations/rewards, trained-policy
  evaluation and short training runs reproduce the pre-package scripts exactly.
- 2026-10-11: all 35 runs trained and evaluated (6 levels × 10 held-out seeds).
  Step 1: every RL config is far better than the fixed timer and roughly equal to max-pressure;
  the `queue` reward is the best (−6 to −18 % delay vs max-pressure at 600–1200 veh/h,
  significant; within noise near saturation). Communication brings no measurable benefit
  (`full_nocomm` ≥ `full` everywhere). All 5 seeds of every config converge to similar
  policies (no failed seeds, unlike the paper's ablation; plausibly the F10 fix).
- 2026-10-11: 2D done. On the same noisy detectors + shield, step-2 RL ≈ max-pressure:
  `full_nocomm` is 2–3 % better at 600–1800 veh/h (significant only at 1200), tied near
  saturation; `full` (with communication) is tied or slightly worse; `det_queue` is 9 % worse
  at 3600. Limiting max-pressure to 50 m detectors costs it up to 14 % (at 2400) vs ground truth.
  Oddity: noisy detector max-pressure beats noise-free detector max-pressure at 3000–3600
  (noise breaks its tendency to hold a stage) — worth a look before publishing.
- 2026-10-11: message diagnostic → communication is used only as a load signal (2E).
  Graphite stack on top of `refactor-v0.2`: `2d-detector-maxpressure`, then the corridor (2E.1).
