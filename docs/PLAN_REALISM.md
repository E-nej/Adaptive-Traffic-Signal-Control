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
- [x] 2E.2 Directional messages (`[observation] message = "directional"`, 5 values per
      neighbour n → receiver i, all from n's detectors): queue on n's lanes whose movements
      lead to i, vehicles n released toward i (stop-line loops, ~10 s memory), share of n's
      links toward i that are green, how long n has served i, and the queue on n's approach
      coming from i (spillback). `Network.toward` / `arrival` map the movements.
      Config: `configs/corridor/dir.toml`.
- [x] 2E.3 Cooperative reward `[reward] neighbour_weight = α`: r_i + α·Σ r_neighbours
      (spatial discount, Chu et al. 2019). α = 0 reproduces the previous training exactly.
- [x] 2E.5 Corridor experiment, 5 seeds each (150 episodes), 10 held-out scenarios per level,
      checked with `tsc diagnose messages`:

      | config               | messages    | α   | delay [s] 1000 / 2000 / 3000 / 4000 / 5000 veh/h |
      |----------------------|-------------|-----|---------------------------------------------------|
      | green wave (baseline)| –           | –   | 45 / 48 / 56 / 72 / 127                            |
      | corridor_full_nocomm | none        | 0   | 62 / 69 / 97 / 235 / 352                           |
      | corridor_full        | aggregate   | 0   | 63 / 67 / 90 / 210 / 326                           |
      | corridor_dir         | directional | 0   | 58 / 67 / 104 / 262 / 361                          |
      | corridor_coop_nocomm | none        | 0.5 | 68 / 72 / 117 / 272 / 374                          |
      | corridor_dir_coop    | directional | 0.5 | 66 / 75 / 136 / 292 / 384                          |

      **Every free-stage RL variant loses badly to the green wave** (×3 at 4000 veh/h; also
      worse than Webster / actuated from 3000 up): the agents switch per junction and break the
      platoons (~7 stops per vehicle vs 1.5). Communication does help on the corridor, unlike on
      the grid: `full` is 6–10 % better than `full_nocomm` at 3000–5000 (significant), and
      directional messages are the first to carry neighbour-specific information (wrong-neighbour
      messages cost `corridor_dir` +6–14 %). The cooperative reward hurts (+5 to +40 %).
- [ ] 2E.4 Only if 2E.1–2E.3 show neighbour-specific use: parameter sharing + learned attention
      over neighbours (CoLight-style), CTDE / value decomposition (CVDMARL).
- Later: a small grid (e.g. 3×4) with a road hierarchy as a second scenario.

## Step 2F — RL on top of a coordinated plan
Free stage choice cannot learn the coordination a green wave has by construction. Instead the
policy *adjusts* the green-wave plan (`[signals] control = "plan"`, `controllers/plan.py`):
every 5 s of green it picks FOLLOW (switch when the plan does), HOLD (extend the green, at most
20 s past the plan) or EARLY (end it now; the next stage keeps it until the plan's own end).
The plan's absolute timing stays the reference, so junctions stay coordinated; a junction two
stages behind rejoins the plan.

- [x] 2F.1 `plans.py` (plan file → stage windows; SUMO starts a cycle at t = offset),
      `PlanActions`, baselines `greenwave_py` (Python replay of the plan, the exact reference)
      and `plan_actuated` (classic coordinated-actuated: gap extension + early end).
- [x] 2F.2 Plan library for training (green wave per level × direction from the calibration
      scenarios); an episode runs its matching plan with probability 0.5, otherwise a random
      one, so the policy learns to repair plans that do not fit. Validation: 3 levels × matching
      and opposite-direction plan (6 runs).
- [x] 2F.3 Evaluation regimes (`[evaluate] plan_level`, `plan_direction`): *tuned* (each level
      and direction its own plan), *stale* (the 3000 veh/h plan everywhere) and *opposite* (the
      other direction's plan: the AM plan in the PM peak). Stale costs the green wave almost
      nothing on the corridor; opposite costs +60–90 % at 4000–5000 veh/h.
- [x] 2F.4 Sweep, 3 seeds × 150 episodes. Learning no longer plateaus: validation improves by
      16–38 s per 100 episodes vs 2.5–9 s noise between checkpoints (the old free-stage runs:
      ±1–3 s per 100 episodes, i.e. flat after ~20 episodes).

      | delay [s], opposite plans    | 1000 | 2000 | 3000 | 4000 | 5000 |
      |------------------------------|------|------|------|------|------|
      | green wave (SUMO)            | 60   | 67   | 77   | 117  | 243  |
      | greenwave_py (reference)     | 66   | 70   | 80   | 124  | 251  |
      | plan_actuated (classic)      | 66   | 96   | 98   | 109  | 187  |
      | plan_nocomm                  | 57   | 65   | 73   | 94   | 180  |
      | plan_dir (directional msgs)  | 55   | 62   | 71   | 93   | 177  |
      | plan_dir_shared (+ shared net) | 56 | 60   | 67   | 89   | 183  |

      **With a plan that does not fit the traffic, RL on the plan beats the plan by 10–29 %
      (significant at 1000–5000) and classic coordinated-actuated control by 13–38 % at
      1000–4000 (significant; tied at 5000).** With
      tuned plans it only matches the green wave (best variant within ±2 % up to 4000 veh/h).
      Messages help vs no messages: `plan_dir_shared` −5 to −11 % at 2000–4000 (significant, both
      regimes), `plan_dir` −2 to −5 %. The "tuned" learning setup (lr 5e-4, τ 0.01, reward ×0.1,
      faster ε decay) is worse everywhere; the free-stage RL with shared network + tuned setup is
      still ×2.5 the green wave at 4000.
- [x] 2F.5 Clean communication test: `plan_dir_shared_long` vs `plan_nocomm_shared_long` (only
      the messages differ), 5 seeds × 300 episodes, paired over 10 scenarios × 5 seeds:

      | messages vs none, delay      | 1000    | 2000    | 3000    | 4000    | 5000   |
      |------------------------------|---------|---------|---------|---------|--------|
      | opposite plans, best weights | +4.9 %* | −1.9 %  | −2.1 %  | +0.4 %  | −1.4 % |
      | opposite plans, final weights| +25.3 %*| −5.4 %* | −6.8 %* | −3.7 %* | −3.0 % |
      | tuned plans, best weights    | +6.5 %* | −0.6 %  | −0.2 %  | −0.7 %  | −0.5 % |
      | tuned plans, final weights   | +19.7 %*| −5.3 %* | −6.5 %* | −4.9 %* | −5.1 %*|

      **Directional messages help by 4–7 % at 2000–5000 veh/h with the final weights (both
      regimes, significant), but not at 1000 veh/h**; picking the best validation checkpoint
      hides the difference (it partly selects the luckiest validation). 300 episodes are no
      better than 150 (`plan_dir_shared`). At 1000 veh/h the best checkpoints reach 39–45 s vs
      52–66 s for the plan; the final weights are less stable there.
      Message diagnostic (`plan_dir_shared_long`, final weights, opposite plans): zeroed
      messages +32 / +36 %, wrong-neighbour messages **+6.7 / +1.9 %** at 2000 / 4000 veh/h (both
      significant) — the messages are used, and partly as neighbour-specific information.

## Step 2G — Choosing the controller per junction
Each road may have a different best controller, so each junction picks one of several experts
for the next ~cycle (`[signals] control = "select"`, `controllers/select.py`): green wave,
coordinated-actuated, detector max-pressure, isolated gap-actuated (`[signals] experts`).
Switching is safe (SignalHead intergreens; a plan expert rejoins the plan).

- [x] 2G.1 Non-learning baseline `tsc assign`: greedy search for the best fixed expert per
      junction and demand level on the calibration scenarios (`select_assigned`).
- [x] 2G.2 RL selector (DQN, shared network, decision every 90 s, γ = 0.8), with and without
      neighbours' choices as messages, local and cooperative (α = 1) reward.
- Results:
  - Corridor: the best fixed assignment is the green wave everywhere (≈ `greenwave_py`). The RL
    selector is worse than the green wave (117–128 vs 74 s at 4000 veh/h, tuned plans): it picks
    max-pressure ~50 % of the time — that shortens the junction's own queue while breaking the
    neighbours' green wave. The cooperative reward does not fix it (126–155 s).
  - Maribor (see step 3): the best fixed assignment uses all four experts at every level and
    beats the green wave by 2–25 %, but a single isolated controller is better still
    (max-pressure at 2000–4000, SUMO actuated at 6000–8000). The RL selector does not beat the
    fixed assignment. **Negative result: controller selection is easy to state but the reward
    for a 90 s choice is too delayed and noisy for DQN here.**
- The RL selector (`control = "select"`) was removed after 2H (the hierarchy replaces it);
  it is in git history up to commit `18e4cf4`. The experts (`ExpertPool`) and `tsc assign` stay.

## Step 3 — Maribor, Titova cesta district
- [x] 3.1 `networks/maribor/`: Overpass extract (ODbL) around Titova cesta, `build.py`
      (main roads, joined junction clusters, roads leaving the extract removed): 2.8 × 3.3 km,
      21 signals (Titova cesta, Pobreška cesta, Cesta proletarskih brigad, Partizanska, …),
      24 boundary points. OSM has no signal timings: netconvert plans as for the synthetic networks.
- [x] 3.2 Generalisations for real networks (synthetic networks unchanged, verified):
      neighbours limited to 1000 m of road; one-way boundary points (cut dual carriageways);
      area detectors continue upstream on short OSM approach lanes; fixed-time plans whose cycle
      is too short for a junction's minimum greens are recomputed.
- [x] 3.3 Demand: O/D gravity model, 20 % through traffic on the eastern trunk, tidal 70/30
      north–south. Probe: no congestion up to 6000 veh/h, saturation ≈ 8000, gridlock from 10000.
      Evaluation levels 2000–8000.
- [x] 3.4 Baselines (10 scenarios per level), delay [s]:

      | controller        | 2000 | 4000 | 6000 | 7000 | 8000 |
      |-------------------|------|------|------|------|------|
      | fixed             | 69   | 134  | 214  | 275  | 345  |
      | Webster           | 53   | 68   | 88   | 124  | 180  |
      | green wave        | 54   | 71   | 89   | 113  | 170  |
      | actuated (SUMO)   | 50   | 57   | **70** | **93** | **159** |
      | max-pressure (detectors) | **43** | **51** | 87 | 127 | 207 |
      | best fixed assignment | 45 | 54 | 86 | 110 | 169 |

      Unlike the corridor, the green wave is not the reference here: signals are sparse and far
      apart, so isolated adaptive control wins. Opposite-direction plans cost the green wave
      +33–37 % at 6000–7000 veh/h.
- [x] 3.5 RL on the plan (`configs/plan/maribor_dir_shared.toml`, 3 seeds × 150 episodes):
      **fails on Maribor** — +58 to +138 % delay vs the plan replay at 4000–8000 veh/h (tuned
      plans), +35 to +78 % with opposite plans, teleports at high demand. Validation gets worse
      during training (144 → ~230 s). One cause found: with short stage windows a 20 s extension
      can put a junction two windows behind, and the catch-up then skips a stage (37 skips per
      hour in one run), starving an approach for a full 122 s cycle.
- [x] 3.6 Retry with `[signals] hold_guard = true` (extend only if the next stage keeps its
      minimum green inside its window; catch up without skipping), with and without the
      cooperative reward (α = 0.5), 3 seeds × 150 episodes. The guard alone does not help
      (validation 200–230 s). Guard + cooperative reward, delay [s] (best weights):

      | plans     | controller                | 2000 | 4000 | 6000 | 7000 | 8000 |
      |-----------|---------------------------|------|------|------|------|------|
      | tuned     | greenwave_py              | 55   | 72   | 91   | 116  | 171  |
      | tuned     | RL guard + coop           | 58   | 76   | 105  | 142  | 204  |
      | opposite  | greenwave_py              | 57   | 79   | 119  | 155  | 222  |
      | opposite  | plan_actuated             | 54   | 77   | 106  | 145  | 213  |
      | opposite  | RL guard + coop           | 59   | 77   | 103  | 138  | 203  |
      | –         | actuated (SUMO, isolated) | 50   | 57   | 70   | 93   | 159  |

      With opposite plans the RL is 3–9 % better than the plan at 4000–8000 but **not
      significantly** (scenario-to-scenario spread ±14–20 %); with tuned plans it is
      significantly worse (+5 to +21 %). **Plan-based RL does not carry over to Maribor**: in a
      network of sparse, far-apart signals coordination is worth little, and isolated actuated
      control beats every plan-based controller by 20–30 % at 6000–8000 veh/h.

## Step 2H — Hierarchy: a manager chooses the experts, junctions fine-tune them
Goal: one system that is at least as good as the best classic controller on every network
(no single classic controller is: the green wave wins the corridor, actuated / max-pressure
win Maribor), with RL adding gains on top. Two levels (`[signals] control = "hierarchy"`,
`controllers/hierarchy.py`):

- **Manager** (top): every `[hierarchy] interval` s (default 300) picks one expert per group:
  `scope = "network"` (one expert everywhere), `"group"` (signals linked by roads of at most
  `group_range` m, `Network.groups`; Maribor at 400 m: 10 groups, the corridor: 1) or
  `"junction"`. One small DQN shared by all groups. Its input does not depend on the network
  (detector aggregates of the group and of the whole network, the group's expert, its share
  of the signals), so a manager can be moved between networks. Reward: −mean queue per lane
  in the group. `manager = "fixed"` runs one expert through the same machinery (baseline).
- **Workers** (bottom): `worker = "off"` runs the chosen expert unmodified; `"residual"`: a
  shared DQN per junction picks AGREE / HOLD / SWITCH every 5 s of green (on a plan expert
  the plan's HOLD / EARLY, with the hold guard); between decisions an agreeing worker leaves
  the expert in control second by second. Input: detectors + directional messages + active
  expert, whether it wants to switch, extension used, plan features. Both levels learn in
  the same run. A worker that always agrees reproduces each expert exactly (tested).

- [x] 2H.0 The selector's `actuated` expert was a weaker copy of SUMO's actuated control
      (stop-line loops: a vehicle standing on the loop holds the green), so every 2G result
      was capped below the baseline. `[signals] actuated_loops`, Maribor, delay [s]
      (`configs/maribor/actuated_py*.toml`, 10 scenarios per level):

      | actuated controller                         | 2000 | 4000 | 6000 | 7000 | 8000 |
      |---------------------------------------------|------|------|------|------|------|
      | SUMO                                        | 50   | 57   | 70   | 93   | 159  |
      | Python, stop-line loops (2G, `stopline`)    | 54   | 64   | 86   | 121  | 176  |
      | Python, loops 2 s upstream (`upstream`)     | 53   | 61   | 81   | 114  | 174  |
      | + lanes green in the next stage ignored (`sumo`) | 52 | 60 | 74 | 105 | 163  |

      `sumo` is within 4–6 % of SUMO up to 6000 veh/h, 2–12 % above. Using only priority (`G`)
      movements' detectors changes nothing. The rest of the gap is unexplained; the exact fix
      would be handing a junction to SUMO's own actuated program while that expert is active.
- [x] 2H.1 Manager with `worker = "off"`: `configs/hierarchy/{maribor_network,maribor_group,
      corridor_network}.toml`, tests in `tests/test_hierarchy.py`.
- [x] 2H.2 Residual workers (`worker = "residual"`).
- [x] 2H.3 Train manager + workers together, 3 seeds × 150 episodes:
      `configs/hierarchy/{maribor_network,maribor_group,corridor_network}.toml`. **Fails**: up to
      +104 % delay vs the green wave on the corridor, +23–39 % vs SUMO actuated at 6000–8000 on
      Maribor. Causes (ablations): the queue reward makes the manager prefer max-pressure (short
      queues, twice the delay of the green wave); workers trained under the wrong experts hurt
      (+5 to +42 %); and two worker settings (`max_red = 150`, `hold_guard`) weakened the experts.
      Fixes:
      - v2: DQN manager with the `delay` reward, a default expert, a fallback checkpoint and
        `tsc accept`;
      - v3c: a lookup-table manager with the experts set up exactly as their baselines.

      v3c beats every single expert on the corridor (69.4 vs 73.8 s, matched plans; 99.6 vs
      102.8 s, stale plans) and equals max-pressure on Maribor (83.3 s). Full report:
      [HIERARCHY_REPORT.md](HIERARCHY_REPORT.md).
- [x] 2H.5 Expert-specific workers: one worker per expert, trained with that expert always active
      (delay reward), kept only if it beats its expert alone; table rebuilt over expert + worker.
      Only the corridor green-wave workers passed (2 of 3 seeds); workers on adaptive experts did
      not. v4 (table over expert + worker): −6.6 % vs the best expert with stale plans. v5 (the
      table decides per level whether the worker acts, `table_alone`): 67.7 s on the corridor,
      −8.2 % vs the green wave, below perfect expert selection (68.0 s). Maribor v4: 82.0 s
      (−1.6 % vs max-pressure).
- [ ] 2H.6 Table: "does the plan fit" input (stale plans); SUMO's own actuated program as the
      actuated expert.
- [ ] 2H.4 Imitation warm start; leave-one-network-out test (train on two networks, test on
      the third).

## Step 2I — Best possible results: the benchmark improvements in the hierarchy (v6)
Goal: results as good as they can be (sensor noise and detector dropout off for now), and a
direct comparison with published numbers. Branch `2i-overnight`, configs in `configs/v6/`
(generated by `make_configs.py`; jobs by `make_jobs.py`, run by `run_queue.py`).

Why: the 2H.5 workers were trained before the benchmark work found what makes RL beat the
classic controllers (approach detector + SMDP discounting: rl2 is the best controller on 5 of
8 RESCO maps). The workers had neither, started from ε = 1 (random overrides wreck the expert's
traffic, validation got worse early) and were selected on 3 noisy validation runs. The
improvements below come from the literature in BENCHMARK_HANDOFF.md plus residual RL:

| change | source | where |
|---|---|---|
| noise-free detectors (no noise, no dropout) | user decision | all v6 configs |
| workers with SMDP (γ = 0.995 / s) and the 150 m approach detector | handoff 2–3 (Zhang et al. 2022; Efficiency & Equity 2020), proven on RESCO | `configs/v6/workers/` |
| workers start as their expert: Q(s, AGREE) = V(s), Q(s, a) = V(s) + A(s, a) with A = 0 at start, ε from 0.2 | residual policy learning (Silver et al. 2018; Johannink et al. 2019) | `agent.ref_action`, `train.eps_start` |
| validation on 2 scenarios per level | less noisy model selection (2H: validation picked lucky checkpoints) | `train.val_per_level` |
| a worker for max-pressure on the corridor (the table picks max-pressure ≤ 1500 veh/h) | 2H.5 table | `configs/v6/workers/corridor_maxpressure.toml` |
| RL experts the table can pick: `rl` (per-junction DQN as rl2) and `rlf` (one phase-wise network for all junctions) | rl2 (BENCHMARKS) and FRAP / MPLight (Zheng 2019, Chen 2020; handoff item 4, the one not done) | `controllers/select.py` RLExpert, `agent.phase_net`, `observation.layout = "phase"` |
| benchmarks: rl2 without noise (rl2nf) and the phase-wise network (rl2f) on all 8 RESCO maps | direct comparison with RESCO Table 1 (IDQN, IPPO, MPLight, FMA2C) and CoSLight | `configs/v6/bench/` |

- [x] 2I.1 Train: 15 workers (corridor: green wave, coordinated-actuated, max-pressure; Maribor:
      max-pressure, actuated; 3 seeds, 150 episodes), RL experts rl / rlf (2 networks × 3 seeds),
      rl2nf / rl2f (8 maps × 3 seeds).
- [x] 2I.2 Tables (per seed, over 6 experts incl. rl / rlf, each classic one with or without its
      worker per level), evaluation on 10 held-out scenarios per level vs the noise-free
      baselines (corridor matched and opposite plans, Maribor).
- [x] 2I.3 Benchmarks: held-out evaluation and RESCO's own protocol (best training episode,
      mean over seeds) against RESCO Table 1 and CoSLight.

RESCO Table 1 (best training episode, mean of 5 seeds; 10 s decisions incl. 3 s yellow; 200 m
perfect sensing), average delay / trip time [s]:

| | Ing. single (ingolstadt1) | Ing. corridor (ingolstadt7) | Ing. region (ingolstadt21) | Col. single (cologne1) | Col. corridor (cologne3) | Col. region (cologne8) |
|---|---|---|---|---|---|---|
| IDQN | 21.48 / 35.29 | 31.19 / 68.69 | 59.64 / 197.23 | 26.05 / 43.59 | 23.99 / 59.0 | 22.06 / 86.02 |
| IPPO | 19.85 / 34.19 | 30.7 / 68.34 | 67.65 / 205.44 | 55.07 / 67.7 | 22.13 / 57.45 | 21.49 / 85.54 |
| MPLight | 28.31* / 41.07* | 48.21 / 76.58 | 78.16 / 215.72 | 28.74 / 45.85 | 83.65 / 102.3 | 60.42 / 123.93 |
| FMA2C | 25.36 / 39.4 | 48.99 / 85.03 | 90.42 / 226.5 | 30.12 / 47.31 | 25.37 / 61.68 | 33.28 / 97.53 |

(* extended-state MPLight.)

**Results ([V6_REPORT.md](V6_REPORT.md)):** corridor v6 table 48.4 s, −34 % vs the green wave
(v5: −8 %), unchanged with stale plans; Maribor 72.7 s, −15 % vs SUMO actuated (v4: −5 %);
benchmarks: lower delay than every published method on 6 of 8 RESCO / CoSLight maps, a tie on
Ingolstadt 21. The gain comes from the stage-control RL experts, above all the phase-wise
network; the tables pick them at almost every level. Caveats: noise-free sensing, ~1 teleport
per scenario for RL on our networks.

## Summary (2026-10-12)
1. Free stage choice by RL (the paper's setup) cannot match a green wave on an arterial
   (×3 delay at 4000 veh/h); communication gives it 6–10 %, not the coordination it lacks.
2. RL that *adjusts* a green-wave plan learns steadily and repairs plans that do not fit the
   traffic: −10 to −29 % delay vs the plan and −13 to −38 % vs classic coordinated-actuated
   control on the corridor (opposite-direction plans, significant). With a fitting plan there is
   nothing to repair (tie).
3. On top of it, directional neighbour messages help by 4–7 % at 2000–5000 veh/h (final weights,
   clean comparison) and are used as neighbour-specific information (wrong-neighbour messages
   +2–7 %, significant) — the first setup in this project where communication does that.
4. The best controller differs by junction (Maribor: the best fixed assignment mixes all four
   experts at every level), but learning the choice with DQN failed (corridor and Maribor), and
   on Maribor a single isolated controller (max-pressure at low, actuated at high demand) beats
   every mix.
5. On the real Maribor district, plan-based RL is at best on par with classic coordinated-
   actuated control. Where RL helps depends on the network: arterials with closely spaced
   signals and a plan that has gone stale.

## Later (not now)
- Real counts (municipality / DRSI counters) → routeSampler, real signal plans.
- Step 4: pedestrians at selected crossings.
- Combine 2F and 2G: plan-RL as one of the experts; learned attention over neighbours.

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
- 2026-10-11/12 (overnight): corridor RL sweep (2E.5), plan-based RL (2F, branch
  `2f-plan-control`), Maribor network and baselines (3, `3a-maribor`), controller selection
  (2G, `2g-controller-selection`). Bugs found on the way: plan files rebuilt concurrently by
  parallel jobs (now built once), tlsCycleAdaptation giving zero/too-short greens at low demand
  (now recomputed), detectors on sub-metre OSM lanes (now multi-lane), a 120 s max-red shield
  overriding 122 s plan cycles (selector uses 150 s).
- 2026-10-12: 2H started (branch `2h-hierarchy`): the actuated expert fixed (2H.0), manager
  implemented (2H.1), residual workers (2H.2); training of 2H.3 started.
- 2026-10-12 (evening): 2H.3 done. The jointly trained hierarchy fails on both networks; with the
  delay reward, a default expert and acceptance on fresh scenarios (v2) it reaches SUMO actuated
  on Maribor, and a lookup-table manager (v3c) beats every single expert on the corridor. Two
  worker settings (max-red shield, hold guard) had been handicapping the experts. RL controller
  selection (2G) removed. RESCO benchmark scenarios fetched (`networks/benchmarks/fetch.py`).
- 2026-10-12/13 (overnight): 2H.5 done (expert workers, v4 / v5; HIERARCHY_REPORT §6).
  Benchmarks on the 8 RESCO / CoSLight scenarios and handoff improvements 1–3, 5–7
  ([BENCHMARKS.md](BENCHMARKS.md)). Found on the way: a max-red shield that forces every stage
  turns max-pressure into a fixed cycle on multi-phase junctions (Cologne 1: 18.5 → 76 s); the
  call-based shield (`signals.red_call`, per signal link, only for waiting vehicles) fixes it.
  Also: parallel jobs racing on shared plan files (now an inter-process lock), and an indexing
  bug on networks with gaps in SUMO link indices (Ingolstadt 21).
- 2026-10-13/14 (overnight): step 2I (v6, branch `2i-overnight`): 114 jobs, 0 failed. Residual
  workers, RL experts in the table (per-junction and phase-wise), noise-free detectors,
  benchmarks vs RESCO Table 1 / CoSLight. Results in [V6_REPORT.md](V6_REPORT.md). Found on the
  way: the eval run pattern `v6_<net>_table_*` missed the table runs (fixed before evaluation).
