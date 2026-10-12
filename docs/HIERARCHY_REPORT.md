# Step 2H: hierarchical control. Results and what we changed

*2026-10-12, branch `2h-hierarchy`. All delays are mean delay per vehicle [s], 10 held-out
evaluation scenarios per demand level, every controller on the same scenarios (paired
comparisons, Wilcoxon p < 0.05 = significant).*

## Summary

1. **The hierarchy as trained first (2H.3) fails on both networks.** A DQN manager picking
   experts every 5 min with residual workers fine-tuning them, both learning together, rewarded
   for short queues: up to **+104 % delay vs the green wave** on the corridor and **+23–39 % vs
   SUMO actuated at 6000–8000 veh/h** on Maribor.
2. **Diagnosis (corridor ablations):** the manager causes most of the loss (its reward, queue
   length, prefers max-pressure, which has the shortest queues but twice the delay of the green
   wave); the workers add more (+5 to +42 % on top of the green wave, trained under the wrong
   experts). The hierarchy machinery itself is exact (the green wave run through it equals the
   green-wave baseline).
3. **v2 (DQN manager, delay reward, default expert, workers off)** fixes most of it: corridor
   from +100 % to +7–16 % vs the green wave; Maribor (group scope) as good as SUMO actuated
   overall (85.8 vs 85.9 s), −9 to −18 % at 2000–4000 veh/h.
4. **Nearly all of what expert selection can gain comes from choosing by demand level**
   (perfect per-level choice vs per-scenario choice differ by about 1 %). So the DQN manager
   was replaced by a **lookup table** (v3): run each expert at a grid of levels, then pick by the
   measured flow and jam. With the experts set up exactly as their baselines (point 5) it **beats
   every single expert on the corridor** (69.4 vs 73.8 s with matching plans, −6 %; 99.6 vs
   102.8 s with stale plans) and **equals the best one on Maribor** (83.3 s).
5. **The last gap was two settings, not the method.** The hierarchy configs used a 150 s max-red
   shield and a plan hold guard, both meant for RL, that the baselines do not have. They cost
   max-pressure 5.5 % and coordinated-actuated up to 53 %. Without them the hierarchy's experts
   equal their baselines exactly (§5, v3c).

## 1. The first round (configs/hierarchy/*.toml, as requested)

Manager: DQN, every 300 s, reward −queue. Workers: shared DQN per junction, AGREE / HOLD /
SWITCH every 5 s of green, reward `multi` + 0.5 × neighbours. 3 seeds × 150 episodes.

**Corridor (7 signals), matched plans:**

| controller | 1000 | 2000 | 3000 | 4000 | 5000 | mean |
|---|---|---|---|---|---|---|
| green wave (best single) | 51.6 | 50.0 | 57.7 | 73.5 | 136.0 | **73.8** |
| coordinated-actuated | 54.8 | 74.1 | 68.6 | 72.9 | **112.6** | 76.6 |
| max-pressure (detectors) | **46.6** | 55.2 | 66.3 | 137.9 | 257.8 | 112.8 |
| actuated (SUMO) | 62.9 | 67.2 | 75.7 | 99.3 | 208.9 | 102.8 |
| hierarchy, first round | 55.7 | 58.9 | 68.9 | 143.8 | 272.2 | 119.9 |

vs the green wave: +9 / +18 / +20 / +96 / +104 % (all significant). The manager picked
max-pressure 75–89 % of the time by the end of training.

**Maribor (Titova cesta, 21 signals):**

| controller | 2000 | 4000 | 6000 | 7000 | 8000 | mean |
|---|---|---|---|---|---|---|
| max-pressure (detectors) | **38.3** | **45.3** | **66.6** | 100.8 | 165.7 | **83.3** |
| actuated (SUMO) | 50.2 | 57.1 | 70.3 | **93.2** | **158.8** | 85.9 |
| green wave | 55.3 | 71.7 | 91.2 | 115.9 | 171.3 | 101.1 |
| first round, one expert for the network | 48.6 | 52.1 | 77.6 | 123.6 | 213.8 | 103.1 |
| first round, one expert per group (10 groups) | 47.8 | 54.8 | 87.4 | 122.8 | 198.4 | 102.2 |

Better than actuated at low demand, +23 to +39 % worse at 6000–8000 (significant), with
teleports at 8000 (up to 8 per scenario).

**Is training length the problem? No.** Validation delay by third of training (mean of 3 seeds):
corridor 168 → 139 → 117 (still improving, but towards max-pressure; the best expert scores 90),
Maribor 126 → 123 → 120 and 129 → 122 → 121 (flat, actuated alone scores 105).

## 2. Diagnosis on the corridor

| corridor, matched plans | 1000 | 2000 | 3000 | 4000 | 5000 |
|---|---|---|---|---|---|
| green wave baseline | 51.6 | 50.0 | 57.7 | 73.5 | 136.0 |
| hierarchy fixed to the green wave, workers off | 55.1 | 50.0 | 57.6 | 73.6 | 138.6 |
| hierarchy fixed to the green wave, trained workers | 57.0 | 65.4 | 69.8 | 98.8 | 193.4 |
| trained manager, workers off | 52.1 | 55.2 | 68.1 | 141.2 | 272.2 |
| full first-round hierarchy | 55.7 | 58.9 | 68.9 | 143.8 | 272.2 |

- **Machinery:** exact (row 2 = row 1).
- **Manager:** responsible for most of the loss (row 4 ≈ max-pressure). Its reward is
  misaligned: max-pressure has the **shortest queues** at 4 of 5 levels (7.5 vs 9.2 vehicles at
  5000) but **twice the delay and stops** of the green wave. The manager learned what it was
  paid for.
- **Workers:** hurt on top of the green wave (+5 to +42 %). They trained while the manager ran
  max-pressure most of the time, overrode the expert on about 25 % of decisions, and were not
  rewarded for delay either.

## 3. Fixes and the variants they produced

| change | where |
|---|---|
| `delay` reward: per-second time lost on the approaches, Σ n·(1 − v̄/v_max) | `reward.py` |
| default expert (`default_expert = "auto"`: best single expert on validation, measured before training) + Q-margin: the manager leaves it only if another expert is better by 5 % | `hierarchy.py`, `train.py` |
| fallback: the default alone is the first `best` checkpoint; a run deploys it unless its manager beats it | `train.py`, `rl.py` (meta file per checkpoint) |
| `tsc accept`: the best checkpoint vs its default on **fresh** scenarios (seeds 750000+), paired, one-sided Wilcoxon p < 0.05, else the default is deployed | `train.py`, `cli.py`, `seeds.py` |
| `manager = "table"`: lookup table instead of DQN | `hierarchy.py`, `train.py` |

- **v2** = DQN manager with the delay reward, default expert, margin and fallback; workers off.
- **v3** = table manager (3 scenarios per level, one key per level).
- **v3b** = table keyed per 5-minute reading (the network fills up from empty, so the k-th reading
  is compared with each level's k-th reading), 10 / 6 scenarios per level (corridor / Maribor).
- **v3c** = v3b without the 150 s max-red shield and the plan hold guard (§5).

### Why a lookup table: what perfect selection could gain

Best expert chosen per scenario or per level, from the evaluation results:

| corridor | best one expert for all levels | perfect pick per level | perfect pick per scenario |
|---|---|---|---|
| matched plans | 73.8 (green wave) | **68.0 (−8 %)** | 67.6 |
| opposite plans | 102.8 (actuated) | **91.0 (−11 %)** | 90.2 |

Maribor: max-pressure alone 83.3, perfect per level 80.4 (−3.5 %). Within a level, a perfect
per-scenario choice adds only about 1 %: **the right expert depends on the demand level**, which a
table captures directly and DQN learns poorly from noisy 5-minute rewards (12 decisions per
episode). No selector can beat the best expert *at* a level, only match it there; it beats every
single expert overall by matching the best one at every level.

## 4. Results of v2 and v3

**Corridor, matched plans:**

| controller | 1000 | 2000 | 3000 | 4000 | 5000 | mean |
|---|---|---|---|---|---|---|
| green wave (best single) | 51.6 | 50.0 | 57.7 | 73.5 | 136.0 | 73.8 |
| first round | 55.7 | 58.9 | 68.9 | 143.8 | 272.2 | 119.9 |
| v2 (DQN, delay, default) | 49.7 | 57.8 | 65.1 | 80.7 | 145.0 | 79.7 |
| v3 (table) | 49.8 | 51.9 | 59.1 | 74.0 | 138.6 | 74.7 |
| v3b (table per reading, more scenarios) | 49.8 | 52.4 | 57.6 | 73.6 | 138.6 | 74.4 |

**Corridor, opposite (stale) plans:**

| controller | 1000 | 2000 | 3000 | 4000 | 5000 | mean |
|---|---|---|---|---|---|---|
| green wave | 66.3 | 70.2 | 79.6 | 123.9 | 250.5 | 118.1 |
| max-pressure | **46.6** | **55.2** | **66.3** | 137.9 | 257.8 | 112.8 |
| coordinated-actuated | 65.9 | 96.1 | 97.5 | 108.6 | **187.4** | 111.1 |
| actuated (SUMO) | 62.9 | 67.2 | 75.7 | **99.3** | 208.9 | 102.8 |
| first round | 60.5 | 64.6 | 72.5 | 150.9 | 280.3 | 125.8 |
| v2 | 50.1 | 64.2 | 78.9 | 122.3 | 243.6 | 111.8 |
| v3 | 49.0 | 66.3 | 77.4 | 120.2 | 240.2 | 110.6 |
| v3b | 49.0 | 68.8 | 79.6 | 124.1 | 240.2 | 112.4 |

The tables are built with matching plans, so with stale plans they still pick the green wave
above 1500 veh/h. A "does the plan fit" input is the open item here.

**Maribor:**

| controller | 2000 | 4000 | 6000 | 7000 | 8000 | mean |
|---|---|---|---|---|---|---|
| max-pressure (best single) | 38.3 | 45.3 | 66.6 | 100.8 | 165.7 | 83.3 |
| actuated (SUMO) | 50.2 | 57.1 | 70.3 | 93.2 | 158.8 | 85.9 |
| v2, one expert for the network | 49.3 | 55.8 | 75.2 | 107.1 | 165.2 | 90.5 |
| v2, one expert per group | 40.9 | 51.9 | 75.0 | 102.1 | 159.2 | 85.8 |
| v3b (table) | 40.4 | 47.8 | 75.2 | 99.5 | 163.3 | 85.2 |

- **v2:** the validation curves are flat and noisy (3 scenarios), and the
  checkpoint picked as the best of 15 readings looks better than it is. On the corridor all
  three v2 managers "beat" the green wave on validation (88–89 vs 90 s) and lose on the test
  (+7 to +16 %).
- **Acceptance (`tsc accept`) catches this:** on fresh scenarios the corridor v2 managers are not
  significantly better (p = 0.11–0.63) and fall back to the green wave. On Maribor, v2 network falls
  back to actuated (3 of 3); v2 group keeps its manager in 2 of 3 seeds (−4.2 / −4.6 % vs
  actuated, p ≤ 0.005).
- **v2 group mixes experts by place, not by demand:** about 45 % actuated and 35 % max-pressure across
  groups at every level (as the best fixed assignment of step 2G).
- **The Maribor table finds the per-level optimum:** max-pressure up to 5750 veh/h, actuated
  from 6500. The corridor table: max-pressure up to 1500, the green wave above.

## 5. Expert parity: two settings made the experts weaker than their baselines

The hierarchy configs carry two settings meant for the RL workers, which the classic baselines
run without:

- `max_red = 150`: a starvation shield;
- `hold_guard = true`: a plan stage is extended only if the next stage keeps its minimum green
  (added for plan RL on Maribor, step 3.6).

Each expert, run alone through the hierarchy, against its baseline:

| | Maribor 2000 | Maribor 4000 | corridor 4000 | corridor 5000 |
|---|---|---|---|---|
| max-pressure baseline | 38.3 | 45.3 | | |
| hierarchy max-pressure, `max_red = 150` | 40.4 | 47.8 | | |
| hierarchy max-pressure, no max-red | **38.3** | **45.3** | | |
| coordinated-actuated baseline | | | 72.9 | 112.6 |
| hierarchy coordinated-actuated, `hold_guard = true` | | | 80.1 | 172.1 |
| hierarchy coordinated-actuated, no guard | | | **72.9** | **112.6** |

Without them every expert equals its baseline exactly.

- **v3b on Maribor:** its gap to max-pressure at 2000–4000 (40.4 / 47.8 s) is precisely the shield.
- **Corridor tables:** the guard is why they never picked coordinated-actuated at high demand. It
  made that expert 53 % worse at 5000 veh/h, where it is in fact the best one.
- **Earlier variants affected:** the first round and v2 carried both settings as well, so their
  experts were handicapped too. For the workers the settings stay: they are for RL.
- **v3c** = v3b with neither setting.

**v3c results** (table without either setting; corridor table: max-pressure up to 1500 veh/h,
green wave 2000–3500, coordinated-actuated from 4000; Maribor: max-pressure up to 7250,
actuated at 8000):

| setting | best single expert (mean) | **v3c** | perfect pick per level |
|---|---|---|---|
| corridor, matched plans | 73.8 (green wave) | **69.4 (−6 %)** | 68.0 |
| corridor, opposite plans | 102.8 (actuated) | **99.6 (−3 %)** | 91.0 |
| Maribor | 83.3 (max-pressure) | **83.3 (equal)** | 80.4 |

Corridor, matched plans, v3c per level: 46.7 / 52.1 / 57.7 / 70.4 / 120.0 vs the green wave
51.6 / 50.0 / 57.7 / 73.5 / 136.0. That is −4.2 % at 4000 and −10.9 % at 5000 (significant), equal at
3000, and +4.3 % at 2000 (significant: the max-pressure start before the first readings
settle). With opposite plans it still runs the green wave at 2000–3000 (+20–24 % vs
max-pressure there): it cannot tell the plan is stale.

On Maribor it equals max-pressure at every level. Its actuated expert is the Python copy of SUMO's
actuated control, 2–12 % weaker than SUMO's own at high demand (2H.0). So it barely beats
max-pressure at 7000–8000 veh/h, where SUMO's actuated control does.

**The hierarchy now meets the target: at least as good as the best single expert in every
setting, and better overall on the corridor (−6 % matched, −3 % stale plans).**

## 6. What this means

- An end-to-end two-level RL hierarchy is worse than the classic controllers, and training longer does not fix it.
- With the reward aligned to delay, a safe default, acceptance on fresh scenarios and decoupled
  levels, it reaches the best classic controller. The remaining gains come from **structure**
  (which expert, where and at which demand) more than from deeper RL.
- **Next:**
  1. a "does the plan fit" input for the table (stale plans), and handing a junction to SUMO's
     own actuated program while that expert is active (Maribor at 7000–8000);
  2. a table per group (combine v2 group's spatial mix with the per-level choice);
  3. **expert-specific workers**: one worker per expert, trained with that expert always active
     and the delay reward, kept only if it beats its expert alone, then the table rebuilt over
     "expert + its worker". The plan-RL results of step 2F (−10 to −29 % vs a stale plan on the
     corridor) say where workers can win.

## Reproduce

```bash
uv run tsc train configs/hierarchy/maribor_group.toml --seeds 0-2       # first round (also maribor_network, corridor_network)
uv run tsc train configs/hierarchy/v2/corridor_network.toml --seeds 0-2  # v2 (also maribor_*)
uv run tsc accept "outputs/train/hier2_corridor_network/s*" --scenarios 6
uv run tsc train configs/hierarchy/v3c/corridor_network.toml --seeds 0   # table (also maribor_network)
uv run tsc evaluate configs/hierarchy/evaluate_corridor.toml --runs "outputs/train/hier3c_corridor_network/s*"
uv run tsc evaluate configs/hierarchy/evaluate_maribor.toml
uv run tsc compare outputs/eval/hier_maribor/results.csv --ref maxpressure_det
```
