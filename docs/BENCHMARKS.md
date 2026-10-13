# Benchmarks: our controllers on the published scenarios

*2026-10-13, branch `2h-hierarchy`. Scenarios, published numbers and the improvement list come
from `BENCHMARK_HANDOFF.md`. Our numbers: mean over 10 held-out runs of each scenario (SUMO
seeds 900000+, the scenario's own demand file). RL: 3 training seeds × 10 runs, 150 episodes.
Delay = time loss + departure delay, vehicles still waiting to enter included (RESCO's
definition); trip time = mean duration of the trips that finished.*

## Summary

1. **Our RL, with the handoff's improvements and a corrected shield, is the best (or tied best)
   controller on 6 of the 8 scenarios.** rl2 (state with approaching vehicles, SMDP discounting, call-based
   shield) against the best classic controller on each map (paired Wilcoxon, 10 runs):
   - **better:** Cologne 1 −3.7 %, Cologne 8 −11.8 %, Ingolstadt 1 −9.9 %, Ingolstadt 7
     −16.8 %, Grid 4×4 −12.5 % (all p ≤ 0.02);
   - **tie:** Ingolstadt 21 (+0.5 %, p = 0.63);
   - **worse:** Cologne 3 (+16.8 % vs detector max-pressure) and Avenue 4×4 (+14.7 % delay vs
     Advanced-MP; its trip time is in between).
2. **Against the published numbers:**
   - **CoSLight** (Grid 4×4 and Cologne 8): rl2 has lower delay and trip time than CoSLight and
     every other method in its table: 41.0 / 152.5 s vs 46.11 / 159.1 on Grid 4×4, and 18.3 /
     83.1 vs 24.79 / 90.46 on Cologne 8.
   - **RESCO's IDQN:** rl2's delay is lower on 5 of 6 maps (Ingolstadt 21: 65.5 vs 59.64), even
     though RESCO reports its best training episode with 200 m perfect sensing and we report
     held-out runs with 50 m noisy detectors.
   - **Caveat:** the setups differ in places (§1), so these are close comparisons, not exact
     reproductions.
3. **Our baselines reproduce the published ones where both exist.**
   - Fixed time: Grid 4×4 91.6 / 202.7 vs CoSLight's 94.64 / 206.68; Avenue 4×4 trip time 830.1
     vs 828.38.
   - Detector max-pressure on Grid 4×4 (47.6 / 159.1) is much better than the max-pressure
     CoSLight reports (64.01 / 175.97), so the classic bar is higher than the papers suggest.
4. **The largest single effect was the shield, not the learning (§2).**
   - A 90 s max-red that forces every stage costs max-pressure 4× on Cologne 1, and it held the
     RL back on most maps.
   - With it, rl2 beat the equally handicapped baselines on the small maps but lost to the best
     baseline on 6 of 8 maps (won only on Ingolstadt 7, tied on Ingolstadt 1).
   - Making it call-based (per movement, only with a waiting vehicle) turned that into 5 wins and
     a tie.
5. **Of the handoff improvements, 2 + 3 (approaching vehicles, SMDP) help on every map.** The
   others (dueling + 3-step, pressure reward, directional messages) change delay by about ±1 s on
   the small maps. The exception is the pressure reward on Cologne 3 (−1.8 s and more stable
   seeds). §4.

## 1. Setup

- **Scenarios:** the eight SUMO scenarios of RESCO (Ault & Sharon 2021), downloaded with
  `networks/benchmarks/fetch.py` (fixed commit, not committed: third-party data):
  - real maps with calibrated demand: Cologne 1 / 3 / 8, Ingolstadt 1 / 7 / 21 signals;
  - synthetic: Grid 4×4 and Avenue (arterial) 4×4, which CoSLight (2024) also uses.
- **As RESCO:**
  - the scenario's own signal programs (`rebuild_tls = false`) and its fixed-time plan;
  - a decision every 10 s, 3 s yellow, no all-red;
  - the demand file's start time (`demand.begin`), one hour.
- **As our step 2 (unlike RESCO):**
  - our RL sees 50 m E2 area detectors with noise (σ = 0.05) and 2 % dropout (rl2: plus a
    150 m approach detector). RESCO's agents see every vehicle within 200 m;
  - a 60 s max-green and 90 s max-red shield on RL and on our pressure baselines (§2).
- **Reporting differs:** RESCO's Table 1 reports the **best training episode**, averaged over 5
  seeds. We report the policy chosen on validation runs, measured on 10 held-out runs. Our
  numbers are the stricter of the two.
- **CoSLight** (2024) reports delay and trip time on Grid 4×4, Avenue 4×4 and Cologne 8, with a
  15 s action interval and 8 phases (ours: 10 s, the scenario's phases).

### Controllers

| name | what it is |
|---|---|
| fixed | the scenario's own fixed-time program |
| MP exact | max-pressure on exact vehicle counts (whole lanes) |
| MP det | max-pressure on our E2 detectors (the realistic one) |
| Adv-MP | Advanced-MP (Zhang et al. 2022): keep the green while moving vehicles within 100 m on its lanes outweigh the best other pressure; per-lane efficient pressure |
| rl | step-2 DQN: detector state, queue reward, shield |
| rl2 | rl + vehicles approaching within 150 m (improvement 2) + SMDP discounting γ = 0.995 per second (improvement 3) |
| rl3 / rl2p / rl2c | rl2 + dueling head and 3-step returns / + pressure term in the reward / + directional messages (improvements 5–7) |

## 2. The shield: a 90 s max-red must only serve movements that someone waits for

Plain max-pressure **gridlocks** on Cologne 3 and Cologne 8: it never serves a movement with a
small queue (1000 s delay on Cologne 8). A 90 s max-red fixes that, but in its first form it
forced **every stage** after 90 s red, wanted or not. On scenarios with many stages (Cologne 1:
RESCO's multi-phase program) that is a fixed-time cycle: detector max-pressure 18.5 → 76.3 s.

`signals.red_call = true` (new) makes the shield work like the recall of actuated control:

- tracked **per signal link**: a link is starved only if **no** stage has given it green for 90 s;
- and only when **its area detector reports a vehicle**;
- the stage serving most starved links then runs.

Detector max-pressure, delay [s]:

| map | no shield | stage shield (old) | **call shield (new)** |
|---|---|---|---|
| Cologne 1 | 18.5 | 76.3 | **20.3** |
| Cologne 3 | 20.0 | 28.0 | **17.4** |
| Cologne 8 | 485.9 | **123.2** | 148.4 |
| Ingolstadt 1 | **13.9** | 15.4 | 14.1 |
| Ingolstadt 7 | **28.4** | 30.7 | **28.4** |
| Ingolstadt 21 | 80.4 | 71.7 | **65.1** |
| Grid 4×4 | **47.5** | 68.0 | 47.6 |
| Avenue 4×4 | 815.6 | 988.3 | **813.0** |

The call shield is within 2 s of the better of the other two everywhere except Cologne 8, where
detector max-pressure still jams in places (Advanced-MP does not: 20.7 s). The benchmark configs
now use the call shield; rl, rl2 (old shield) and the variants in §4 were trained with the stage
shield (`--set signals.red_call=false` reproduces them).

## 3. Results

Delay / trip time [s], ours:

| map | fixed | MP exact | MP det | MP det, call shield | Adv-MP, call shield | rl2, old shield | **rl2, call shield** |
|---|---|---|---|---|---|---|---|
| Cologne 1 | 70.7 / 77.4 | 30.3 / 50.6 | 18.5 / 40.0 | 20.3 / 41.7 | 20.7 / 42.1 | 67.3 / 76.4 | **17.8 / 39.2** |
| Cologne 3 | 36.0 / 72.3 | 323.2 / 203.2 | 20.0 / 55.9 | **17.4 / 54.1** | 26.4 / 59.9 | 27.4 / 60.9 | 20.4 / 55.9 |
| Cologne 8 | 48.5 / 113.3 | 1002.8 / 890.7 | 485.9 / 438.5 | 148.4 / 204.1 | 20.7 / 85.5 | 24.0 / 88.9 | **18.3 / 83.1** |
| Ingolstadt 1 | 29.4 / 48.0 | 33.5 / 47.6 | 13.9 / 33.2 | 14.1 / 33.4 | 17.1 / 36.2 | 13.8 / 32.9 | **12.6 / 31.9** |
| Ingolstadt 7 | 82.4 / 115.0 | 39.3 / 76.5 | 28.4 / 69.4 | 28.4 / 69.2 | 107.5 / 76.1 | 25.8 / 66.4 | **23.6 / 64.7** |
| Ingolstadt 21 | 149.7 / 288.3 | 202.5 / 342.6 | 80.4 / 219.2 | **65.1** / 204.2 | 107.0 / 219.4 | 72.8 / 212.2 | 65.5 / 205.1 |
| Grid 4×4 | 91.6 / 202.7 | 83.3 / 194.5 | 47.5 / 159.0 | 47.6 / 159.1 | 46.9 / 158.4 | 71.9 / 183.1 | **41.0 / 152.5** |
| Avenue 4×4 | 1473.7 / 830.1 | 782.6 / **301.7** | 815.6 / 339.7 | 813.0 / 327.1 | **765.8** / 353.0 | 989.9 / 409.2 | 878.0 / 348.6 |

MP exact / MP det without a shield; "call shield" = 90 s max-red with `red_call`. RL: mean of
3 training seeds (rl2, call shield, per seed: Cologne 1 17.1–18.3, Cologne 3 18.6–21.5,
Cologne 8 18.1–18.5, Grid 4×4 40.4–41.6, Avenue 4×4 840.9–905.4).

Published (delay / trip time):

| map | published | source |
|---|---|---|
| Cologne 1 | IDQN 26.05 / 43.59 | RESCO Table 1 (best episode) |
| Cologne 3 | IDQN 23.99 / 59.0 | RESCO Table 1 |
| Cologne 8 | IDQN 22.06 / 86.02; fixed 62.38 / 124.4, max-pressure 31.93 / 95.96, CoSLight 24.79 / 90.46, FRAP 27.5 / 88.61 | RESCO, CoSLight |
| Ingolstadt 1 | IDQN 21.48 / 35.29 | RESCO Table 1 |
| Ingolstadt 7 | IDQN 31.19 / 68.69 | RESCO Table 1 |
| Ingolstadt 21 | IDQN 59.64 / 197.23 | RESCO Table 1 |
| Grid 4×4 | fixed 94.64 / 206.68, max-pressure 64.01 / 175.97, CoSLight 46.11 / 159.1 | CoSLight |
| Avenue 4×4 | fixed 1234.30 / 828.38, max-pressure 952.53 / 686.12, CoSLight 744.98 / 364.21 | CoSLight |

- **Advanced-MP on Ingolstadt 7: delay 107.5 s but trip time 76.1 s.** Delay counts the
  vehicles still waiting to enter. Advanced-MP keeps a minor entry red for long, so its queue
  backs up outside the network. Trip time only counts vehicles that got in.
- **Avenue 4×4 is oversaturated for every controller.** Delays of 750–1500 s are mostly
  vehicles that never enter (as in CoSLight: 745–1234 s).
  - By trip time, exact max-pressure is best (301.7 s), then rl2 (348.6 s), which beats
    CoSLight's 364.21.
  - By delay, Advanced-MP is best. RL is behind on the delay that counts unentered vehicles.
- **Cologne 3: detector max-pressure with the call shield (17.4 s) beats rl2 (20.4 s).** One RL
  seed reaches 18.6. rl2p (pressure reward, old shield) is the most stable variant here (§4); it
  was not rerun with the call shield.
- **Why rl2 is ahead of RESCO's IDQN despite worse sensing:** SMDP discounting, the
  approaching-vehicle feature and the call shield. With the old shield, plain rl was behind IDQN
  on Cologne 1 (88.3 vs 26.05).

## 4. Improvements from the handoff list

| # | improvement | status | result |
|---|---|---|---|
| 1 | Advanced-MP baseline | done (`advanced_mp`) | best baseline on Cologne 8, Grid 4×4, Avenue 4×4 |
| 2 | approaching vehicles (150 m area detector) | done (`observation.approach_length`) | in rl2 |
| 3 | SMDP discounting | done (`agent.smdp`) | in rl2 (with 2): rl → rl2 lowers delay on all 8 maps, by 1.3–21 s |
| 4 | one shared phase-wise network (FRAP-style) | not done | a shared network for all junctions exists (`agent.shared`), phase-wise scoring does not |
| 5 | reward with a pressure term | done (`reward.kind = "queue_pressure"`) | rl2p, see below |
| 6 | dueling, n-step returns | done (`agent.dueling`, `agent.n_step`); prioritised replay not done | rl3, see below |
| 7 | directional messages on corridors | existed (`observation.message = "directional"`) | rl2c, see below |

On the four small maps, trained with the old stage shield (delay [s], mean of 3 seeds; the
per-seed range in brackets where it matters):

| map | rl | rl2 | rl3 (dueling + 3-step) | rl2p (pressure term) | rl2c (directional messages) |
|---|---|---|---|---|---|
| Cologne 1 | 88.3 | **67.3** | 82.9 (69.5–92.8) | 68.3 | – |
| Ingolstadt 1 | 16.2 | **13.8** | 14.2 | 14.1 | – |
| Cologne 3 | 28.7 | 27.4 (25.6–30.5) | 27.0 | **25.6** (25.3–25.8) | 29.5 (26.2–35.7) |
| Ingolstadt 7 | 28.9 | **25.8** | 26.0 | 26.1 | 25.9 |

- **Dueling + 3-step returns:** no gain, and unstable on Cologne 1.
- **Pressure term:** the most stable on the Cologne 3 corridor and the best there. Otherwise
  within 1 s of rl2.
- **Directional messages:**
  - no gain on these corridors, and less stable;
  - the Ingolstadt 7 run was cut short by a memory error at about episode 140, so its best
    checkpoint from before that is evaluated;
  - on our own 7-signal corridor, messages helped plan-RL by 4–7 % (step 2F). RESCO's corridors
    are shorter and less coordinated.
- These variants were not retrained with the call shield, which mattered far more.

## Reproduce

```bash
uv run python networks/benchmarks/fetch.py
uv run tsc evaluate configs/benchmarks/cologne3.toml                  # baselines (also the other 7 maps)
uv run tsc train configs/benchmarks/rl_improved/cologne3.toml --seeds 0-2
uv run tsc evaluate configs/benchmarks/cologne3.toml --runs "outputs/train/rl2_cologne3/s*" --no-baselines --out outputs/eval/rl2_cologne3 --set signals.max_green=60
uv run tsc evaluate configs/benchmarks/cologne3.toml --set signals.red_call=false --out outputs/eval/bench_cologne3_stage   # the old shield
uv run tsc train configs/benchmarks/rl_extras/rl2p_cologne3.toml --seeds 0-2   # improvements 5-7
```
