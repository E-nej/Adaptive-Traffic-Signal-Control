# Step 2I (v6): the benchmark improvements in the hierarchy, and ours vs published

*2026-10-13/14 overnight, branch `2i-overnight`. 114 jobs, 0 failed (`outputs/v6/queue.log`).
Noise-free detectors everywhere (no sensor noise, no dropout). Delay = mean delay per vehicle
[s]; held-out = 10 evaluation scenarios per level (seeds 900000+) that no run trained, validated
or built a table on; RL and tables: mean of 3 training seeds. Paired Wilcoxon over scenarios,
p < 0.05 = significant.*

## Summary

1. **Corridor: −34 % delay vs the green wave** (v6 table 48.4 s vs 73.8 s; v5 was 67.7 s,
   −8 %). Significant at every level (−52 % at 1000 veh/h … −23 % at 5000). With stale
   (opposite-direction) plans it does not change at all (48.4 s), so it is −48 % vs the best
   classic controller there (max-pressure, 92.8 s; v5 was 98.3 s).
2. **Maribor: −15 % vs SUMO actuated** (72.7 s vs 85.9 s; v4 was 82.0 s, −5 %), better than
   every classic controller at every level.
3. **Benchmarks (RESCO / CoSLight maps): our noise-free RL has lower delay than every
   published method on 6 of 8 maps**, is level with RESCO's best on Ingolstadt 21 (62.3
   held-out / 59.9 RESCO protocol vs IDQN 59.64), and on Avenue 4×4 has a better trip time than
   CoSLight (329.7 vs 364.2) but a higher delay (773–793 vs 745). Against the best classic controller on each map it wins on 6
   (−11 to −45 %, significant), ties on Ingolstadt 21, and loses on Cologne 3 (+9 %) and on the
   oversaturated Avenue 4×4 by delay (+4 %, phase-wise net), though not by trip time.
4. **What did it:** the stage-control RL experts (approach detector + SMDP, noise-free), above
   all the new **phase-wise network** (FRAP / MPLight style, handoff item 4). The tables pick
   an RL expert at almost every level, so the workers, now much better, are not used.
5. **Workers:** the new recipe (start as the expert, SMDP, approach detector) passes acceptance
   far more often: corridor green wave 3/3 seeds (−21 to −30 %; 2H.5: 2/3, −15 to −21 %),
   coordinated-actuated 3/3 (−6 to −13 %; 2H.5: 0/3). Maribor 0/6 (max-pressure workers −12 to
   −19 % but not significant on 12 scenarios; admitted as candidates, the table still preferred
   RL).

## 1. What changed (details in PLAN_REALISM.md, step 2I)

| change | where |
|---|---|
| noise-free detectors | all `configs/v6/` |
| workers: SMDP (γ = 0.995 / s), 150 m approach detector, residual head that starts as the expert (`agent.ref_action = 0`), ε from 0.2, 150 episodes, validation on 2 scenarios / level | `configs/v6/workers/` |
| RL experts the table can pick: `rl` (per-junction DQN, as rl2) and `rlf` (one phase-wise network for all junctions) | `RLExpert` in `controllers/select.py`; `agent.phase_net`, `observation.layout = "phase"` |
| table over 6 experts, each classic one with or without its worker per level | `configs/v6/tables/` |
| benchmarks: rl2 without noise (`rl2nf`) and the phase-wise network (`rl2f`) | `configs/v6/bench/` |

## 2. Our networks

**Corridor (7 signals), matched plans, held-out:**

| controller | 1000 | 2000 | 3000 | 4000 | 5000 | mean |
|---|---|---|---|---|---|---|
| green wave (best classic, the reference) | 51.6 | 50.0 | 57.7 | 73.5 | 136.0 | 73.8 |
| coordinated-actuated | 54.8 | 74.1 | 68.6 | 72.9 | 112.6 | 76.6 |
| max-pressure (detectors) | 36.4 | 46.5 | 57.6 | 103.7 | 219.7 | 92.8 |
| actuated (SUMO) | 62.9 | 67.2 | 75.7 | 99.3 | 208.9 | 102.8 |
| v5 (2H.5, noisy detectors) | 41.6 | 51.9 | 57.9 | 72.1 | 115.2 | 67.7 |
| RL expert alone, per-junction (`rl`) | 24.6 | 28.8 | 34.9 | 49.4 | 113.2 | 50.2 |
| RL expert alone, phase-wise (`rlf`) | 29.6 | 33.2 | 38.7 | 51.5 | 107.5 | 52.1 |
| **v6 table** | **24.6** | **28.8** | **34.9** | **49.5** | **104.3** | **48.4** |

v6 vs green wave: −51.7 / −42.4 / −39.4 / −32.6 / −22.6 % (all significant). Trip time 187 vs
213 s, stops 1.43 vs 1.73 per vehicle, throughput 0.967 vs 0.963. The tables choose `rl` up to
3500–4500 veh/h and `rlf` above.

**Corridor, opposite-direction (stale) plans:** v6 48.4 s (it uses no plan expert), green wave
118.1, coordinated-actuated 111.1, actuated 102.8, max-pressure 92.8 (v5: 98.3, v4: 96.1).

**Maribor (21 signals), held-out:**

| controller | 2000 | 4000 | 6000 | 7000 | 8000 | mean |
|---|---|---|---|---|---|---|
| actuated (SUMO, best classic) | 50.2 | 57.1 | 70.3 | 93.2 | 158.8 | 85.9 |
| max-pressure (detectors, noise-free) | 36.1 | 43.1 | 68.5 | 107.8 | 179.9 | 87.1 |
| green wave | 55.3 | 71.7 | 91.2 | 115.9 | 171.3 | 101.1 |
| v4 (2H.5, noisy detectors) | | | | | | 82.0 |
| RL expert alone, per-junction (`rl`) | 31.2 | 45.8 | 96.8 | 145.5 | 217.4 | 107.4 |
| RL expert alone, phase-wise (`rlf`) | 31.4 | 38.4 | 55.4 | 88.1 | 149.2 | 72.5 |
| **v6 table** | **30.8** | **39.5** | **55.3** | **88.2** | **149.6** | **72.7** |
| v6 table with candidate MP workers (`tablew`) | 30.8 | 39.7 | 55.3 | 88.1 | 149.6 | 72.7 |

Trip time 254 vs 268 s (actuated), stops 1.80 vs 2.22, throughput 0.949 vs 0.948. The phase-wise
network is what wins here: with 21 junctions sharing one network it has 21× the data of the
per-junction agents, which fall apart above 4000 veh/h. Tables: `rl` at 2000–3500 in two seeds,
`rlf` elsewhere, SUMO-like actuated at 8000 in one seed.

## 3. Ours vs published (RESCO Table 1, CoSLight 2024)

Delay / trip time [s]. **Ours** = noise-free RL with the call-based shield (60 s max green, 90 s
max red only for waiting vehicles), 3 seeds. Two protocols: *held-out* (best-validation policy on
10 held-out runs; ours, strict) and *RESCO* (best training episode per seed, mean over seeds:
how RESCO Table 1 reports). Full tables: `outputs/v6/benchmarks.md`.

| map | **ours, per-junction (rl2nf)** held-out | **ours, phase-wise (rl2f)** held-out | ours, best of the two, RESCO protocol | best published | best classic (our runs) |
|---|---|---|---|---|---|
| Cologne 1 | 17.23 / 38.6 | **16.92 / 38.3** | 16.04 | IDQN 26.05 / 43.59 | MP det 20.42 / 41.8 |
| Cologne 3 | 19.00 / 54.7 | 18.85 / 55.1 | 17.68 | IPPO 22.13 / 57.45 | **MP det 17.43 / 54.1** |
| Cologne 8 | **17.22 / 82.0** | 18.58 / 83.4 | 16.86 | IPPO 21.49 / 85.54 (CoSLight 24.79 / 90.46) | Adv-MP 20.71 / 85.5 |
| Ingolstadt 1 | **12.22 / 31.5** | 12.53 / 31.9 | 11.37 | IPPO 19.85 / 34.19 | MP det 14.13 / 33.4 |
| Ingolstadt 7 | **22.37 / 63.5** | 23.91 / 64.8 | 22.06 | IPPO 30.70 / 68.34 | MP det 28.36 / 69.2 |
| Ingolstadt 21 | **62.28 / 201.8** | 62.68 / 201.1 | 59.88 | **IDQN 59.64 / 197.23** | MP det 65.13 / 204.2 |
| Grid 4×4 | 27.89 / 139.5 | **25.97 / 137.6** | 26.52 | CoSLight 46.11 / 159.1 | Adv-MP 46.90 / 158.4 |
| Avenue 4×4 | 874.5 / 356.4 | 793.3 / **329.7** | 772.9 | **CoSLight 744.98** / 364.21 | Adv-MP 765.8 / 353.0 |

- **vs published:** lower delay on 6 of 8 maps. Ingolstadt 21 is a tie with IDQN (59.88 vs
  59.64, RESCO protocol). On Avenue 4×4 our trip time beats CoSLight (329.7 vs 364.2) but
  CoSLight's delay, which counts the vehicles that never enter, is lower (745 vs 773–793). Against MPLight, the published
  phase-wise network, ours is far ahead everywhere (e.g. Cologne 3 18.9 vs 83.65, Cologne 8
  18.6 vs 60.42): MPLight's divergence on irregular junctions does not appear.
- **vs the best classic controller (paired, held-out):** Cologne 1 −15.5 / −17.0 %, Cologne 8
  −16.8 / −10.2 %, Ingolstadt 1 −13.3 / −11.1 %, Ingolstadt 7 −21.1 / −15.7 %, Grid 4×4
  −40.5 / −44.6 % (rl2nf / rl2f, all significant); Ingolstadt 21 −3.9 / −3.3 % (n.s.);
  Cologne 3 +9.2 / +8.6 %; Avenue 4×4 +14.2 / +3.6 % (delay; by trip time rl2f is level with
  detector max-pressure, 329.7 vs 327.1).
- **Noise matters:** without sensor noise rl2 improves on every map vs the noisy rl2 of
  BENCHMARKS.md (e.g. Grid 4×4 41.0 → 27.9, Cologne 8 18.3 → 17.2, Ingolstadt 7 23.6 → 22.4).

## 4. Caveats

- **Noise-free sensing** (by choice for this round). With noise and dropout the numbers are
  worse (BENCHMARKS.md, 2H). Noise-free detector max-pressure on Maribor is *worse* than the
  noisy one of 2H (87.1 vs 83.3 s): noise breaks its tendency to hold a stage (noted in step 2).
- **Teleports** (a vehicle stuck > 300 s is removed by SUMO): about 1 per scenario for the RL
  controllers on the corridor and Maribor, vs 0.2 for the green wave / actuated. Few, but real,
  probably left-turn bays at high demand; worth a look before publishing.
- **Vehicles still waiting to enter** at the end (not in the delay on our own networks):
  Maribor v6 9.4 per scenario vs actuated 0.3 (of ~5000–8000 trips); corridor 0.4 vs 0.
- **Comparison with papers:** close, not exact. RESCO's agents see every vehicle within 200 m,
  ours use 50 m + 150 m area detectors; RESCO averages 5 seeds, we use 3; CoSLight uses 15 s
  actions and 8 phases. The RESCO-protocol numbers include training episodes with exploration,
  as RESCO's do.
- **The hierarchy's gain is the RL experts'.** The table adds safety (it falls back to a classic
  expert where RL is worse: SUMO-like actuated at 8000 veh/h in one Maribor seed) but little
  delay beyond the best RL expert alone (48.4 vs 50.2 s corridor, 72.7 vs 72.5 s Maribor).
- On the corridor the new free-stage RL beats the green wave, which 2E.5 found impossible
  (×3 worse). The difference: approach detector, SMDP, no communication, noise-free detectors.
  Checked: same insertions, throughput and fewer stops, not a metric artefact.

## 5. Next

1. Put the noise / dropout back and rerun (how much of this survives realistic sensing?).
2. Teleports of the RL experts: find the junctions, add the call-based shield to the RL experts
   of our networks, or a penalty.
3. More seeds (5, as RESCO) for the benchmark rows that are close: Ingolstadt 21, Cologne 3.
4. Cologne 3 and Avenue 4×4: the two maps where a classic controller still wins.
5. Workers on top of the RL experts (they are what the table uses).

## Reproduce

```bash
python configs/v6/make_configs.py          # configs/v6/{workers,rl,tables,eval,bench}
python configs/v6/make_jobs.py             # outputs/v6/jobs.txt (114 jobs + 3 tablew)
python configs/v6/run_queue.py outputs/v6/jobs.txt --slots 15
python configs/v6/admit_candidates.py outputs/train/v6_wk_maribor_maxpressure   # tablew
python configs/v6/summarize.py > outputs/v6/benchmarks.md
```
