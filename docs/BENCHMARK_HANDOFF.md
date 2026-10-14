# Handoff: papers to benchmark against + improvements to implement

Written from a separate chat that compared our step-1/step-2 results with the literature.
Numbers marked (!) were read from garbled PDF text extraction. Re-check them against the
original paper before relying on them.

## Our position in one line
On held-out seeds, RL (`queue`, `full_nocomm`) is within about ±3 % delay of max-pressure + shield.
Published "best RL vs best rule-based" gaps are about 2.5–8.5 % in travel time, which I estimate at
roughly 5–10 % in delay (conversion assumed, not measured).

---

## Part 1 — Papers and their scenarios

### 1. RESCO (Ault & Sharon, NeurIPS 2021 Datasets & Benchmarks) — EXACT reproduction possible
- Paper: https://people.engr.tamu.edu/guni/papers/NeurIPS-signals.pdf — code: github.com/Pi-Star-Lab/RESCO
- Simulator: SUMO. Scenarios: Cologne single / corridor / region (cologne1/3/8), Ingolstadt single /
  corridor / region (ingolstadt1/7/21). Network + calibrated demand ship with the repo.
- Setup: decisions every 10 s with the first 3 s used for yellow when switching. Sensing within 200 m.
  Hyper-parameters are not tuned per scenario.
- Metric: average delay (trip time minus minimum possible trip time), plus trip time, wait and queue.
- Reporting caveat: Table 1 reports the **best training episode**, averaged over 5 seeds, not the final policy.
- IDQN row from Table 1 (Ing single, Ing corr, Ing reg, Col single, Col corr, Col reg):
  - avg delay: 21.48, 31.19, 59.64, 26.05, 23.99, 22.06
  - avg trip time: 35.29, 68.69, 197.23, 43.59, 59.0, 86.02
- The other rows (IPPO, MPLight, FMA2C) and the fixed-time / max-pressure baselines were garbled in
  the extraction. Take them from the paper or the README results image.
- Findings: IDQN is the most stable; MPLight diverges on irregular junctions; IPPO is unstable.

### 2. CoSLight (2024) — mostly EXACT (SUMO)
- https://arxiv.org/html/2405.17152v2
- Simulator: SUMO. Episodes are 3600 s, the action interval is 15 s, and there are 8 phases.
- Scenarios:
  - Grid 4×4, Avenue 4×4 and Grid 5×5 (synthetic). The 4×4 grids appear to be the RESCO/FMA2C ones and
    Grid 5×5 the MA2C (Chu 2019) one; both are unverified.
  - Cologne8 (in RESCO).
  - Nanshan (find its source, probably the CoSLight/DuaLight repo).
- Average trip time (s): Grid4x4 / Avenue4x4 / Grid5x5 / Cologne8 / Nanshan

  | Method | G4x4 | Av4x4 | G5x5 | Col8 | Nanshan |
  |---|---|---|---|---|---|
  | FixedTime | 206.68 | 828.38 | 550.38 | 124.4 | 729.02 |
  | MaxPressure | 175.97 | 686.12 | 274.15 | 95.96 | 720.89 |
  | IPPO | 167.62 | 431.31 | 259.28 | 90.87 | 743.69 |
  | MAPPO | 164.96 | 565.67 | 300.90 | 97.68 | 744.47 |
  | FRAP | 161.58 | 383.71 | 238.41 | 88.61 | 709.18 |
  | MPLight | 179.51 | 541.29 | 261.76 | 98.44 | 668.81 |
  | CoLight | 163.52 | 409.93 | 242.37 | 89.72 | 608.01 |
  | Advanced-CoLight | 171.63 | 421.44 | 237.67 | 91.22 | 612.34 |
  | DuaLight | 161.0 | 396.65 | 221.83 | 89.74 | 609.89 |
  | CoSLight | 159.1 | 364.21 | 220.32 | 90.46 | 621.23 |

- Average delay (s):

  | Method | G4x4 | Av4x4 | G5x5 | Col8 | Nanshan |
  |---|---|---|---|---|---|
  | FixedTime | 94.64 | 1234.30 | 790.18 | 62.38 | 561.69 |
  | MaxPressure | 64.01 | 952.53 | 240.00 | 31.93 | 553.94 |
  | IPPO | 56.38 | 914.58 | 243.58 | 26.82 | 577.99 |
  | FRAP | 50.02 | 794.13 | 203.95 | 27.5 | 542.43 |
  | CoLight | 51.58 | 776.61 | 248.32 | 25.56 | 428.95 |
  | CoSLight | 46.11 | 744.98 | 178.54 | 24.79 | 476.57 |

### 3. Advanced-MP / Advanced-XLight ("Expression might be enough", Zhang et al., ICML 2022) — APPROXIMATE
- https://arxiv.org/abs/2112.10107 — code: github.com/LiangZhang1996/Advanced_XLight
- Simulator: **CityFlow**. Datasets: JiNan 1-3 (12 junctions), HangZhou 1-2 (16), NewYork (196), from
  traffic-signal-control.github.io. LibSignal has SUMO conversions of some of them; any result there is
  close to the paper's, not identical.
- Setup: 4 phases, 15 s action duration, 3 s yellow + 2 s all-red, 60-min episodes. Results are the mean
  of the last 10 test episodes, averaged over 3 runs.
- Metric: average travel time.
- Table 2 (!), columns JiNan1 / JiNan2 / JiNan3 / HangZhou1 / HangZhou2 / NewYork:

  | Method | J1 | J2 | J3 | H1 | H2 | NY |
  |---|---|---|---|---|---|---|
  | FixedTime | 428.11 | 368.77 | 383.01 | 495.57 | 406.65 | 1507.12 |
  | MaxPressure | 273.96 | 245.38 | 245.81 | 288.54 | 348.98 | 1179.55 |
  | Efficient-MP | 269.87 | 239.75 | 240.03 | 284.44 | 327.62 | 1122.00 |
  | MPLight | 297.46 | 270.05 | 276.15 | 314.60 | 357.61 | 1321.40 |
  | CoLight | 272.06 | 252.44 | 249.56 | 297.02 | 347.27 | 1065.64 |
  | Efficient-CoLight | 256.84 | 239.58 | 236.72 | 282.07 | 324.27 | 1032.11 |
  | Advanced-MP | 253.61 | 238.62 | 235.21 | 279.47 | 318.67 | 1060.41 |
  | Advanced-MPLight | 251.29 | 234.78 | 231.76 | 273.26 | 312.68 | 1198.64 |
  | Advanced-CoLight | 245.73 | 232.63 | 229.01 | 270.45 | 310.74 | 970.05 |

### 4. DHLight (arXiv 2409.05037) / SocialLight (arXiv 2305.16145) — same CityFlow data as #3
- DHLight reports on Hangzhou / Jinan: MaxPressure 407.17 / 343.90, PressLight 416.73 / 311.39,
  CoLight 302.37 / 309.93.
- SocialLight reports CoLight and MPLight *worse* than MaxPressure on the same datasets.
- Use: only as evidence that baseline numbers differ a lot between papers. Not worth reproducing.

### 5. "Efficiency and Equity are Both Essential" (arXiv 2003.04046) — APPROXIMATE only
- SUMO, single 4-arm junction, 50 km/h. 1-hour demand episodes in bins 500–5500 veh/h, 30 episodes per bin.
  A real whole-day demand from Freiburg (Loerracherstr./Wiesentalstr.) is probably not public.
- Results are in figures only:
  - More than 20 % better travel time than max-pressure at medium demand.
  - Freiburg mean travel time (!), probably Uniform 141 s / Webster 77 s / Max-pressure 72 s / Ours 51 s,
    i.e. about 29 % better than max-pressure. The order of methods in the figure is uncertain.
- Comparison: rebuild a similar single junction and demand sweep; compare relative gains only.

### 6. LibSignal (arXiv 2211.10649)
- A library with the same methods and datasets in SUMO and CityFlow. Use it as the source of SUMO
  versions of the CityFlow datasets. I did not extract any numbers.

### Lower priority (no numbers extracted)
DynamicLight (2211.01025), PDLight (2009.13711), GPLight (2403.17328), X-Light (2404.12090, cross-city
transfer), Unicorn (2503.11488).

### Suggested order
RESCO → CoSLight's SUMO scenarios → LibSignal SUMO versions of JiNan/HangZhou → the Efficiency &
Equity–style single junction. First check that our max-pressure and fixed-time reproduce the papers'
baseline numbers on each scenario.

---

## Part 2 — Improvements to implement (in this order)

1. **Advanced-MP baseline** (rule-based, from paper #3), placed next to `DetectorMaxPressure` in
   `controllers.py`.
   - Keep the current phase if `W1 * (moving vehicles within the effective range on its lanes)` is
     greater than the max pressure of the other phases; otherwise switch to the max-pressure phase.
   - Effective range = V_max × action duration, roughly 70–150 m. Sweep W1.
   - This shows our true gap.
2. **See approaching vehicles.** Our E2 detectors only cover the last 50 m, so the agent sees queues but
   not arriving platoons.
   - Add a count of *moving* vehicles over the effective range (an advance E1 loop or a longer E2) to
     the detector state.
   - Paper #3's ablation: the effective range matters, and whole-lane counts were *worse*.
3. **Time-based (SMDP) discounting.** `rl_agent.py` currently discounts per decision (γ = 0.95), and
   the reward is *averaged* over decision intervals of different lengths (10 s min-green, 5 s
   decisions, 5 s intergreen).
   - Sum the per-second reward over the interval, store `dt`, and use target `r + γ^dt · Q'`
     (re-tune γ per second).
   - Paper #5 calls this essential; without it, switching phases looks cheaper than it is.
4. **One shared, phase-wise network for all junctions** (FRAP/MPLight style).
   - Score each green stage from its movements' features, mask invalid stages, take the argmax.
   - One network trained on all 5 junctions' data gives 5× more samples and works for any orientation.
5. **Reward.** Keep `queue` (our best, and it matches the literature). Try adding a small pressure term
   (helps near saturation), or a travel-time-weighted throughput reward (paper #5's "equity factor").
6. **DQN extras:** n-step returns (3–5), dueling head, prioritised replay.
7. **Communication only on a corridor** (PLAN_REALISM 2E): directional messages and learned
   neighbour selection (CoSLight: a learned top-k beats fixed adjacency; its biggest win is on the
   arterial Avenue 4×4).

Retrain `queue` and `full_nocomm` (5 seeds) after steps 2–3 and compare against Advanced-MP on
held-out seeds with confidence intervals.
