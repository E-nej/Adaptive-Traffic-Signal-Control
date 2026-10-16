# Refactor to v0.2.0: checklist

Branch: `refactor-v0.2` (worktree `..\PSIS-seminarska-refactor`), created from `realism`
at `d468260` (the realism code, steps 1–2 of [PLAN_REALISM.md](PLAN_REALISM.md)).
Goal: one clean, config-driven, network-agnostic package (`src/tsc/`) that replaces both the
paper code and the `Deep-Qlearning-TSC/` scripts, with **unchanged behaviour**.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` dropped

## Decisions (agreed 2026-10-11)

- v0.1.0 = `3b729fa` (head of `origin/multiagent`, the paper code). `main` fast-forwarded to it,
  tag created **locally** — not pushed yet.
- Keep TensorFlow for v0.2.0 (the running step 1/2 runs use it). A numpy forward pass in
  `act()` (or PyTorch) is a v0.3 item. *Done in 0.3.0: the agents use PyTorch (CHANGELOG).*
- No "legacy" 4-phase / grid config: v0.1.0 covers the paper setup.
- Paper sources may go in `paper/` (build outputs gitignored).
- Configs are TOML (`tomllib` to read, `tomli-w` to write the resolved config).
- SUMO pinned to 1.28.0: generated networks/routes depend on the SUMO version.
- Training seeds ≥ 7 are rejected (train seeds `100000·seed + ep` would collide with the
  validation / calibration / evaluation ranges 700000+ / 800000+ / 900000+).
- Parallel workers must run TensorFlow single-threaded (`TSC_TF_THREADS`, default 1).

## Steps

- [x] 0. Commit realism work on `realism` (`d468260`); fast-forward `main`, tag `v0.1.0` (local)
- [x] 1. Skeleton: `pyproject.toml` (uv, Python 3.12), `.gitignore`, `docs/`
- [x] 2. Move realism modules into `src/tsc/` (`git mv`), split scenario → network / detectors /
        demand / webster, observation → observation / reward; SUMO bootstrap in `sumo_env.py`
- [x] 3. Config (`config.py`) + seed ranges (`seeds.py`)
- [x] 4. Experiments + CLI: `tsc train | evaluate | plot`, parallel over seeds, resolved config
        saved next to results, loader for v0.2-era run dirs (`config.json` + `*.weights.h5`)
- [x] 5. Tests (pytest): signal intergreens, observation shapes, metrics fixture, config,
        SUMO smoke run per controller type, tiny train → evaluate → plot
- [x] 6. Parity checks (see below)
- [x] 7. Delete `Deep-Qlearning-TSC/` (old code, tracked results), `images/`, `_config.yml`,
        `repopackpy-output.txt`, `OLD_README.md`
- [x] 8. README + CHANGELOG (v0.2.0, fixes F1–F11 vs v0.1.0)
- [ ] 9. Review with the user; then (only on their OK) merge into `main`, set the v0.2.0 date in
        CHANGELOG, tag `v0.2.0`, push `main` + tags
- [ ] 10. Once the running trainings are finished: evaluate them with
        `tsc evaluate configs/step1/evaluate.toml --runs ".../Deep-Qlearning-TSC/runs/train/*"`
        (and step 2 with `runs/train2/*`, `--set evaluate.tag=...` if names collide)

## Parity checks (behaviour must not change)

Reference data is in the main checkout under `Deep-Qlearning-TSC/runs/` (read only!).
All comparisons ignore XML comments (netconvert / randomTrips write a timestamped header).

- [x] Rebuilt networks equal `runs/scenario/{static,actuated}.net.xml`
- [x] All 162 evaluation / calibration route files and Webster plans equal `runs/eval/routes/`
- [x] Full baseline sweep (240 runs, `tsc evaluate configs/baselines.toml`) reproduces every
      metric of `runs/eval/baselines.csv` exactly (max |diff| = 0), max-pressure included
- [x] Observation / reward vectors identical to the realism code, step by step (grid ± comm,
      detector with noise + dropout, all 4 rewards, with the shield active)
- [x] Trained runs (`train/base_s0`, `train/full_s0`, `train2/det_queue_s0`) evaluated with old
      and new code: identical metrics
- [x] Training: 3-episode detector/noise/shield run with seed 3 gives a `train_log.csv`
      identical to the realism `train.py` (needed TF to be imported before seeding)

## Notes for a later session

- Paths containing `--` break SUMO: randomTrips/netconvert write the paths into an XML
  comment, and `--` is illegal inside XML comments. Keep `output_dir` free of `--`.
- Run names: legacy step-1 and step-2 runs both use config names like `full`; evaluate them
  in separate `tsc evaluate` calls (they write separate CSVs) or set `evaluate.tag`.
- `network.rebuild_tls = false` uses a network's own signal plans (step 3); the actuated
  baseline then is not available yet.
- v0.3 ideas: numpy forward pass in `DQNAgent.act()` (most per-step TF overhead), widen the
  seed ranges, demand generator from counts (routeSampler) for the Maribor corridor.
