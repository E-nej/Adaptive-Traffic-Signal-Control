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
  `act()` (or PyTorch) is a v0.3 item.
- No "legacy" 4-phase / grid config: v0.1.0 covers the paper setup.
- Paper sources may go in `paper/` (build outputs gitignored).
- Configs are TOML (`tomllib` to read, `tomli-w` to write the resolved config).
- SUMO pinned to 1.28.0: generated networks/routes depend on the SUMO version.
- Training seeds ≥ 7 are rejected (train seeds `100000·seed + ep` would collide with the
  validation / calibration / evaluation ranges 700000+ / 800000+ / 900000+).
- Parallel workers must run TensorFlow single-threaded (`TSC_TF_THREADS`, default 1).

## Steps

- [x] 0. Commit realism work on `realism`; fast-forward `main`, tag `v0.1.0` (local)
- [x] 1. Skeleton: `pyproject.toml` (uv, Python 3.12), `.gitignore`, `docs/`
- [ ] 2. Move realism modules into `src/tsc/` (`git mv`), split scenario → network / detectors /
        demand / webster, observation → observation / reward; SUMO bootstrap in `sumo_env.py`
- [ ] 3. Config (`config.py`) + seed ranges (`seeds.py`)
- [ ] 4. Experiments + CLI: `tsc train | evaluate | plot`, parallel over seeds, resolved config
        saved next to results, loader for v0.2-era run dirs (`config.json` + `*.weights.h5`)
- [ ] 5. Tests (pytest): signal intergreens, observation shapes, metrics fixture, config,
        SUMO smoke run per controller type
- [ ] 6. Parity checks (see below)
- [ ] 7. Delete `Deep-Qlearning-TSC/` (old code, tracked results), `images/`, `_config.yml`,
        `repopackpy-output.txt`, `OLD_README.md`
- [ ] 8. README + CHANGELOG (v0.2.0, fixes F1–F11 vs v0.1.0)
- [ ] 9. Ask before merging / tagging v0.2.0 / pushing

## Parity checks (behaviour must not change)

Reference data is in the main checkout under `Deep-Qlearning-TSC/runs/` (read only!).

- [ ] Rebuilt networks equal `runs/scenario/{static,actuated}.net.xml` (ignoring the
      generated-on header comment)
- [ ] Evaluation / calibration routes equal `runs/eval/routes/` (ignoring header comments)
- [ ] Full baseline sweep (240 runs) reproduces `runs/eval/baselines.csv` exactly. Note: that
      file predates loading the detector file for Python-controlled runs (max-pressure);
      detectors are passive, so it should still match.
- [ ] Observation / reward vectors identical to the realism code, step by step
- [ ] A v0.2-era trained run evaluated with old and new code gives identical metrics

## Notes for a later session
