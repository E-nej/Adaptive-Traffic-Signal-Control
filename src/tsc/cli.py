"""Command line: tsc train | evaluate | plot | compare | diagnose | assign | accept.

    tsc train configs/step1/full.toml --seeds 0-4 --workers 5
    tsc evaluate configs/baselines.toml
    tsc evaluate configs/step1/evaluate.toml --runs "outputs/train/full/s*"
    tsc plot outputs/eval/baselines/results.csv outputs/eval/step1/results.csv --out outputs/plots
    tsc compare outputs/eval/baselines/results.csv outputs/eval/step1/results.csv --ref maxpressure
    tsc diagnose messages configs/step2/evaluate.toml --runs "outputs/train/step2_full/s*"
    tsc accept "outputs/train/hier2_corridor_network/s*" --scenarios 6
    tsc parity record configs/step1/full.toml      (then, after a change: tsc parity check ...)
    tsc devices                                     (what PyTorch sees; GPU support verdicts)
    tsc bench configs/step1/full.toml --mix cpu:5 --mix gpu:2,cpu:4

Any config value can be overridden with --set section.key=value (TOML syntax),
e.g. --set observation.noise=0.1 --set name=full_noise
"""

import argparse
import os
import sys

from tsc import config as config_mod


def parse_seeds(text):
    """'0-4' -> [0..4], '0,2,5' -> [0, 2, 5]."""
    seeds = []
    for part in text.split(','):
        lo, sep, hi = part.partition('-')
        seeds.extend(range(int(lo), int(hi) + 1) if sep else [int(lo)])
    return seeds


def _load(args):
    cfg = config_mod.load(args.config, args.set)
    if args.workers is not None:
        cfg = config_mod.replace(cfg, workers=args.workers)
    return cfg


def cmd_train(args):
    from tsc.experiments.train import train
    cfg = _load(args)
    seeds = parse_seeds(args.seeds) if args.seeds else None
    for run_dir in train(cfg, seeds=seeds, out_root=args.out, gpu_workers=args.gpu_workers):
        print(f'finished {run_dir}')


def cmd_evaluate(args):
    from tsc.experiments.evaluate import evaluate
    cfg = _load(args)
    if args.runs is not None:
        cfg = config_mod.replace(cfg, evaluate={'runs': args.runs})
    if args.no_baselines:
        cfg = config_mod.replace(cfg, evaluate={'baselines': []})
    evaluate(cfg, out_dir=args.out)


def cmd_plot(args):
    from tsc.experiments.plot import plot
    plot(args.csv, args.out)


def cmd_compare(args):
    import pandas as pd
    from tsc.experiments.stats import paired_table
    df = pd.concat([pd.read_csv(f) for f in args.csv], ignore_index=True)
    table = paired_table(df, args.ref, metric=args.metric)
    print(f'{args.metric} relative to {args.ref} (paired over scenarios, * = Wilcoxon p < 0.05)')
    print(table.to_string())
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(table.to_markdown())


def cmd_accept(args):
    import glob
    from tsc.experiments.train import accept_runs
    runs = sorted(d for pattern in args.runs for d in glob.glob(pattern))
    if not runs:
        raise SystemExit(f'No run directories match {args.runs}')
    for run_dir, r in accept_runs(runs, args.scenarios, args.workers):
        detail = (f" ({r['change']:+.1%} vs {r['default']} alone, p={r['p']:.3f})" if 'p' in r
                  else f" ({r['reason']})")
        print(f"{run_dir}: deployed {r['deployed']}{detail}")


def cmd_assign(args):
    from tsc.experiments.assign import assign
    assign(_load(args), out_dir=args.out, passes=args.passes)


def cmd_diagnose(args):
    from tsc.experiments import diagnose
    cfg = _load(args)
    if args.runs is not None:
        cfg = config_mod.replace(cfg, evaluate={'runs': args.runs})
    getattr(diagnose, args.what)(cfg, out_dir=args.out)


TORCH_PIN = '2.14.'          # the PyTorch minor version pyproject.toml pins


def cmd_devices(args):
    import json

    from tsc import devices
    from tsc.experiments.common import physical_cores, software_stack
    info = devices.describe()
    stack = software_stack()
    print(f"PyTorch {info['torch']} ({info['backend']} build) {json.dumps(info['build'])}")
    if not info['torch'].startswith(TORCH_PIN):
        print(f"  WARNING: PyTorch {info['torch']} differs from the pinned {TORCH_PIN}*; "
              f"recorded in run_info.json of every run")
    if info['hidden_by_env']:
        print(f"  GPUs hidden by environment: {info['hidden_by_env']}")
    print(f"CPU: {stack['cpu']}, {physical_cores()} physical cores, {os.cpu_count()} threads, "
          f"SIMD {' '.join(stack['cpu_flags'])}")
    print(f"  suggested CPU-only --workers: {max(1, physical_cores())}")
    if not info['gpus']:
        print('GPUs: none visible to PyTorch (see docs/GPU.md)')
    for g in info['gpus']:
        print(f"GPU #{g['index']}: {g['name']} [{g['arch'] or 'arch ?'}] -> {g['status'].upper()}: "
              f"{g['reason']}")
    if info['backend'] == 'rocm' and len(info['gpus']) > 1:
        print('  note: several ROCm devices; if one is an integrated GPU, hide it with '
              'HIP_VISIBLE_DEVICES (docs/GPU.md)')


def cmd_bench(args):
    from tsc.experiments.bench import bench, parse_mix
    cfg = _load(args)
    bench(cfg, [parse_mix(m) for m in args.mix], episodes=args.episodes, repeats=args.repeats,
          out_dir=args.out)


def _parity_config(args):
    from tsc.experiments import parity
    cfg = parity.short_config(_load(args), args.episodes, args.val_every, args.seed)
    return cfg, args.out or parity.parity_root(cfg)


def cmd_parity_record(args):
    from tsc.experiments import parity
    cfg, root = _parity_config(args)
    out = os.path.join(root, args.label)
    parity.record(cfg, out)
    print(f'recorded {os.path.join(out, parity.MANIFEST)}')


def cmd_parity_check(args):
    from tsc.experiments import parity
    cfg, root = _parity_config(args)
    golden = parity.load(os.path.join(root, args.against))
    out = os.path.join(root, args.label)
    current = parity.record(cfg, out)
    print(parity.report(parity.compare(golden, current), args.against, args.label))
    return 0 if not parity.compare(golden, current) else 1


def cmd_parity_compare(args):
    from tsc.experiments import parity
    diffs = parity.compare(parity.load(args.a), parity.load(args.b))
    print(parity.report(diffs, 'a', 'b'))
    return 0 if not diffs else 1


def build_parser():
    p = argparse.ArgumentParser(prog='tsc', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)

    def with_config(sp):
        sp.add_argument('config', help='experiment config (TOML)')
        sp.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='override a config value, e.g. observation.noise=0.05')
        sp.add_argument('--workers', type=int, help='parallel processes (default: config)')
        sp.add_argument('--out', help='output directory (default: <output_dir>/<kind>/<name>)')

    t = sub.add_parser('train', help='train RL controllers, one run per seed')
    with_config(t)
    t.add_argument('--seeds', help="training seeds, e.g. '0-4' or '0,2' (default: config)")
    t.add_argument('--gpu-workers', type=int, default=0, metavar='K',
                   help='hybrid scheduling: the first K seeds train on the GPU, the rest on the CPU')
    t.set_defaults(func=cmd_train)

    e = sub.add_parser('evaluate', help='evaluate baselines and trained runs over a demand sweep')
    with_config(e)
    e.add_argument('--runs', nargs='*', help='trained run directories (globs ok); replaces the config')
    e.add_argument('--no-baselines', action='store_true', help='only evaluate the trained runs')
    e.set_defaults(func=cmd_evaluate)

    pl = sub.add_parser('plot', help='plots and summary table from evaluation CSVs')
    pl.add_argument('csv', nargs='+')
    pl.add_argument('--out', default='outputs/plots')
    pl.set_defaults(func=cmd_plot)

    c = sub.add_parser('compare', help='paired comparison against a reference controller')
    c.add_argument('csv', nargs='+')
    c.add_argument('--ref', required=True, help="reference controller, e.g. 'maxpressure'")
    c.add_argument('--metric', default='delay')
    c.add_argument('--out', help='also write the table as Markdown')
    c.set_defaults(func=cmd_compare)

    ac = sub.add_parser('accept', help='hierarchy: keep the learned manager only if it beats '
                                       'its default expert on fresh scenarios')
    ac.add_argument('runs', nargs='+', help='trained run directories (globs ok)')
    ac.add_argument('--scenarios', type=int, help='per validation level (default: config, else 4)')
    ac.add_argument('--workers', type=int, default=1, help='parallel processes')
    ac.set_defaults(func=cmd_accept)

    a = sub.add_parser('assign',help='best fixed controller per junction and demand level')
    with_config(a)
    a.add_argument('--passes', type=int, default=2, help='greedy search rounds')
    a.set_defaults(func=cmd_assign)

    d = sub.add_parser('diagnose', help='diagnostics of trained policies')
    d.add_argument('what', choices=['messages'], help='messages: real vs zero vs wrong-neighbour')
    with_config(d)
    d.add_argument('--runs', nargs='*', help='trained run directories (globs ok)')
    d.set_defaults(func=cmd_diagnose)

    dv = sub.add_parser('devices', help='CPU / GPUs PyTorch can use, with support verdicts')
    dv.set_defaults(func=cmd_devices)

    b = sub.add_parser('bench', help='training throughput (episodes/min) under CPU/GPU worker mixes')
    with_config(b)
    b.add_argument('--mix', action='append', required=True,
                   help="concurrent runs per device, e.g. 'cpu:5' or 'gpu:2,cpu:4' (repeatable)")
    b.add_argument('--episodes', type=int, default=10, help='episodes per run (default 10)')
    b.add_argument('--repeats', type=int, default=3, help='repetitions per mix (default 3)')
    b.set_defaults(func=cmd_bench)

    pa = sub.add_parser('parity', help='record / check bit-exact fingerprints of a short training run')
    pa_sub = pa.add_subparsers(dest='action', required=True)

    def with_parity(sp, label):
        with_config(sp)
        sp.add_argument('--label', default=label, help=f'name of this recording (default: {label})')
        sp.add_argument('--episodes', type=int, default=6)
        sp.add_argument('--val-every', type=int, default=2)
        sp.add_argument('--seed', type=int, default=0)

    pr = pa_sub.add_parser('record', help='train a short run and record its manifest')
    with_parity(pr, 'golden')
    pr.set_defaults(func=cmd_parity_record)
    pc = pa_sub.add_parser('check', help='record again and compare with an earlier recording')
    with_parity(pc, 'check')
    pc.add_argument('--against', default='golden', help='recording to compare with (default: golden)')
    pc.set_defaults(func=cmd_parity_check)
    pm = pa_sub.add_parser('compare', help='compare two recordings (directories or manifest files)')
    pm.add_argument('a')
    pm.add_argument('b')
    pm.set_defaults(func=cmd_parity_compare)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
