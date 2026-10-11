"""Command line: tsc train | evaluate | plot.

    tsc train configs/step1/full.toml --seeds 0-4 --workers 5
    tsc evaluate configs/baselines.toml
    tsc evaluate configs/step1/evaluate.toml --runs "outputs/train/full/s*"
    tsc plot outputs/eval/baselines/results.csv outputs/eval/step1/results.csv --out outputs/plots

Any config value can be overridden with --set section.key=value (TOML syntax),
e.g. --set observation.noise=0.1 --set name=full_noise
"""

import argparse
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
    for run_dir in train(cfg, seeds=seeds, out_root=args.out):
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
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())
