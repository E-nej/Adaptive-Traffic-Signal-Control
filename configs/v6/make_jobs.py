"""Write the step-2I job file for run_queue.py (run from the repository root).

    python configs/v6/make_jobs.py            # -> outputs/v6/jobs.txt
    python configs/v6/run_queue.py outputs/v6/jobs.txt --slots 12

Priorities: 0 = tables and evaluations of the hierarchy (start as soon as their runs exist),
1 = hierarchy training (workers, RL experts; longest first), 2 = benchmarks rl2 noise-free
and their evaluations, 3 = benchmarks with the phase-wise network.
"""

import os

TSC = r'.venv\Scripts\tsc.exe'
SEEDS = [0, 1, 2]
WORKERS = {'maribor': ['maxpressure', 'actuated'],
           'corridor': ['greenwave', 'coord_actuated', 'maxpressure']}
BENCH = ['ingolstadt21', 'arterial4x4', 'grid4x4', 'cologne8', 'ingolstadt7', 'cologne3',
         'cologne1', 'ingolstadt1']          # longest first
COMPARE_REF = {'corridor': 'greenwave_py', 'corridor_opposite': 'greenwave_py',
               'maribor': 'maxpressure_det'}


def main():
    lines = []

    def job(label, prio, slots, deps, cmd):
        lines.append(f"{label} | {prio} | {slots} | {','.join(deps)} | {cmd}")

    # hierarchy training (Maribor first: its runs are the longest)
    for net in ('maribor', 'corridor'):
        for s in SEEDS:
            for e in WORKERS[net]:
                job(f'wk_{net}_{e}_s{s}', 1, 1, [],
                    f'{TSC} train configs/v6/workers/{net}_{e}.toml --seeds {s} --workers 1')
            for k in ('rl', 'rlf'):
                job(f'{k}_{net}_s{s}', 1, 1, [],
                    f'{TSC} train configs/v6/rl/{net}_{k}.toml --seeds {s} --workers 1')
    # tables (one per seed, with that seed's workers and RL experts) and evaluations
    for net in ('maribor', 'corridor'):
        tables = []
        for s in SEEDS:
            deps = [f'wk_{net}_{e}_s{s}' for e in WORKERS[net]] + [f'{k}_{net}_s{s}' for k in ('rl', 'rlf')]
            job(f'table_{net}_s{s}', 0, 4, deps,
                f'{TSC} train configs/v6/tables/{net}_table.toml --seeds {s} --workers 4')
            tables.append(f'table_{net}_s{s}')
        evals = ['', '_opposite'] if net == 'corridor' else ['']
        for ev in evals:
            job(f'eval_{net}{ev}', 0, 8, tables,
                f'{TSC} evaluate configs/v6/eval/{net}{ev}.toml --workers 8')
            job(f'compare_{net}{ev}', 0, 1, [f'eval_{net}{ev}'],
                f'{TSC} compare outputs/eval/v6_{net}{ev}/results.csv --ref {COMPARE_REF[net + ev]} '
                f'--out outputs/v6/compare_{net}{ev}.md')
    # published benchmarks: baselines, rl2 without noise, phase-wise network
    for name in BENCH:
        job(f'bench_base_{name}', 2, 4, [],
            f'{TSC} evaluate configs/benchmarks/{name}.toml --workers 4 '
            f'--out outputs/eval/v6_bench_{name}')
        runs = []
        for kind, prio in (('rl2nf', 2), ('rl2f', 3)):
            for s in SEEDS:
                job(f'{kind}_{name}_s{s}', prio, 1, [],
                    f'{TSC} train configs/v6/bench/{kind}_{name}.toml --seeds {s} --workers 1')
                runs.append(f'{kind}_{name}_s{s}')
        job(f'bench_rl_{name}', 2, 4, runs,
            f'{TSC} evaluate configs/benchmarks/{name}.toml --runs "outputs/train/rl2nf_{name}/s*" '
            f'"outputs/train/rl2f_{name}/s*" --no-baselines --set signals.max_green=60 --workers 4 '
            f'--out outputs/eval/v6_bench_{name}_rl')
        job(f'compare_bench_{name}', 2, 1, [f'bench_base_{name}', f'bench_rl_{name}'],
            f'{TSC} compare outputs/eval/v6_bench_{name}/results.csv '
            f'outputs/eval/v6_bench_{name}_rl/results.csv --ref maxpressure_det '
            f'--out outputs/v6/compare_bench_{name}.md')
    lines.append('#END')
    os.makedirs('outputs/v6', exist_ok=True)
    with open('outputs/v6/jobs.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'{len(lines) - 1} jobs -> outputs/v6/jobs.txt')


if __name__ == '__main__':
    main()
