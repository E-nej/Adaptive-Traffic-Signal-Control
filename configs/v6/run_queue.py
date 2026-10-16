"""Run a job file with dependencies and priorities on a fixed number of CPU slots.

    python configs/v6/run_queue.py outputs/v6/jobs.txt --slots 12

Job file, one job per line (blank lines and '#' comments ignored; '#END' marks the file
complete, so jobs can be appended while the queue runs):

    label | priority | slots | dep1,dep2 | command

A job starts when every dependency has finished (successfully or not; failures are
logged) and enough slots are free.  Lower priority numbers start first; if the first
ready job does not fit, nothing behind it starts (so multi-slot jobs are not starved).
The slot count is re-read from <jobs dir>/slots.txt every poll if that file exists.
Logs: <jobs dir>/logs/<label>.log; status lines in <jobs dir>/queue.log and on stdout.
"""

import argparse
import datetime
import os
import subprocess
import time


def parse(path):
    jobs, complete = [], False
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '#END':
                complete = True
            if not line or line.startswith('#'):
                continue
            label, prio, slots, deps, cmd = [x.strip() for x in line.split('|', 4)]
            jobs.append({'label': label, 'prio': int(prio), 'slots': int(slots),
                         'deps': [d for d in deps.split(',') if d.strip()], 'cmd': cmd})
    return jobs, complete


def main():
    p = argparse.ArgumentParser()
    p.add_argument('jobs')
    p.add_argument('--slots', type=int, default=12)
    p.add_argument('--poll', type=float, default=15.0)
    args = p.parse_args()
    root = os.path.dirname(os.path.abspath(args.jobs))
    os.makedirs(os.path.join(root, 'logs'), exist_ok=True)
    status_path = os.path.join(root, 'queue.log')
    slots_file = os.path.join(root, 'slots.txt')
    env = dict(os.environ, TSC_THREADS='1', PYTHONUNBUFFERED='1')

    def log(msg):
        line = f'{datetime.datetime.now():%m-%d %H:%M:%S} {msg}'
        print(line, flush=True)
        with open(status_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    running, finished, failed = {}, {}, set()
    log(f'queue started: {args.jobs}')
    while True:
        jobs, complete = parse(args.jobs)
        labels = {j['label'] for j in jobs}
        slots = args.slots
        if os.path.exists(slots_file):
            try:
                slots = int(open(slots_file).read().strip())
            except ValueError:
                pass
        for label, (proc, job, t0, fh) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            hours = (time.time() - t0) / 3600
            finished[label] = rc
            del running[label]
            if rc:
                failed.add(label)
                log(f'FAIL {label} rc={rc} after {hours:.2f} h (see logs/{label}.log)')
            else:
                log(f'DONE {label} in {hours:.2f} h')
        used = sum(job['slots'] for _, job, _, _ in running.values())
        ready = [j for j in jobs if j['label'] not in finished and j['label'] not in running
                 and all(d in finished or d not in labels for d in j['deps'])]
        ready.sort(key=lambda j: j['prio'])          # stable: file order within a priority
        for job in ready:
            if used + job['slots'] > slots:
                break
            fh = open(os.path.join(root, 'logs', f"{job['label']}.log"), 'w', encoding='utf-8')
            proc = subprocess.Popen(job['cmd'], shell=True, stdout=fh, stderr=subprocess.STDOUT,
                                    env=env)
            running[job['label']] = (proc, job, time.time(), fh)
            used += job['slots']
            log(f"START {job['label']} (prio {job['prio']}, {job['slots']} slot(s), "
                f"{used}/{slots} used)")
        pending = [j for j in jobs if j['label'] not in finished and j['label'] not in running]
        if complete and not pending and not running:
            log(f'queue finished: {len(finished)} jobs, {len(failed)} failed '
                f'{sorted(failed) if failed else ""}')
            return
        time.sleep(args.poll)


if __name__ == '__main__':
    main()
