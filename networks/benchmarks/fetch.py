"""Download the RESCO benchmark scenarios (Ault & Sharon, NeurIPS 2021) used for the
literature comparison.

    uv run python networks/benchmarks/fetch.py            # all scenarios
    uv run python networks/benchmarks/fetch.py cologne1   # some of them

Source: github.com/Pi-Star-Lab/RESCO at a fixed commit, folder resco_benchmark/environments.
The files are third-party data (RESCO: GPL-3.0; the Cologne and Ingolstadt maps come from
TAPAS Cologne and InTAS with their own licence files, downloaded alongside) and are not
committed: they land in networks/benchmarks/<scenario>/ (git-ignored).  grid4x4 and
arterial4x4 ship their route files zipped; the zip is extracted into the same folder.

Scenarios: cologne1/3/8, ingolstadt1/7/21 (real maps, calibrated demand) and grid4x4 /
arterial4x4 (synthetic, also used by CoSLight as Grid 4x4 / Avenue 4x4).
"""

import json
import os
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = 'Pi-Star-Lab/RESCO'
COMMIT = 'f1ed9a174f8de41fc9d8689373b836bc882570dc'     # main, 2025-10-28
ROOT = 'resco_benchmark/environments'
SCENARIOS = ('cologne1', 'cologne3', 'cologne8', 'ingolstadt1', 'ingolstadt7', 'ingolstadt21',
             'grid4x4', 'arterial4x4')


def _get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def files_of(scenario):
    url = f'https://api.github.com/repos/{REPO}/contents/{ROOT}/{scenario}?ref={COMMIT}'
    return [e['name'] for e in json.loads(_get(url)) if e['type'] == 'file']


def fetch(scenario):
    if scenario not in SCENARIOS:
        raise SystemExit(f'Unknown scenario {scenario!r}; choose from {SCENARIOS}')
    out = os.path.join(HERE, scenario)
    os.makedirs(out, exist_ok=True)
    for name in files_of(scenario):
        path = os.path.join(out, name)
        if not os.path.exists(path):
            raw = f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{ROOT}/{scenario}/{name}'
            with open(path, 'wb') as f:
                f.write(_get(raw))
        if name.endswith('.zip'):
            with zipfile.ZipFile(path) as z:
                for member in z.namelist():         # no paths outside the folder
                    target = os.path.realpath(os.path.join(out, member))
                    if not target.startswith(os.path.realpath(out) + os.sep):
                        raise SystemExit(f'{name}: unsafe path {member!r}')
                z.extractall(out)
    names = os.listdir(out)
    routes = [n for n in names if n.endswith('.rou.xml')]
    print(f'{scenario}: {len(names)} files, {len(routes)} route file(s)')


def main(argv):
    for scenario in argv or SCENARIOS:
        fetch(scenario)


if __name__ == '__main__':
    main(sys.argv[1:])
