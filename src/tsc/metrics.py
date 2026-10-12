"""Standard per-vehicle traffic metrics from SUMO output files.

Delay is SUMO's timeLoss (time lost compared to driving at the desired
speed) plus departDelay (time spent waiting to enter the network), i.e. the
usual "delay per vehicle" used in traffic engineering.  It is RESCO's
"average delay"; trip_time (mean duration of the trips that started) is the
"average trip / travel time" most RL papers report.
"""

import xml.etree.ElementTree as ET

import numpy as np


def parse_tripinfo(path):
    rows = []
    for _, el in ET.iterparse(path, events=('end',)):
        if el.tag != 'tripinfo':
            continue
        co2 = 0.0
        em = el.find('emissions')
        if em is not None:
            co2 = float(em.get('CO2_abs', 0.0))
        rows.append({
            'arrived': float(el.get('arrival', -1)) >= 0,
            'departed': float(el.get('depart', 0)) >= 0,
            'duration': float(el.get('duration')),
            'time_loss': float(el.get('timeLoss')),
            'depart_delay': float(el.get('departDelay', 0.0)),
            'waiting_time': float(el.get('waitingTime')),
            'stops': float(el.get('waitingCount')),
            'co2_mg': co2,
        })
        el.clear()
    return rows


def parse_statistics(path):
    root = ET.parse(path).getroot()
    veh = root.find('vehicles')
    tele = root.find('teleports')
    return {
        'loaded': int(veh.get('loaded')),
        'inserted': int(veh.get('inserted')),
        'running_end': int(veh.get('running')),
        'waiting_insert_end': int(veh.get('waiting')),
        'teleports': int(tele.get('total')) if tele is not None else 0,
    }


def summarise(tripinfo_path, statistics_path, mean_queue=None):
    rows = parse_tripinfo(tripinfo_path)
    stats = parse_statistics(statistics_path)
    if not rows:
        raise RuntimeError(f'No tripinfo entries in {tripinfo_path}')

    def mean(key, only_arrived=False):
        vals = [r[key] for r in rows if r['arrived'] or not only_arrived]
        return float(np.mean(vals)) if vals else float('nan')

    started = [r['duration'] for r in rows if r['departed']]

    arrived = sum(r['arrived'] for r in rows)
    out = {
        **stats,
        'arrived': arrived,
        'throughput': arrived / max(1, stats['loaded']),
        'delay': float(np.mean([r['time_loss'] + r['depart_delay'] for r in rows])),
        'waiting_time': mean('waiting_time'),
        'stops': mean('stops'),
        'travel_time_arrived': mean('duration', only_arrived=True),
        'trip_time': float(np.mean(started)) if started else float('nan'),
        'co2_g_per_veh': mean('co2_mg') / 1000.0,
        'co2_kg_total': sum(r['co2_mg'] for r in rows) / 1e6,
    }
    if mean_queue is not None:
        out['queue'] = mean_queue
    return out
