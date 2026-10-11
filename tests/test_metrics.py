import os

import pytest
from conftest import FIXTURES

from tsc import metrics


def test_summarise_fixture():
    m = metrics.summarise(os.path.join(FIXTURES, 'tripinfo.xml'),
                          os.path.join(FIXTURES, 'statistics.xml'), mean_queue=1.5)
    assert m['loaded'] == 4 and m['inserted'] == 3 and m['running_end'] == 1
    assert m['waiting_insert_end'] == 1 and m['teleports'] == 2
    assert m['arrived'] == 2
    assert m['throughput'] == pytest.approx(2 / 4)
    # delay = timeLoss + departDelay over all vehicles, unfinished ones included
    assert m['delay'] == pytest.approx((11 + 20 + 9) / 3)
    assert m['waiting_time'] == pytest.approx(12 / 3)
    assert m['stops'] == pytest.approx(1.0)
    assert m['travel_time_arrived'] == pytest.approx(45.0)
    assert m['co2_g_per_veh'] == pytest.approx(120.0)
    assert m['co2_kg_total'] == pytest.approx(0.36)
    assert m['queue'] == 1.5
