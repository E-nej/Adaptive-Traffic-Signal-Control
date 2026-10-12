import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, 'tests', 'fixtures')
TEST5 = os.path.join(ROOT, 'networks', 'test5', 'test.net.xml')
SINGLE_4L = os.path.join(ROOT, 'networks', 'single_4l', 'intersection.4L.net.xml')


@pytest.fixture(scope='session')
def work_root(tmp_path_factory):
    return str(tmp_path_factory.mktemp('tsc'))


def make_network(path, work_root):
    from tsc.config import NetworkConfig
    from tsc.network import Network
    return Network(NetworkConfig(path=path), os.path.join(work_root, 'networks'))


@pytest.fixture(scope='session')
def test5(work_root):
    return make_network(TEST5, work_root)


@pytest.fixture(scope='session')
def single_4l(work_root):
    return make_network(SINGLE_4L, work_root)
