import math

import numpy as np
import pytest

from cadac.constants import R
from cadac.env.us76 import atmosphere76
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.sraam6.environment import Sraam6Environment

RTOL = 1e-12
ATOL = 1e-14


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ready(*, hbe=5000.0, dvbe=250.0, mguid=0, mfreeze=0, trmach=0.5, trdynm=1e4, trcond=0):
    vehicle = _Vehicle()
    env = Sraam6Environment()
    env.define(vehicle)
    for name, value, ftype in (
        ("hbe", hbe, "real"),
        ("dvbe", dvbe, "real"),
        ("mguid", mguid, "int"),
        ("mfreeze", mfreeze, "int"),
        ("trmach", trmach, "real"),
        ("trdynm", trdynm, "real"),
        ("trcond", trcond, "int"),
    ):
        vehicle.store.define(Field(name, value, ftype, "data", "ext"))
    env.execute(vehicle, SimContext(0.0, 0.001, 0.0, 0.0, None, 0))
    return vehicle.store


def test_us76_mach_at_1v1_launch():
    store = _ready()
    rho, press, tempk = atmosphere76(5000.0)
    vsound = math.sqrt(1.4 * R * tempk)
    np.testing.assert_allclose(store.get("rho"), rho, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("vmach"), abs(250.0 / vsound), rtol=RTOL, atol=ATOL)


def test_mguid6_low_mach_sets_trcond_2():
    store = _ready(dvbe=1.0, mguid=6, trmach=0.5, trdynm=0)
    assert store.get("trcond") == 2


def test_mguid6_both_limits_trcond_3_wins():
    store = _ready(dvbe=1.0, mguid=6, trmach=0.5, trdynm=1e4)
    assert store.get("trcond") == 3
