"""ROCKET3 G2 environment — ISO atmosphere at launch altitude (Fortran MODULE.FOR G2)."""

import math

import numpy as np

from cadac.constants import R
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round3.rocket3.environment import Rocket3Environment

RTOL = 1e-12
ATOL = 0.0

# INLAUNCH.ASC launch conditions (BALT / DVBE).
BALT = 1.0
DVBE = 1.0


def _fortran_g2_rho(balt: float) -> float:
    """Air density from ROCKET3 MODULE.FOR G2 ISO atmosphere (MAIR=0)."""
    if balt < 11000.0:
        tempk = 288.15 - 0.0065 * balt
        press = 101325.0 * (tempk / 288.15) ** 5.2559
    else:
        tempk = 216.0
        press = 22630.0 * math.exp(-0.00015769 * (balt - 11000.0))
    return press / (R * tempk)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ready(*, balt=BALT, dvbe=DVBE, mair=0):
    vehicle = _Vehicle()
    env = Rocket3Environment()
    env.define(vehicle)
    for name, value, ftype in (
        ("balt", balt, "real"),
        ("dvbe", dvbe, "real"),
        ("mair", mair, "int"),
    ):
        if name not in vehicle.store:
            vehicle.store.define(Field(name, value, ftype, "data", "ext"))
        else:
            vehicle.store.set(name, value)
    env.execute(vehicle, SimContext(0.0, 0.001, 0.0, 0.0, None, 0))
    return vehicle.store


def test_name_is_environment():
    assert Rocket3Environment().name == "environment"


def test_iso_air_density_at_launch_altitude():
    """Fortran G2: RHO = PRESS/(R*TEMPK) at BALT=1 m (INLAUNCH launch alt)."""
    store = _ready()
    expected_rho = _fortran_g2_rho(BALT)
    np.testing.assert_allclose(store.get("rho"), expected_rho, rtol=RTOL, atol=ATOL)
