"""SRAAM5 G2 environment — ISO/air density vs altitude (Fortran MODULE.FOR G2)."""

import math

import numpy as np

from cadac.constants import R
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat5.sraam5.environment import Sraam5Environment

RTOL = 1e-12
ATOL = 0.0

# Planted troposphere sample (below 11 km tropopause), Fortran G2 equations.
HBE = 5000.0
DVBE = 250.0


def _fortran_g2_rho(hbe: float) -> float:
    """Air density from SRAAM5 MODULE.FOR G2 ISO atmosphere."""
    if hbe < 11000.0:
        tempk = 288.15 - 0.0065 * hbe
        press = 101325.0 * (tempk / 288.15) ** 5.2559
    else:
        tempk = 216.0
        press = 22630.0 * math.exp(-0.00015769 * (hbe - 11000.0))
    return press / (R * tempk)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ready(*, hbe=HBE, dvbe=DVBE, mguid=0, mfreeze=0, trmach=0.5, trdynm=1e4, trcode=0.0):
    vehicle = _Vehicle()
    env = Sraam5Environment()
    env.define(vehicle)
    for name, value, ftype in (
        ("hbe", hbe, "real"),
        ("dvbe", dvbe, "real"),
        ("mguid", mguid, "int"),
        ("mfreeze", mfreeze, "int"),
        ("trmach", trmach, "real"),
        ("trdynm", trdynm, "real"),
        ("trcode", trcode, "real"),
    ):
        vehicle.store.define(Field(name, value, ftype, "data", "ext"))
    env.execute(vehicle, SimContext(0.0, 0.001, 0.0, 0.0, None, 0))
    return vehicle.store


def test_name_is_environment():
    assert Sraam5Environment().name == "environment"


def test_iso_air_density_at_planted_altitude():
    """Fortran G2: RHO = PRESS/(R*TEMPK) at HBE=5000 m (troposphere branch)."""
    store = _ready()
    expected_rho = _fortran_g2_rho(HBE)
    np.testing.assert_allclose(store.get("rho"), expected_rho, rtol=RTOL, atol=ATOL)
