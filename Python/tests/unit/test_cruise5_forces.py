from math import cos, sin

import numpy as np
import pytest

from cadac.constants import RAD
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.cruise5.forces import Cruise5Forces

RTOL = 1e-12
ATOL = 1e-14
PDYNMC = 5000.0
AREA = 0.929
CD = 0.05
CL = 0.2
THRUST = 1500.0
MASS = 1000.0
ALPHAX = 0.0
PHIMVX = 0.0


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ready(phimvx=PHIMVX):
    vehicle = _Vehicle()
    forces = Cruise5Forces()
    forces.define(vehicle)
    store = vehicle.store
    for name, value in (
        ("pdynmc", PDYNMC), ("area", AREA), ("cd", CD), ("cl", CL),
        ("thrust", THRUST), ("mass", MASS), ("alphax", ALPHAX),
        ("phimvx", phimvx),
    ):
        if name not in store.names():
            store.define(Field(name, value, "real", "data", "test"))
        else:
            store.set(name, value)
    return vehicle, forces


def test_name_is_forces():
    assert Cruise5Forces().name == "forces"


def test_fspv_phimvx_0_and_90_match_cpp():
    alpha = ALPHAX * RAD
    for phimvx in (0.0, 90.0):
        phimv = phimvx * RAD
        fspv1 = (-PDYNMC * AREA * CD + THRUST * cos(alpha)) / MASS
        fspv2 = sin(phimv) * (PDYNMC * AREA * CL + THRUST * sin(alpha)) / MASS
        fspv3 = -cos(phimv) * (PDYNMC * AREA * CL + THRUST * sin(alpha)) / MASS
        vehicle, forces = _ready(phimvx)
        forces.execute(vehicle, None)
        np.testing.assert_allclose(
            vehicle.store.get("FSPV"), [fspv1, fspv2, fspv3], rtol=RTOL, atol=ATOL
        )


def test_define_skips_existing_fspv():
    vehicle = _Vehicle()
    vehicle.store.define(Field("FSPV", (1.0, 2.0, 3.0), "vec", "out", "newton"))
    Cruise5Forces().define(vehicle)
    np.testing.assert_allclose(vehicle.store.get("FSPV"), [1.0, 2.0, 3.0])
