from math import cos, sin

import numpy as np
import pytest

from cadac.constants import RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.hyper5.forces import Hyper5Forces

RTOL = 1e-12
ATOL = 1e-14

PDYNMC = 72000.0
AREA = 11.6986
CD = 0.05
CL = 0.2
THRUST = 0.0
MASS = 1352.0
ALPHAX = -1.5

EXTERNALS = (
    "pdynmc",
    "cl",
    "cd",
    "area",
    "thrust",
    "mass",
    "alphax",
    "phimvx",
    "time",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.05,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _expected_fspv(pdynmc, area, cd, cl, thrust, mass, alphax, phimvx):
    alpha = alphax * RAD
    phimv = phimvx * RAD
    fspv1 = (-pdynmc * area * cd + thrust * cos(alpha)) / mass
    fspv2 = sin(phimv) * (pdynmc * area * cl + thrust * sin(alpha)) / mass
    fspv3 = -cos(phimv) * (pdynmc * area * cl + thrust * sin(alpha)) / mass
    return np.array([fspv1, fspv2, fspv3])


def _ready_forces(pdynmc, area, cd, cl, thrust, mass, alphax, phimvx):
    vehicle = _Vehicle()
    forces = Hyper5Forces()
    forces.define(vehicle)
    store = vehicle.store
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("cl", cl, "real", "out", "aerodynamics"))
    store.define(Field("cd", cd, "real", "out", "aerodynamics"))
    store.define(Field("area", area, "real", "data", "aerodynamics"))
    store.define(Field("thrust", thrust, "real", "out", "propulsion"))
    store.define(Field("mass", mass, "real", "out", "propulsion"))
    store.define(Field("alphax", alphax, "real", "out", "control"))
    store.define(Field("phimvx", phimvx, "real", "out", "control"))
    forces.initialize(vehicle, _ctx())
    return vehicle, forces


def test_name_is_forces():
    assert Hyper5Forces().name == "forces"


def test_define_registers_fspv_only():
    vehicle = _Vehicle()
    Hyper5Forces().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == ["FSPV"]
    fspv = store.field("FSPV")
    np.testing.assert_array_equal(fspv.value, np.zeros(3))
    assert fspv.type == "vec"
    assert fspv.role == "out"
    assert fspv.module == "forces"
    assert fspv.outputs == ("plot",)
    for name in EXTERNALS:
        assert name not in store.names()


def test_define_skips_existing_fspv():
    vehicle = _Vehicle()
    store = vehicle.store
    sentinel = np.array([1.0, 2.0, 3.0])
    store.define(Field("FSPV", sentinel, "vec", "out", "newton", ("plot",)))
    Hyper5Forces().define(vehicle)
    np.testing.assert_array_equal(store.get("FSPV"), sentinel)
    field = store.field("FSPV")
    assert field.module == "newton"


def test_initialize_is_pass():
    vehicle, _forces = _ready_forces(
        PDYNMC, AREA, CD, CL, THRUST, MASS, ALPHAX, 0.0
    )
    np.testing.assert_array_equal(vehicle.store.get("FSPV"), np.zeros(3))


def test_fspv_level_phimvx_0():
    vehicle, forces = _ready_forces(
        PDYNMC, AREA, CD, CL, THRUST, MASS, ALPHAX, 0.0
    )
    expected = _expected_fspv(PDYNMC, AREA, CD, CL, THRUST, MASS, ALPHAX, 0.0)
    assert expected[1] == pytest.approx(0.0, abs=ATOL)

    forces.execute(vehicle, _ctx())

    np.testing.assert_allclose(
        vehicle.store.get("FSPV"), expected, rtol=RTOL, atol=ATOL
    )
    assert vehicle.store.get("FSPV")[1] == pytest.approx(0.0, abs=ATOL)


def test_fspv_banked_phimvx_90():
    vehicle, forces = _ready_forces(
        PDYNMC, AREA, CD, CL, THRUST, MASS, ALPHAX, 90.0
    )
    expected = _expected_fspv(PDYNMC, AREA, CD, CL, THRUST, MASS, ALPHAX, 90.0)
    assert expected[1] != pytest.approx(0.0, abs=ATOL)

    forces.execute(vehicle, _ctx())

    np.testing.assert_allclose(
        vehicle.store.get("FSPV"), expected, rtol=RTOL, atol=ATOL
    )
    fspv = vehicle.store.get("FSPV")
    assert fspv[1] != pytest.approx(0.0, abs=ATOL)
