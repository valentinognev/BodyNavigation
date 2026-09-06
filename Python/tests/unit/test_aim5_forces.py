import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.aim5.forces import Aim5Forces

RTOL = 1e-12
ATOL = 1e-14

PDYNMC = 21800.0
AREA = 0.01767
CAAIM = 0.3
CYAIM = 0.01
CNAIM = 0.4
THRUST = 28075.0
MASS = 63.8
GRAV = 9.8

EXTERNALS = (
    "pdynmc",
    "area",
    "caaim",
    "cyaim",
    "cnaim",
    "thrust",
    "mass",
    "grav",
    "acc_longx",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.002,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _expected_fspv(pdynmc, area, caaim, cyaim, cnaim, thrust, mass):
    fspv0 = (thrust - caaim * pdynmc * area) / mass
    fspv1 = (cyaim * pdynmc * area) / mass
    fspv2 = (-cnaim * pdynmc * area) / mass
    return np.array([fspv0, fspv1, fspv2])


def _ready_forces():
    vehicle = _Vehicle()
    forces = Aim5Forces()
    forces.define(vehicle)
    store = vehicle.store
    store.define(Field("pdynmc", PDYNMC, "real", "out", "environment"))
    store.define(Field("area", AREA, "real", "data", "aerodynamics"))
    store.define(Field("caaim", CAAIM, "real", "out", "aerodynamics"))
    store.define(Field("cyaim", CYAIM, "real", "out", "aerodynamics"))
    store.define(Field("cnaim", CNAIM, "real", "out", "aerodynamics"))
    store.define(Field("thrust", THRUST, "real", "out", "propulsion"))
    store.define(Field("mass", MASS, "real", "out", "propulsion"))
    store.define(Field("grav", GRAV, "real", "out", "environment"))
    forces.initialize(vehicle, _ctx())
    return vehicle, forces


def test_name_is_forces():
    assert Aim5Forces().name == "forces"


def test_define_registers_fspv_aax_alx_anx():
    vehicle = _Vehicle()
    Aim5Forces().define(vehicle)
    store = vehicle.store
    names = store.names()
    assert "FSPV" in names
    assert "aax" in names
    assert "alx" in names
    assert "anx" in names
    assert "FSPA" not in names
    assert "acc_longx" not in names
    for name in EXTERNALS:
        assert name not in names

    fspv = store.field("FSPV")
    np.testing.assert_array_equal(fspv.value, np.zeros(3))
    assert fspv.type == "vec"
    assert fspv.role == "out"
    assert fspv.module == "forces"

    aax = store.field("aax")
    assert aax.type == "real"
    assert aax.role == "diag"
    assert aax.module == "forces"
    assert aax.outputs == ()
    assert aax.value == pytest.approx(0.0, abs=ATOL)

    plot = ("scrn", "plot")
    for name in ("alx", "anx"):
        field = store.field(name)
        assert field.type == "real"
        assert field.role == "diag"
        assert field.module == "forces"
        assert field.outputs == plot
        assert field.value == pytest.approx(0.0, abs=ATOL)


def test_define_skips_existing_fspv():
    vehicle = _Vehicle()
    store = vehicle.store
    sentinel = np.array([1.0, 2.0, 3.0])
    store.define(Field("FSPV", sentinel, "vec", "out", "pre", ("plot",)))
    Aim5Forces().define(vehicle)
    np.testing.assert_array_equal(store.get("FSPV"), sentinel)
    field = store.field("FSPV")
    assert field.module == "pre"
    assert field.role == "out"
    assert "aax" in store.names()
    assert "alx" in store.names()
    assert "anx" in store.names()
    assert "acc_longx" not in store.names()


def test_initialize_is_pass():
    vehicle, _forces = _ready_forces()
    np.testing.assert_array_equal(vehicle.store.get("FSPV"), np.zeros(3))
    assert vehicle.store.get("aax") == pytest.approx(0.0, abs=ATOL)
    assert vehicle.store.get("alx") == pytest.approx(0.0, abs=ATOL)
    assert vehicle.store.get("anx") == pytest.approx(0.0, abs=ATOL)


def test_terminate_is_pass():
    vehicle, forces = _ready_forces()
    forces.terminate(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("FSPV"), np.zeros(3))


def test_fspv_and_diagnostics_match_formulas():
    vehicle, forces = _ready_forces()
    expected = _expected_fspv(PDYNMC, AREA, CAAIM, CYAIM, CNAIM, THRUST, MASS)
    aax = expected[0] / GRAV
    alx = expected[1] / GRAV
    anx = -expected[2] / GRAV

    forces.execute(vehicle, _ctx())

    np.testing.assert_allclose(
        vehicle.store.get("FSPV"), expected, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(vehicle.store.get("aax"), aax, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("alx"), alx, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("anx"), anx, rtol=RTOL, atol=ATOL)
    assert "FSPA" not in vehicle.store.names()
    assert "acc_longx" not in vehicle.store.names()
