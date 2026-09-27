import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat3.aim5.aircraft import Aim5AircraftForces

RTOL = 1e-12
ATOL = 1e-14

GRAV = 9.8
ANX = 1.0

EXTERNALS = ("grav", "anx")


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


def _ready_forces(acc_longx=0.0, anx=ANX, grav=GRAV):
    vehicle = _Vehicle()
    forces = Aim5AircraftForces()
    forces.define(vehicle)
    store = vehicle.store
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("anx", anx, "real", "diag", "control"))
    store.set("acc_longx", acc_longx)
    forces.initialize(vehicle, _ctx())
    return vehicle, forces


def test_name_is_forces():
    assert Aim5AircraftForces().name == "forces"


def test_define_registers_fspv_and_acc_longx():
    vehicle = _Vehicle()
    Aim5AircraftForces().define(vehicle)
    store = vehicle.store
    names = store.names()
    assert "FSPV" in names
    assert "acc_longx" in names
    assert "FSPA" not in names
    for name in EXTERNALS:
        assert name not in names

    fspv = store.field("FSPV")
    np.testing.assert_array_equal(fspv.value, np.zeros(3))
    assert fspv.type == "vec"
    assert fspv.role == "out"
    assert fspv.module == "forces"

    acc_longx = store.field("acc_longx")
    assert acc_longx.type == "real"
    assert acc_longx.role == "data"
    assert acc_longx.module == "forces"
    assert acc_longx.outputs == ()
    assert acc_longx.value == pytest.approx(0.0, abs=ATOL)


def test_define_skips_existing_fspv():
    vehicle = _Vehicle()
    store = vehicle.store
    sentinel = np.array([1.0, 2.0, 3.0])
    store.define(Field("FSPV", sentinel, "vec", "out", "pre", ("plot",)))
    Aim5AircraftForces().define(vehicle)
    np.testing.assert_array_equal(store.get("FSPV"), sentinel)
    field = store.field("FSPV")
    assert field.module == "pre"
    assert field.role == "out"
    assert "acc_longx" in store.names()
    acc_longx = store.field("acc_longx")
    assert acc_longx.module == "forces"
    assert acc_longx.value == pytest.approx(0.0, abs=ATOL)


def test_initialize_is_pass():
    vehicle, _forces = _ready_forces()
    np.testing.assert_array_equal(vehicle.store.get("FSPV"), np.zeros(3))
    assert vehicle.store.get("acc_longx") == pytest.approx(0.0, abs=ATOL)
    assert vehicle.store.get("anx") == pytest.approx(ANX, abs=ATOL)


def test_terminate_is_pass():
    vehicle, forces = _ready_forces()
    forces.terminate(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("FSPV"), np.zeros(3))
    assert vehicle.store.get("acc_longx") == pytest.approx(0.0, abs=ATOL)


def test_fspv_zero_long_acc_and_unit_load():
    vehicle, forces = _ready_forces(acc_longx=0.0, anx=ANX, grav=GRAV)
    forces.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        vehicle.store.get("FSPV"), np.array([0.0, 0.0, -9.8]), rtol=RTOL, atol=ATOL
    )
    assert "FSPA" not in vehicle.store.names()


def test_fspv_half_g_long_acc():
    vehicle, forces = _ready_forces(acc_longx=0.5, anx=ANX, grav=GRAV)
    forces.execute(vehicle, _ctx())
    fspv = vehicle.store.get("FSPV")
    np.testing.assert_allclose(fspv[0], 4.9, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(fspv[1], 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(fspv[2], -9.8, rtol=RTOL, atol=ATOL)
