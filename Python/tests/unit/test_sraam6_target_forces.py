import numpy as np

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.sraam6.target import Sraam6TargetForces

RTOL = 1e-12
ATOL = 1e-14

GRAV = 9.80675445
ANX = 1.0

DEFINED = ("FSPA", "acc_longx")
EXTERNALS = ("grav", "anx")


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.001,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=1,
    )


def _plant_externals(store, *, grav=GRAV, anx=ANX):
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("anx", anx, "real", "state", "control", ("com",)))


def _ready(*, acc_longx=0.0, anx=ANX, grav=GRAV, plant=True):
    vehicle = _Vehicle()
    forces = Sraam6TargetForces()
    forces.define(vehicle)
    if plant:
        _plant_externals(vehicle.store, grav=grav, anx=anx)
    store = vehicle.store
    store.set("acc_longx", acc_longx)
    forces.initialize(vehicle, _ctx())
    return vehicle, forces, _ctx()


def test_name_is_forces():
    assert Sraam6TargetForces().name == "forces"


def test_define_registers_def_forces_fields():
    vehicle = _Vehicle()
    Sraam6TargetForces().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    field = store.field("FSPA")
    assert field.type == "vec"
    assert field.role == "out"
    assert field.module == "forces"
    assert field.outputs == ()
    np.testing.assert_allclose(store.get("FSPA"), np.zeros(3), rtol=RTOL, atol=ATOL)
    field = store.field("acc_longx")
    assert field.type == "real"
    assert field.role == "data"
    assert field.module == "forces"
    assert field.outputs == ()
    np.testing.assert_allclose(store.get("acc_longx"), 0.0, rtol=RTOL, atol=ATOL)
    for name in EXTERNALS:
        assert name not in store.names()


def test_default_acc_longx_zero_anx_one_fspa_minus_grav():
    vehicle, forces, ctx = _ready(acc_longx=0.0, anx=1.0, grav=GRAV)
    forces.execute(vehicle, ctx)
    want = np.array([0.0, 0.0, -GRAV])
    np.testing.assert_allclose(vehicle.store.get("FSPA"), want, rtol=RTOL, atol=ATOL)


def test_acc_longx_scales_axial_specific_force():
    acc_longx = 0.5
    anx = 2.0
    vehicle, forces, ctx = _ready(acc_longx=acc_longx, anx=anx, grav=GRAV)
    forces.execute(vehicle, ctx)
    want = np.array([acc_longx * GRAV, 0.0, -anx * GRAV])
    np.testing.assert_allclose(vehicle.store.get("FSPA"), want, rtol=RTOL, atol=ATOL)


def test_initialize_and_terminate_are_pass():
    vehicle, forces, ctx = _ready(acc_longx=0.25, anx=1.5, grav=GRAV)
    np.testing.assert_allclose(vehicle.store.get("FSPA"), np.zeros(3), rtol=RTOL, atol=ATOL)
    forces.terminate(vehicle, ctx)
    np.testing.assert_allclose(vehicle.store.get("FSPA"), np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("acc_longx"), 0.25, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("anx"), 1.5, rtol=RTOL, atol=ATOL)
