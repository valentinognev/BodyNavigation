from pathlib import Path

import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.sam6.aircraft import Sam6AircraftForces

RTOL = 1e-12
ATOL = 1e-14
GRAV = 9.8
ZEROS3 = (0.0, 0.0, 0.0)

DEFINED = ("FSPA", "acc_longx")
ROLES = {
    "FSPA": "out",
    "acc_longx": "data",
}
TYPES = {
    "FSPA": "vec",
    "acc_longx": "real",
}
NOT_DEFINED = (
    "grav",
    "anx",
    "acft_option",
    "ACOML",
    "TVL",
    "phiav",
    "phiavout",
    "ancomx",
    "FAPB",
    "FMB",
    "time",
    "pdynmc",
    "SAEL",
    "VAEL",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ctx(int_step=0.001):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _plant(store, *, grav=GRAV, anx=1.0):
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("anx", anx, "real", "state", "control", ("com",)))


def _ready(*, acc_longx=0.0, grav=GRAV, anx=1.0):
    vehicle = _Vehicle()
    forces = Sam6AircraftForces()
    forces.define(vehicle)
    forces.initialize(vehicle, _ctx())
    store = vehicle.store
    store.set("acc_longx", acc_longx)
    _plant(store, grav=grav, anx=anx)
    return vehicle, forces


def test_name_is_forces():
    assert Sam6AircraftForces().name == "forces"


def test_define_registers_cpp_fields():
    vehicle = _Vehicle()
    Sam6AircraftForces().define(vehicle)
    store = vehicle.store
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        assert field.module == "forces", name
        assert field.role == ROLES[name], name
        assert field.outputs == ()
        assert field.type == TYPES[name], name
    np.testing.assert_allclose(store.get("FSPA"), ZEROS3, rtol=RTOL, atol=ATOL)
    assert store.get("FSPA").shape == (3,)
    assert _approx(store.get("acc_longx"), 0.0)
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_initialize_is_pass():
    vehicle = _Vehicle()
    forces = Sam6AircraftForces()
    forces.define(vehicle)
    vehicle.store.set("FSPA", (1.0, 2.0, 3.0))
    vehicle.store.set("acc_longx", 4.0)
    before_fspa = np.array(vehicle.store.get("FSPA"), copy=True)
    before_acc = vehicle.store.get("acc_longx")
    assert forces.initialize(vehicle, _ctx()) is None
    np.testing.assert_allclose(
        vehicle.store.get("FSPA"), before_fspa, rtol=RTOL, atol=ATOL
    )
    assert _approx(vehicle.store.get("acc_longx"), before_acc)


def test_level_flight_fspa_is_hang_gravity():
    vehicle, forces = _ready(anx=1.0, acc_longx=0.0, grav=GRAV)
    forces.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        vehicle.store.get("FSPA"), (0.0, 0.0, -GRAV), rtol=RTOL, atol=ATOL
    )


def test_longitudinal_accel_scales_fspa_x():
    vehicle, forces = _ready(anx=1.0, acc_longx=2.0, grav=GRAV)
    forces.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        vehicle.store.get("FSPA"), (19.6, 0.0, -GRAV), rtol=RTOL, atol=ATOL
    )


def test_negative_load_factor_flips_fspa_z():
    vehicle, forces = _ready(anx=-1.5, acc_longx=0.5, grav=GRAV)
    forces.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        vehicle.store.get("FSPA"), (4.9, 0.0, 14.7), rtol=RTOL, atol=ATOL
    )


def test_execute_does_not_write_plant_names():
    vehicle, forces = _ready(anx=1.0, acc_longx=0.0, grav=GRAV)
    forces.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("grav"), GRAV)
    assert _approx(store.get("anx"), 1.0)
    assert _approx(store.get("acc_longx"), 0.0)


def test_fspa_is_a_fresh_vector():
    vehicle, forces = _ready(anx=1.0, acc_longx=0.0, grav=GRAV)
    first = vehicle.store.get("FSPA")
    forces.execute(vehicle, _ctx())
    written = vehicle.store.get("FSPA")
    assert written is not first
    written[0] = 99.0
    forces.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        vehicle.store.get("FSPA"), (0.0, 0.0, -GRAV), rtol=RTOL, atol=ATOL
    )


def test_terminate_is_pass():
    vehicle, forces = _ready(anx=1.0, acc_longx=0.0, grav=GRAV)
    assert forces.terminate(vehicle, _ctx()) is None
    np.testing.assert_allclose(
        vehicle.store.get("FSPA"), ZEROS3, rtol=RTOL, atol=ATOL
    )


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.flat6.sam6.aircraft as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "Plane5" not in src
    assert "Plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src
