import inspect
import pathlib

import numpy as np
import pytest

from cadac.eom.round3 import Round3Environment, Round3Newton
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round3.hyper5.satellite import Satellite3, Satellite3Forces

RTOL = 1e-12
ATOL = 1e-14

FORCES_FIELDS = {
    "FSPV": ("vec", "out", (0.0, 0.0, 0.0), ()),
    "sat_thrust": ("real", "data", 0.0, ()),
    "sat_mass": ("real", "data", 100.0, ()),
}

NOT_DEFINED = (
    "pdynmc",
    "cl",
    "cd",
    "area",
    "thrust",
    "mass",
    "alphax",
    "phimvx",
    "time",
    "mseeker",
    "fwd_accel",
    "side_accel",
)

COM_AT_LEAST = (
    "time",
    "mach",
    "lonx",
    "latx",
    "alt",
    "dvbe",
    "psivgx",
    "thtvgx",
    "vbeg",
    "sbii",
)

SATELLITE_PY = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "cadac"
    / "vehicles"
    / "round3"
    / "hyper5"
    / "satellite.py"
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


def _ready_forces(sat_thrust=0.0, sat_mass=100.0):
    vehicle = _Vehicle()
    forces = Satellite3Forces()
    forces.define(vehicle)
    store = vehicle.store
    store.set("sat_thrust", sat_thrust)
    store.set("sat_mass", sat_mass)
    forces.initialize(vehicle, _ctx())
    return vehicle, forces


def test_name_is_forces():
    assert Satellite3Forces().name == "forces"


def test_define_registers_cpp_def_forces_fields():
    vehicle = _Vehicle()
    Satellite3Forces().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(FORCES_FIELDS)
    for name, (ftype, role, default, outputs) in FORCES_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "forces"
        assert field.outputs == outputs
        if ftype == "real":
            assert store.get(name) == default
        else:
            np.testing.assert_array_equal(store.get(name), np.zeros(3))


def test_define_does_not_register_plant_or_seeker():
    vehicle = _Vehicle()
    Satellite3Forces().define(vehicle)
    for name in NOT_DEFINED:
        with pytest.raises(KeyError):
            vehicle.store.get(name)


def test_define_skips_existing_fspv():
    vehicle = _Vehicle()
    store = vehicle.store
    sentinel = np.array([1.0, 2.0, 3.0])
    store.define(Field("FSPV", sentinel, "vec", "out", "newton", ("plot",)))
    Satellite3Forces().define(vehicle)
    np.testing.assert_array_equal(store.get("FSPV"), sentinel)
    field = store.field("FSPV")
    assert field.module == "newton"
    assert store.get("sat_thrust") == 0.0
    assert store.get("sat_mass") == 100.0


def test_initialize_is_pass():
    vehicle, _forces = _ready_forces()
    np.testing.assert_array_equal(vehicle.store.get("FSPV"), np.zeros(3))


def test_terminate_exists_and_is_pass():
    vehicle, forces = _ready_forces(sat_thrust=100.0, sat_mass=100.0)
    store = vehicle.store
    forces.terminate(vehicle, _ctx())
    assert store.get("sat_thrust") == 100.0
    assert store.get("sat_mass") == 100.0
    np.testing.assert_array_equal(store.get("FSPV"), np.zeros(3))


def test_zero_thrust_zero_fspv0():
    vehicle, forces = _ready_forces(sat_thrust=0.0, sat_mass=100.0)
    forces.execute(vehicle, _ctx())
    fspv = vehicle.store.get("FSPV")
    np.testing.assert_allclose(fspv[0], 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(fspv[1], 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(fspv[2], 0.0, rtol=RTOL, atol=ATOL)


def test_thrust_100_fspv0_is_one():
    vehicle, forces = _ready_forces(sat_thrust=100.0, sat_mass=100.0)
    forces.execute(vehicle, _ctx())
    fspv = vehicle.store.get("FSPV")
    np.testing.assert_allclose(fspv[0], 1.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(fspv[1], 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(fspv[2], 0.0, rtol=RTOL, atol=ATOL)


def test_satellite3_type_health_constructor_and_modules():
    sig = inspect.signature(Satellite3.__init__)
    assert list(sig.parameters) == ["self", "name", "events"]
    assert sig.parameters["events"].default is None
    vehicle = Satellite3("Satellite_s1")
    assert vehicle.type == "SATELLITE3"
    assert vehicle.name == "Satellite_s1"
    assert vehicle.health == 1
    assert [type(m) for m in vehicle.modules] == [
        Round3Environment,
        Satellite3Forces,
        Round3Newton,
    ]
    assert all(type(m).__name__ != "Hyper5Seeker" for m in vehicle.modules)
    vehicle.define()
    for name in COM_AT_LEAST:
        assert name in vehicle.com_names
        assert "com" in vehicle.store.field(name).outputs
    assert vehicle.store.get("sat_thrust") == 0.0
    assert vehicle.store.get("sat_mass") == 100.0


def test_satellite3_default_thrust_over_mass():
    vehicle = Satellite3("Satellite_s1")
    vehicle.define()
    named = {m.name: m for m in vehicle.modules}
    named["forces"].execute(vehicle, _ctx())
    fspv = vehicle.store.get("FSPV")
    np.testing.assert_allclose(fspv[0], 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(fspv[1:], 0.0, rtol=RTOL, atol=ATOL)


def test_no_plane5_or_cruise3_imports():
    text = SATELLITE_PY.read_text(encoding="utf-8")
    assert "plane5" not in text
    assert "cruise3" not in text
    assert "Plane5" not in text
    assert "Cruise3" not in text
