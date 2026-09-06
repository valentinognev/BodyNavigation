import inspect

import numpy as np
import pytest

from cadac.eom.round3 import Round3Environment, Round3Newton
from cadac.env.gravity import gravity
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.hyper5.target import Target3, Target3Forces, Target3Intercept

RTOL = 1e-12
ATOL = 1e-14

# Demo 4.7 TARGET3 Truck_t1
LONX = -106.28
LATX = 33.4
ALT = 1200.0

FORCES_FIELDS = {
    "FSPV": ("vec", "out", (0.0, 0.0, 0.0), ()),
    "fwd_accel": ("real", "data", 0.0, ()),
    "side_accel": ("real", "data", 0.0, ()),
    "CORIO_V": ("vec", "diag", (0.0, 0.0, 0.0), ()),
    "CENTR_V": ("vec", "diag", (0.0, 0.0, 0.0), ()),
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
    "grav",
    "tgv",
    "tig",
    "tge",
    "weii",
    "vbeg",
    "sbii",
    "thtvgx",
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


def _expected_fspv(store, fwd_accel=0.0, side_accel=0.0):
    tgv = store.get("tgv")
    tig = store.get("tig")
    tge = store.get("tge")
    weii = store.get("weii")
    vbeg = store.get("vbeg")
    sbii = store.get("sbii")
    grav = store.get("grav")
    tvg = tgv.T
    tgi = tig.T
    teg = tge.T
    weig = tge @ weii @ teg
    corio_v = tvg @ weig @ vbeg * 2
    centr_v = tvg @ weig @ weig @ tgi @ sbii
    grav_g = np.zeros(3)
    grav_g[2] = grav
    grav_v = tvg @ grav_g
    acc_v = corio_v + centr_v - grav_v
    return acc_v + np.array([fwd_accel, side_accel, 0.0]), corio_v, centr_v


def _ready_forces(fwd_accel=0.0, side_accel=0.0):
    vehicle = _Vehicle()
    newton = Round3Newton()
    newton.define(vehicle)
    store = vehicle.store
    store.set("lonx", LONX)
    store.set("latx", LATX)
    store.set("alt", ALT)
    newton.initialize(vehicle, _ctx())
    store.define(Field("grav", gravity(ALT), "real", "out", "environment"))
    forces = Target3Forces()
    forces.define(vehicle)
    store.set("fwd_accel", fwd_accel)
    store.set("side_accel", side_accel)
    forces.initialize(vehicle, _ctx())
    return vehicle, forces


def test_name_is_forces():
    assert Target3Forces().name == "forces"


def test_define_registers_cpp_def_forces_fields():
    vehicle = _Vehicle()
    Target3Forces().define(vehicle)
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


def test_define_does_not_register_plant_or_unused_thtvgx():
    vehicle = _Vehicle()
    Target3Forces().define(vehicle)
    for name in NOT_DEFINED:
        with pytest.raises(KeyError):
            vehicle.store.get(name)


def test_define_skips_existing_fspv():
    vehicle = _Vehicle()
    store = vehicle.store
    sentinel = np.array([1.0, 2.0, 3.0])
    store.define(Field("FSPV", sentinel, "vec", "out", "newton", ("plot",)))
    Target3Forces().define(vehicle)
    np.testing.assert_array_equal(store.get("FSPV"), sentinel)
    field = store.field("FSPV")
    assert field.module == "newton"
    assert "fwd_accel" in store.names()
    assert "side_accel" in store.names()
    assert "CORIO_V" in store.names()
    assert "CENTR_V" in store.names()


def test_initialize_is_pass():
    vehicle, _forces = _ready_forces()
    np.testing.assert_array_equal(vehicle.store.get("FSPV"), np.zeros(3))


def test_terminate_exists_and_is_pass():
    vehicle, forces = _ready_forces()
    store = vehicle.store
    store.set("fwd_accel", 1.5)
    store.set("side_accel", -0.2)
    forces.terminate(vehicle, _ctx())
    assert store.get("fwd_accel") == 1.5
    assert store.get("side_accel") == -0.2
    np.testing.assert_array_equal(store.get("FSPV"), np.zeros(3))


def test_fspv_finite_after_newton_init_fwd_side_zero():
    vehicle, forces = _ready_forces(fwd_accel=0.0, side_accel=0.0)
    forces.execute(vehicle, _ctx())
    fspv = vehicle.store.get("FSPV")
    assert np.all(np.isfinite(fspv))
    assert np.linalg.norm(fspv) > 0.0


def test_fspv_matches_cpp_after_newton_init_fwd_side_zero():
    vehicle, forces = _ready_forces(fwd_accel=0.0, side_accel=0.0)
    expected, corio, centr = _expected_fspv(vehicle.store)
    forces.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        vehicle.store.get("FSPV"), expected, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("CORIO_V"), corio, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("CENTR_V"), centr, rtol=RTOL, atol=ATOL
    )


def test_commanded_accels_add_to_fspv():
    vehicle, forces = _ready_forces(fwd_accel=0.3, side_accel=-0.4)
    expected, _, _ = _expected_fspv(vehicle.store, fwd_accel=0.3, side_accel=-0.4)
    forces.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        vehicle.store.get("FSPV"), expected, rtol=RTOL, atol=ATOL
    )


def test_target3_type_health_constructor_and_modules():
    sig = inspect.signature(Target3.__init__)
    assert list(sig.parameters) == ["self", "name", "events"]
    assert sig.parameters["events"].default is None
    vehicle = Target3("Truck_t1")
    assert vehicle.type == "TARGET3"
    assert vehicle.name == "Truck_t1"
    assert vehicle.health == 1
    assert [type(m) for m in vehicle.modules] == [
        Round3Environment,
        Target3Forces,
        Round3Newton,
        Target3Intercept,
    ]
    vehicle.define()
    for name in COM_AT_LEAST:
        assert name in vehicle.com_names
        assert "com" in vehicle.store.field(name).outputs


def test_target3_fspv_finite_after_newton_init():
    vehicle = Target3("Truck_t1")
    vehicle.define()
    store = vehicle.store
    store.set("lonx", LONX)
    store.set("latx", LATX)
    store.set("alt", ALT)
    ctx = _ctx()
    named = {m.name: m for m in vehicle.modules}
    named["environment"].initialize(vehicle, ctx)
    named["newton"].initialize(vehicle, ctx)
    named["environment"].execute(vehicle, ctx)
    named["forces"].execute(vehicle, ctx)
    fspv = store.get("FSPV")
    assert np.all(np.isfinite(fspv))
    expected, _, _ = _expected_fspv(store)
    np.testing.assert_allclose(fspv, expected, rtol=RTOL, atol=ATOL)
    assert store.get("fwd_accel") == 0.0
    assert store.get("side_accel") == 0.0
