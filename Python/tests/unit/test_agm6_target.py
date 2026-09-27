import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cadac.cli import _VEHICLE_FAMILIES
from cadac.eom.flat3 import Flat3Kinematics
from cadac.env.gravity import gravity
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.agm6.flat3io import (
    Agm6Flat3Environment,
    Agm6Flat3Newton,
    copy_in,
    copy_out,
)
from cadac.vehicles.flat6.agm6.target import Agm6Target, Agm6TargetForces

RTOL = 1e-12
ATOL = 1e-14
ZEROS3 = (0.0, 0.0, 0.0)
ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

SAEL1 = 33000.0
SAEL2 = 10000.0
SAEL3 = -100.0
DVAE = 5.0
PSIVLX = -90.0
ACC_LATX = 0.01
ACC_LONGX = 0.0
ALT = -SAEL3
GRAV = gravity(ALT)

FORCES_FIELDS = {
    "FSPA": ("vec", "out", ZEROS3, ()),
    "aax": ("real", "diag", 0.0, ("com",)),
    "alx": ("real", "diag", 0.0, ("com",)),
    "anx": ("real", "diag", 0.0, ("com",)),
    "acc_longx": ("real", "data", 0.0, ()),
    "acc_latx": ("real", "data", 0.0, ()),
}


def _ctx(int_step=0.001):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _define_bridge_names(store):
    for name, ftype, default in (
        ("sael1", "real", 0.0),
        ("sael2", "real", 0.0),
        ("sael3", "real", 0.0),
        ("sbel1", "real", 0.0),
        ("sbel2", "real", 0.0),
        ("sbel3", "real", 0.0),
        ("dvae", "real", 0.0),
        ("dvbe", "real", 0.0),
    ):
        store.define(Field(name, default, ftype, "init", "newton"))
    for name in ("SAEL", "SBEL", "VAEL", "VBEL", "FSPA", "FSPV"):
        store.define(Field(name, ZEROS3, "vec", "state", "newton"))
    store.define(Field("TAL", ZEROS33, "mat", "out", "newton"))
    store.define(Field("TBL", ZEROS33, "mat", "out", "newton"))


def _plant_ics(store):
    store.set("sael1", SAEL1)
    store.set("sael2", SAEL2)
    store.set("sael3", SAEL3)
    store.set("dvae", DVAE)
    store.set("psivlx", PSIVLX)
    store.set("acc_latx", ACC_LATX)
    store.set("acc_longx", ACC_LONGX)


def test_environment_name_is_environment():
    assert Agm6Flat3Environment.name == "environment"
    assert Agm6Flat3Environment().name == "environment"


def test_newton_name_is_newton():
    assert Agm6Flat3Newton.name == "newton"
    assert Agm6Flat3Newton().name == "newton"


def test_forces_name_is_forces():
    assert Agm6TargetForces.name == "forces"
    assert Agm6TargetForces().name == "forces"


def test_copy_in_maps_cpp_names_and_builds_sael_from_components():
    store = StateStore()
    _define_bridge_names(store)
    store.set("sael1", SAEL1)
    store.set("sael2", SAEL2)
    store.set("sael3", SAEL3)
    store.set("dvae", DVAE)
    store.set("FSPA", (1.0, 2.0, 3.0))
    np.testing.assert_array_equal(store.get("SAEL"), np.zeros(3))

    copy_in(store)

    want = np.array([SAEL1, SAEL2, SAEL3])
    np.testing.assert_allclose(store.get("SAEL"), want, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBEL"), want, rtol=RTOL, atol=ATOL)
    assert store.get("sbel1") == SAEL1
    assert store.get("sbel2") == SAEL2
    assert store.get("sbel3") == SAEL3
    assert store.get("dvbe") == DVAE
    np.testing.assert_allclose(
        store.get("FSPV"), np.array([1.0, 2.0, 3.0]), rtol=RTOL, atol=ATOL
    )


def test_copy_in_does_not_rebuild_nonzero_sael():
    store = StateStore()
    _define_bridge_names(store)
    planted = np.array([1.0, 2.0, 3.0])
    store.set("SAEL", planted)
    store.set("sael1", SAEL1)
    store.set("sael2", SAEL2)
    store.set("sael3", SAEL3)
    copy_in(store)
    np.testing.assert_allclose(store.get("SAEL"), planted, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBEL"), planted, rtol=RTOL, atol=ATOL)
    assert store.get("sbel1") == SAEL1


def test_copy_out_maps_inner_flat3_to_cpp_names():
    store = StateStore()
    _define_bridge_names(store)
    sbel = np.array([10.0, 20.0, 30.0])
    vbel = np.array([4.0, 5.0, 6.0])
    tbl = np.diag([1.0, 2.0, 3.0])
    store.set("SBEL", sbel)
    store.set("VBEL", vbel)
    store.set("dvbe", 7.0)
    store.set("TBL", tbl)

    copy_out(store)

    np.testing.assert_allclose(store.get("SAEL"), sbel, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VAEL"), vbel, rtol=RTOL, atol=ATOL)
    assert store.get("dvae") == 7.0
    np.testing.assert_allclose(store.get("TAL"), tbl, rtol=RTOL, atol=ATOL)
    assert store.get("sbel1") == 10.0
    assert store.get("sbel2") == 20.0
    assert store.get("sbel3") == 30.0


def test_forces_define_registers_cpp_fields_and_fspv():
    vehicle = SimpleNamespace(store=StateStore())
    Agm6TargetForces().define(vehicle)
    store = vehicle.store
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
    assert store.field("FSPV").type == "vec"
    assert store.field("aax").outputs == ("com",)
    assert store.field("alx").outputs == ("com",)
    assert store.field("anx").outputs == ("com",)


def test_forces_execute_fspa_fspv_and_diags():
    vehicle = SimpleNamespace(store=StateStore())
    forces = Agm6TargetForces()
    forces.define(vehicle)
    store = vehicle.store
    store.define(Field("grav", GRAV, "real", "out", "environment"))
    store.set("acc_longx", ACC_LONGX)
    store.set("acc_latx", ACC_LATX)
    forces.execute(vehicle, _ctx())
    want = np.array([ACC_LONGX * GRAV, ACC_LATX * GRAV, -GRAV])
    np.testing.assert_allclose(store.get("FSPA"), want, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FSPV"), want, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FSPA")[1], 0.01 * GRAV, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FSPA")[2], -GRAV, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("aax"), ACC_LONGX, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("alx"), ACC_LATX, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("anx"), 1.0, rtol=RTOL, atol=ATOL)


def test_target_type_constructor_modules_and_com_names():
    sig = inspect.signature(Agm6Target.__init__)
    assert list(sig.parameters) == ["self", "name", "events"]
    assert sig.parameters["events"].default is None
    vehicle = Agm6Target("Ground")
    assert vehicle.type == "TARGET3"
    assert vehicle.name == "Ground"
    assert vehicle.health == 1
    assert [type(m) for m in vehicle.modules] == [
        Agm6Flat3Environment,
        Flat3Kinematics,
        Agm6TargetForces,
        Agm6Flat3Newton,
    ]
    vehicle.define()
    for name in ("SAEL", "VAEL"):
        assert name in vehicle.com_names
        assert "com" in vehicle.store.field(name).outputs


def test_define_param_set_initialize_sael_and_one_execute_fspa():
    vehicle = Agm6Target("Ground")
    vehicle.define()
    _plant_ics(vehicle.store)
    ctx = _ctx()
    for module in vehicle.modules:
        module.initialize(vehicle, ctx)
    np.testing.assert_allclose(
        vehicle.store.get("SAEL")[0], SAEL1, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("SAEL"),
        np.array([SAEL1, SAEL2, SAEL3]),
        rtol=RTOL,
        atol=ATOL,
    )
    for module in vehicle.modules:
        module.execute(vehicle, ctx)
    grav = vehicle.store.get("grav")
    np.testing.assert_allclose(grav, GRAV, rtol=RTOL, atol=ATOL)
    fspa = vehicle.store.get("FSPA")
    np.testing.assert_allclose(fspa[1], 0.01 * grav, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(fspa[2], -grav, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("FSPV"), fspa, rtol=RTOL, atol=ATOL)


def test_hyper5_target3_class_not_imported():
    import cadac.vehicles.flat6.agm6.flat3io as flat3io
    import cadac.vehicles.flat6.agm6.target as target_mod

    for mod in (flat3io, target_mod):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "hyper5" not in src.lower()
        assert "from cadac.vehicles.round3.hyper5" not in src


def test_target3_registered_in_families():
    from cadac.vehicles.flat6.agm6.target import Agm6Target

    assert _VEHICLE_FAMILIES[("agm6", "TARGET3")] is Agm6Target
