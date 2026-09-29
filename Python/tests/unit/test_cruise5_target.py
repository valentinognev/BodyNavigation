import inspect
import pathlib

import numpy as np

from cadac.eom.round3 import Round3Newton
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round3.cruise5.environment import Cruise5Environment
from cadac.vehicles.round3.cruise5.target import (
    Cruise5Target,
    Cruise5TargetForces,
    Cruise5TargetIntercept,
)

LONX, LATX, ALT = 15.4, 35.3, 100.0
PSIVGX, DVBE, THTVGX = 45.0, 10.0, 0.0
COM_AT_LEAST = (
    "time",
    "mach",
    "lonx",
    "latx",
    "alt",
    "dvbe",
    "psivgx",
    "thtvgx",
    "sbeg",
    "vbeg",
    "sbii",
)
TARGET_PY = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "cadac"
    / "vehicles"
    / "round3"
    / "cruise5"
    / "target.py"
)


def _ctx(status=1, vehicle_slot=0):
    combus = [
        Packet(name="Tank_t1", type="TARGET3", status=status, vars={}),
    ]
    return SimContext(
        sim_time=0.0,
        int_step=0.05,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=vehicle_slot,
    )


def test_constructor_type_health_no_aero_deck():
    sig = inspect.signature(Cruise5Target.__init__)
    assert list(sig.parameters) == ["self", "name", "events"]
    assert sig.parameters["events"].default is None
    vehicle = Cruise5Target("Tank_t1")
    assert vehicle.type == "TARGET3"
    assert vehicle.name == "Tank_t1"
    assert vehicle.health == 1
    assert not hasattr(vehicle, "aero_deck")
    assert [type(m) for m in vehicle.modules] == [
        Cruise5Environment,
        Cruise5TargetForces,
        Round3Newton,
        Cruise5TargetIntercept,
    ]


def test_com_names_from_com_outputs():
    vehicle = Cruise5Target("Tank_t1")
    vehicle.define()
    for name in COM_AT_LEAST:
        assert name in vehicle.com_names
        assert "com" in vehicle.store.field(name).outputs
    assert vehicle.com_names == [
        name
        for name in vehicle.store.names()
        if "com" in vehicle.store.field(name).outputs
    ]


def test_define_skips_existing_fspv():
    vehicle = type("V", (), {"store": StateStore()})()
    sentinel = np.array([1.0, 2.0, 3.0])
    vehicle.store.define(Field("FSPV", sentinel, "vec", "out", "newton"))
    Cruise5TargetForces().define(vehicle)
    np.testing.assert_array_equal(vehicle.store.get("FSPV"), sentinel)
    assert vehicle.store.field("FSPV").module == "newton"
    assert vehicle.store.get("fwd_accel") == 0.0
    assert vehicle.store.get("side_accel") == 0.0


def test_tank_ics_forces_fspv_finite_shape():
    vehicle = Cruise5Target("Tank_t1")
    vehicle.define()
    store = vehicle.store
    store.set("lonx", LONX)
    store.set("latx", LATX)
    store.set("alt", ALT)
    store.set("psivgx", PSIVGX)
    store.set("dvbe", DVBE)
    store.set("thtvgx", THTVGX)
    assert store.get("fwd_accel") == 0.0
    assert store.get("side_accel") == 0.0
    ctx = _ctx()
    named = {m.name: m for m in vehicle.modules}
    named["environment"].initialize(vehicle, ctx)
    named["newton"].initialize(vehicle, ctx)
    named["forces"].execute(vehicle, ctx)
    fspv = store.get("FSPV")
    assert np.all(np.isfinite(fspv))
    assert fspv.shape == (3,)


def test_intercept_combus_status_0_sets_targ_health_0():
    vehicle = Cruise5Target("Tank_t1")
    vehicle.define()
    ctx = _ctx(status=0)
    named = {m.name: m for m in vehicle.modules}
    named["intercept"].execute(vehicle, ctx)
    assert vehicle.store.get("targ_health") == 0


def test_no_hyper5_target_import():
    text = TARGET_PY.read_text()
    assert "cadac.vehicles.round3.hyper5.target" not in text
    assert "Target3" not in text
