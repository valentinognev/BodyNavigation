from pathlib import Path

import pytest

from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.sam6.rocket import Sam6RocketIntercept

RTOL = 1e-12
ATOL = 1e-14

DEFINED = ("write",)
NOT_DEFINED = (
    "time",
    "dvae",
    "SAEL",
    "psivlx",
    "thtvlx",
    "alt",
    "dta",
    "dvta",
    "STAL",
    "mguide",
    "miss",
    "hit_time",
    "MISS",
    "mterm",
    "stop",
    "hbe",
    "SBEL",
    "VBEL",
    "FSPA",
    "alphax",
    "ancomx",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()
        self.health = 1


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ctx(combus=None, vehicle_slot=0):
    if combus is None:
        combus = [
            Packet(name="R1", type="ROCKET5", status=1, vars={}),
            Packet(name="T1", type="AIRCRAFT3", status=1, vars={}),
        ]
    return SimContext(
        sim_time=0.0,
        int_step=0.001,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=vehicle_slot,
    )


def _plant(store, *, alt=1000.0, dta=2000.0, dvta=0.0, mguide=0):
    store.define(Field("alt", alt, "real", "out", "newton"))
    store.define(Field("dta", dta, "real", "out", "sensor", ("com",)))
    store.define(Field("dvta", dvta, "real", "out", "sensor", ("com",)))
    store.define(Field("mguide", mguide, "int", "data", "guidance"))


def _ready(*, write=1, alt=1000.0, dta=2000.0, dvta=0.0, mguide=0, combus=None):
    vehicle = _Vehicle()
    intercept = Sam6RocketIntercept()
    intercept.define(vehicle)
    vehicle.store.set("write", write)
    _plant(vehicle.store, alt=alt, dta=dta, dvta=dvta, mguide=mguide)
    return vehicle, intercept, _ctx(combus=combus)


def test_name_is_intercept():
    assert Sam6RocketIntercept().name == "intercept"


def test_define_registers_write_latch():
    vehicle = _Vehicle()
    Sam6RocketIntercept().define(vehicle)
    store = vehicle.store
    assert tuple(store.names()) == DEFINED
    field = store.field("write")
    assert field.module == "intercept"
    assert field.role == "init"
    assert field.outputs == ()
    assert field.type == "int"
    assert store.get("write") == 1
    assert type(store.get("write")) is int
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_initialize_and_terminate_are_pass():
    vehicle, intercept, ctx = _ready()
    assert intercept.initialize(vehicle, ctx) is None
    assert intercept.terminate(vehicle, ctx) is None
    assert vehicle.store.get("write") == 1
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1


def test_ground_alt_negative_write_1_sets_health_0():
    vehicle, intercept, ctx = _ready(alt=-1.0, write=1, mguide=0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert vehicle.store.get("write") == 0
    assert type(vehicle.store.get("write")) is int
    assert ctx.combus[1].status == 1


def test_alt_1000_write_1_mguide_0_leaves_health_unchanged():
    vehicle, intercept, ctx = _ready(alt=1000.0, write=1, mguide=0, dta=500.0, dvta=1.0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 1
    assert ctx.combus[1].status == 1


def test_closest_approach_dta_under_1000_mguide_and_dvta_positive_kills():
    vehicle, intercept, ctx = _ready(
        alt=1000.0, write=1, dta=500.0, mguide=1, dvta=1.0
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert vehicle.store.get("write") == 0
    assert ctx.combus[1].status == 1
    assert _approx(vehicle.store.get("dta"), 500.0)
    assert _approx(vehicle.store.get("dvta"), 1.0)


def test_dta_1000_does_not_kill():
    vehicle, intercept, ctx = _ready(
        alt=1000.0, write=1, dta=1000.0, mguide=1, dvta=1.0
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 1


def test_dvta_not_positive_does_not_kill():
    vehicle, intercept, ctx = _ready(
        alt=1000.0, write=1, dta=500.0, mguide=1, dvta=0.0
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 1


def test_dvta_negative_does_not_kill():
    vehicle, intercept, ctx = _ready(
        alt=1000.0, write=1, dta=500.0, mguide=1, dvta=-1.0
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 1


def test_alt_zero_does_not_kill():
    vehicle, intercept, ctx = _ready(alt=0.0, write=1, mguide=0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 1


def test_ground_write_0_does_not_kill():
    vehicle, intercept, ctx = _ready(alt=-1.0, write=0, mguide=0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 0


def test_closest_approach_write_0_does_not_kill():
    vehicle, intercept, ctx = _ready(
        alt=1000.0, write=0, dta=500.0, mguide=1, dvta=1.0
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 0


def test_write_latch_does_not_keep_killing():
    vehicle, intercept, ctx = _ready(alt=-1.0, write=1, mguide=0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert vehicle.store.get("write") == 0
    vehicle.health = 1
    ctx.combus[ctx.vehicle_slot].status = 1
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 0


def test_closest_approach_latches_before_ground():
    vehicle, intercept, ctx = _ready(
        alt=-1.0, write=1, dta=500.0, mguide=1, dvta=1.0
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert vehicle.store.get("write") == 0


def test_always_writes_write_back_when_unchanged():
    vehicle, intercept, ctx = _ready(alt=1000.0, write=1, mguide=0)
    intercept.execute(vehicle, ctx)
    assert vehicle.store.get("write") == 1
    assert type(vehicle.store.get("write")) is int


def test_does_not_sys_exit_on_ground():
    vehicle, intercept, ctx = _ready(alt=-1.0, write=1)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0


def test_no_sys_exit_or_print():
    import cadac.vehicles.flat6.sam6.rocket as mod

    text = Path(mod.__file__).read_text(encoding="utf-8")
    assert "sys.exit" not in text
    assert "print(" not in text


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.flat6.sam6.rocket as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "cadac.eom.flat3" not in src
    assert "Flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "Plane5" not in src
    assert "Plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src
    assert "class Sam6RocketIntercept:" in src
    assert "from cadac.vehicles.flat6.sam6.intercept" not in src
    assert "from cadac.eom.flat3" not in src
