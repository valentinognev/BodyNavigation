import numpy as np
import pytest

from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.hyper5.intercept import Hyper5Intercept

RTOL = 1e-12
ATOL = 1e-14

# Closest-approach fixture (hand-derived from C++ linear interpolation).
# Integer-friendly steps so IEEE doubles match the exact miss [0, -3, 0].
TIME = 11.0
TIME_M = 10.0
INT_STEP = 1.0
SBEG = np.array([20.0, 0.0, 0.0])
SBMEG = np.array([10.0, 0.0, 0.0])
STEG = np.array([25.0, 3.0, 0.0])
STMEG = np.array([25.0, 3.0, 0.0])
STBG = np.array([5.0, 3.0, 0.0])
SBTGM = np.array([-15.0, -3.0, 0.0])
HIT_TIME = 11.5
MISS_G = np.array([0.0, -3.0, 0.0])
MISS = 3.0
SBTG = np.array([-5.0, -3.0, 0.0])

INTERCEPT_FIELDS = {
    "write": ("int", "save", 1, ()),
    "miss": ("real", "diag", 0.0, ()),
    "hit_time": ("real", "diag", 0.0, ()),
    "MISS_G": ("vec", "diag", (0.0, 0.0, 0.0), ()),
    "time_m": ("real", "save", 0.0, ()),
    "SBTGM": ("vec", "save", (0.0, 0.0, 0.0), ()),
    "STMEG": ("vec", "save", (0.0, 0.0, 0.0), ()),
    "SBMEG": ("vec", "save", (0.0, 0.0, 0.0), ()),
    "halt": ("int", "data", 0, ()),
}

NOT_DEFINED = (
    "time",
    "alt",
    "dvbe",
    "psivgx",
    "thtvgx",
    "ground_range",
    "sbeg",
    "SBEG",
    "phimvx",
    "mguidance",
    "wp_lonx",
    "wp_latx",
    "wp_alt",
    "SWBG",
    "wp_flag",
    "mseeker",
    "range_go",
    "STBG",
    "closing_speed",
    "targ_com_slot",
    "stop_run",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()
        self.health = 1


def _ctx(combus=None, int_step=INT_STEP, vehicle_slot=0):
    if combus is None:
        combus = [
            Packet(name="RR3X", type="HYPER5", status=1, vars={}),
            Packet(
                name="Truck",
                type="TARGET3",
                status=1,
                vars={"sbeg": STEG.copy()},
            ),
        ]
    return SimContext(
        sim_time=TIME,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=vehicle_slot,
    )


def _plant(
    store,
    *,
    time=TIME,
    alt=2400.0,
    sbeg=None,
    mseeker=0,
    range_go=5000.0,
    stbg=None,
    closing_speed=100.0,
    targ_com_slot=1,
):
    if sbeg is None:
        sbeg = SBEG
    if stbg is None:
        stbg = STBG
    store.define(Field("time", time, "real", "exec", "environment"))
    store.define(Field("alt", alt, "real", "init/out", "newton"))
    store.define(Field("sbeg", sbeg, "vec", "state", "newton"))
    store.define(Field("mseeker", mseeker, "int", "data/save", "seeker"))
    store.define(Field("range_go", range_go, "real", "out", "seeker"))
    store.define(Field("STBG", stbg, "vec", "out", "seeker"))
    store.define(Field("closing_speed", closing_speed, "real", "out", "seeker"))
    store.define(Field("targ_com_slot", targ_com_slot, "int", "save", "seeker"))


def _ready(
    *,
    halt=0,
    write=1,
    time_m=TIME_M,
    sbtgm=None,
    stmeg=None,
    sbmeg=None,
    alt=2400.0,
    mseeker=0,
    range_go=5000.0,
    closing_speed=100.0,
    combus=None,
    int_step=INT_STEP,
):
    if sbtgm is None:
        sbtgm = SBTGM
    if stmeg is None:
        stmeg = STMEG
    if sbmeg is None:
        sbmeg = SBMEG
    vehicle = _Vehicle()
    intercept = Hyper5Intercept()
    intercept.define(vehicle)
    store = vehicle.store
    store.set("halt", halt)
    store.set("write", write)
    store.set("time_m", time_m)
    store.set("SBTGM", sbtgm)
    store.set("STMEG", stmeg)
    store.set("SBMEG", sbmeg)
    _plant(
        store,
        alt=alt,
        mseeker=mseeker,
        range_go=range_go,
        closing_speed=closing_speed,
    )
    return vehicle, intercept, _ctx(combus=combus, int_step=int_step)


def test_name_is_intercept():
    assert Hyper5Intercept().name == "intercept"


def test_define_registers_def_intercept_fields():
    vehicle = _Vehicle()
    Hyper5Intercept().define(vehicle)
    store = vehicle.store
    assert list(INTERCEPT_FIELDS) == store.names()
    for name, (ftype, role, default, outputs) in INTERCEPT_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "intercept"
        assert field.outputs == outputs
        if ftype == "int":
            assert store.get(name) == default
            assert type(store.get(name)) is int
        elif ftype == "real":
            assert store.get(name) == default
        else:
            np.testing.assert_array_equal(store.get(name), np.zeros(3))


def test_define_does_not_register_plant_seeker_or_guidance():
    vehicle = _Vehicle()
    Hyper5Intercept().define(vehicle)
    for name in NOT_DEFINED:
        with pytest.raises(KeyError):
            vehicle.store.get(name)


def test_write_default_is_one():
    vehicle = _Vehicle()
    Hyper5Intercept().define(vehicle)
    assert vehicle.store.get("write") == 1
    assert type(vehicle.store.get("write")) is int


def test_initialize_is_pass():
    vehicle = _Vehicle()
    intercept = Hyper5Intercept()
    intercept.define(vehicle)
    intercept.initialize(vehicle, _ctx())
    assert vehicle.store.get("write") == 1
    assert vehicle.store.get("halt") == 0
    assert vehicle.health == 1


def test_terminate_exists_and_is_pass():
    vehicle = _Vehicle()
    intercept = Hyper5Intercept()
    intercept.define(vehicle)
    intercept.terminate(vehicle, _ctx())
    assert vehicle.store.get("write") == 1
    assert vehicle.health == 1


def test_halt_1_write_1_sets_health_and_packet_status_0():
    vehicle, intercept, ctx = _ready(halt=1, write=1)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert vehicle.store.get("write") == 0
    assert type(vehicle.store.get("write")) is int
    assert ctx.combus[1].status == 1


def test_halt_0_write_1_does_not_kill():
    vehicle, intercept, ctx = _ready(halt=0, write=1)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 1


def test_halt_1_write_0_does_not_kill():
    vehicle, intercept, ctx = _ready(halt=1, write=0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 0


def test_ground_alt_le_0_write_1_sets_health_and_packet_status_0():
    vehicle, intercept, ctx = _ready(halt=0, write=1, alt=0.0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert vehicle.store.get("write") == 0


def test_ground_alt_negative_write_1_kills():
    vehicle, intercept, ctx = _ready(halt=0, write=1, alt=-1.0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert vehicle.store.get("write") == 0


def test_ground_alt_positive_does_not_kill():
    vehicle, intercept, ctx = _ready(halt=0, write=1, alt=0.1)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 1


def test_ground_alt_le_0_write_0_does_not_kill():
    vehicle, intercept, ctx = _ready(halt=0, write=0, alt=0.0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 0


def test_mseeker_3_range_go_under_1000_closing_negative_interpolates_and_kills():
    vehicle, intercept, ctx = _ready(
        halt=0,
        write=1,
        mseeker=3,
        range_go=500.0,
        closing_speed=-1.0,
    )
    intercept.execute(vehicle, ctx)
    store = vehicle.store
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert ctx.combus[1].status == 1
    assert store.get("write") == 0
    assert store.get("hit_time") == pytest.approx(HIT_TIME, rel=RTOL, abs=ATOL)
    assert store.get("miss") == pytest.approx(MISS, rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(store.get("MISS_G"), MISS_G, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBTGM"), SBTG, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("STMEG"), STEG, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBMEG"), SBEG, rtol=RTOL, atol=ATOL)
    assert store.get("time_m") == pytest.approx(TIME, rel=RTOL, abs=ATOL)


def test_mseeker_3_closing_nonnegative_updates_previous_without_kill():
    vehicle, intercept, ctx = _ready(
        halt=0,
        write=1,
        mseeker=3,
        range_go=500.0,
        closing_speed=0.0,
    )
    intercept.execute(vehicle, ctx)
    store = vehicle.store
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert store.get("write") == 1
    assert store.get("miss") == 0.0
    assert store.get("hit_time") == 0.0
    np.testing.assert_array_equal(store.get("MISS_G"), np.zeros(3))
    np.testing.assert_allclose(store.get("SBTGM"), SBTG, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("STMEG"), STEG, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBMEG"), SBEG, rtol=RTOL, atol=ATOL)
    assert store.get("time_m") == pytest.approx(TIME, rel=RTOL, abs=ATOL)


def test_mseeker_3_range_go_1000_does_not_update_previous_or_kill():
    vehicle, intercept, ctx = _ready(
        halt=0,
        write=1,
        mseeker=3,
        range_go=1000.0,
        closing_speed=-1.0,
        time_m=TIME_M,
    )
    intercept.execute(vehicle, ctx)
    store = vehicle.store
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert store.get("write") == 1
    np.testing.assert_allclose(store.get("SBTGM"), SBTGM, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("STMEG"), STMEG, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBMEG"), SBMEG, rtol=RTOL, atol=ATOL)
    assert store.get("time_m") == pytest.approx(TIME_M, rel=RTOL, abs=ATOL)


def test_mseeker_not_3_does_not_intercept():
    vehicle, intercept, ctx = _ready(
        halt=0,
        write=1,
        mseeker=1,
        range_go=500.0,
        closing_speed=-1.0,
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 1


def test_closing_negative_write_0_updates_previous_without_kill():
    vehicle, intercept, ctx = _ready(
        halt=0,
        write=0,
        mseeker=3,
        range_go=500.0,
        closing_speed=-1.0,
    )
    intercept.execute(vehicle, ctx)
    store = vehicle.store
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert store.get("write") == 0
    np.testing.assert_allclose(store.get("SBTGM"), SBTG, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("STMEG"), STEG, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBMEG"), SBEG, rtol=RTOL, atol=ATOL)
    assert store.get("time_m") == pytest.approx(TIME, rel=RTOL, abs=ATOL)


def test_target_geographic_position_from_combus_sbeg():
    steg = np.array([25.0, 4.0, 0.0])
    stmeg = np.array([25.0, 4.0, 0.0])
    sbtgm = np.array([-15.0, -4.0, 0.0])
    combus = [
        Packet(name="RR3X", type="HYPER5", status=1, vars={}),
        Packet(name="Truck", type="TARGET3", status=1, vars={"sbeg": steg.copy()}),
    ]
    vehicle, intercept, ctx = _ready(
        halt=0,
        write=1,
        mseeker=3,
        range_go=500.0,
        closing_speed=-1.0,
        stmeg=stmeg,
        sbtgm=sbtgm,
        combus=combus,
    )
    intercept.execute(vehicle, ctx)
    # Same interpolation as HIT_TIME; miss east component is 4 m
    assert vehicle.store.get("hit_time") == pytest.approx(HIT_TIME, rel=RTOL, abs=ATOL)
    assert vehicle.store.get("miss") == pytest.approx(4.0, rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(
        vehicle.store.get("MISS_G"), np.array([0.0, -4.0, 0.0]), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(vehicle.store.get("STMEG"), steg, rtol=RTOL, atol=ATOL)


def test_always_writes_write_back_when_unchanged():
    vehicle, intercept, ctx = _ready(halt=0, write=1, alt=2400.0)
    intercept.execute(vehicle, ctx)
    assert vehicle.store.get("write") == 1
    assert type(vehicle.store.get("write")) is int


def test_does_not_sys_exit_on_halt():
    vehicle, intercept, ctx = _ready(halt=1, write=1)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
