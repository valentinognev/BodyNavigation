import inspect
from math import sqrt
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import RAD
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr
from cadac.vehicles.agm6.intercept import Agm6Intercept

RTOL = 1e-12
ATOL = 1e-14
PLOT = ("plot",)
SCRN = ("scrn",)
SCRN_PLOT = ("scrn", "plot")
ZEROS3 = (0.0, 0.0, 0.0)

# C++ Missile::def_intercept order.
FIELDS = {
    "mterm": ("int", "data", 0, ()),
    "write": ("int", "init", 1, ()),
    "miss": ("real", "diag", 0.0, PLOT),
    "hit_time": ("real", "diag", 0.0, ()),
    "MISS_P": ("vec", "diag", ZEROS3, PLOT),
    "time_m": ("real", "save", 0.0, ()),
    "SBMTP": ("vec", "save", ZEROS3, ()),
    "mode": ("int", "diag", 0, SCRN),
    "dbt": ("real", "diag", 0.0, SCRN_PLOT),
    "psiplx": ("real", "/diag/data", 0.0, PLOT),
    "thtplx": ("real", "diag/data", 0.0, PLOT),
    "critmax": ("real", "diag/data", 100.0, PLOT),
}
DEFINED = tuple(FIELDS)
EXTERNALS = (
    "time",
    "halt",
    "stop",
    "lconv",
    "TBL",
    "SBEL",
    "VBEL",
    "dvbe",
    "hbe",
    "psivlx",
    "thtvlx",
    "tgt_num",
    "mprop",
    "trcond",
    "mseek",
    "mguid",
    "maut",
)

TIME = 12.0
TIME_M = 11.999
INT_STEP = 0.001
SBEL_AIR = np.array([0.0, 0.0, -7000.0], dtype=float)
SBEL_GROUND = np.array([100.0, 200.0, 1.0], dtype=float)
# Airborne (sbel3=-7000). Tilted plane thtplx=90 deg: TPL maps SBTL=[10,0,0] → SBTP=[0,0,10].
STEL_HIT = np.array([0.0, 0.0, -7000.0], dtype=float)
SBEL_HIT = np.array([10.0, 0.0, -7000.0], dtype=float)
SBMTP_HIT = np.array([0.0, 0.0, -10.0], dtype=float)
THTPLX_HIT = 90.0
DBT_HIT = 10.0

# Lateral miss with identity plane (psiplx=thtplx=0); alt=6990 so ground does not fire.
STEL_LAT = np.array([0.0, 0.0, -7000.0], dtype=float)
SBEL_LAT = np.array([3.0, 4.0, -6990.0], dtype=float)
SBMTP_LAT = np.array([3.0, 4.0, -10.0], dtype=float)
MISS_P_LAT = np.array([3.0, 4.0, 0.0], dtype=float)
MISS_LAT = 5.0
HIT_TIME_LAT = 0.5 * INT_STEP + TIME_M
DBT_LAT = sqrt(3.0 * 3.0 + 4.0 * 4.0 + 10.0 * 10.0)
SBTP_LAT = np.array([3.0, 4.0, 10.0], dtype=float)

MSEEK = 4
MGUID_MID4 = 40
MGUID_TERM5 = 5
MGUID_TERM6 = 6
MGUID_MID3 = 30
MAUT = 3
MPROP = 1
MODE_MID4 = 10000 * MSEEK + 1000 * 4 + 100 * 0 + 10 * MAUT + MPROP


def _ctx(combus=None, int_step=INT_STEP, vehicle_slot=0, sim_time=TIME):
    if combus is None:
        combus = _combus()
    return SimContext(
        sim_time=sim_time,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=vehicle_slot,
    )


def _packet(name, ptype, **vars_):
    return Packet(name=name, type=ptype, status=1, vars=dict(vars_))


def _combus(stel=None, extra_targets=None):
    if stel is None:
        stel = STEL_HIT.copy()
    missile = _packet("m1", "MISSILE6")
    target = _packet("t1", "TARGET3", SAEL=stel.copy(), VAEL=np.zeros(3))
    packets = [missile, target]
    if extra_targets:
        packets.extend(extra_targets)
    return packets


def _plant(
    store,
    *,
    time=TIME,
    sbel=None,
    tgt_num=1,
    mprop=MPROP,
    trcond=0,
    mseek=MSEEK,
    mguid=0,
    maut=MAUT,
    halt=None,
    stop=None,
):
    if sbel is None:
        sbel = SBEL_AIR
    store.define(Field("time", time, "real", "state", "newton"))
    store.define(Field("SBEL", sbel, "vec", "state", "newton"))
    store.define(Field("tgt_num", tgt_num, "int", "data", "sensor"))
    store.define(Field("mprop", mprop, "int", "data", "propulsion"))
    store.define(Field("trcond", trcond, "int", "init", "aerodynamics"))
    store.define(Field("mseek", mseek, "int", "data/diag", "sensor"))
    store.define(Field("mguid", mguid, "int", "data", "guidance"))
    store.define(Field("maut", maut, "int", "data", "control"))
    if halt is not None:
        store.define(Field("halt", halt, "int", "exec", "kinematics"))
    if stop is not None:
        store.define(Field("stop", stop, "int", "exec", "kinematics"))


def _ready(
    *,
    halt=0,
    stop=0,
    write=1,
    trcond=0,
    mguid=0,
    sbel=None,
    stel=None,
    time_m=TIME_M,
    sbmtp=None,
    psiplx=0.0,
    thtplx=0.0,
    tgt_num=1,
    combus=None,
    plant_halt=True,
    plant_stop=True,
    int_step=INT_STEP,
):
    if sbel is None:
        sbel = SBEL_AIR
    if sbmtp is None:
        sbmtp = np.zeros(3)
    vehicle = SimpleNamespace(store=StateStore(), health=1)
    intercept = Agm6Intercept()
    intercept.define(vehicle)
    store = vehicle.store
    store.set("write", write)
    store.set("time_m", time_m)
    store.set("SBMTP", sbmtp)
    store.set("psiplx", psiplx)
    store.set("thtplx", thtplx)
    _plant(
        store,
        sbel=sbel,
        tgt_num=tgt_num,
        trcond=trcond,
        mguid=mguid,
        halt=halt if plant_halt else None,
        stop=stop if plant_stop else None,
    )
    if combus is None:
        combus = _combus(stel=stel)
    return vehicle, intercept, _ctx(combus=combus, int_step=int_step)


def test_name_is_intercept():
    assert Agm6Intercept.name == "intercept"
    assert Agm6Intercept().name == "intercept"


def test_define_registers_cpp_def_intercept_fields():
    vehicle = SimpleNamespace(store=StateStore())
    Agm6Intercept().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    zeros3 = np.zeros(3)
    for name, (ftype, role, default, outputs) in FIELDS.items():
        field = store.field(name)
        assert field.module == "intercept"
        assert field.type == ftype
        assert field.role == role
        assert field.outputs == outputs
        if ftype == "int":
            assert store.get(name) == default
            assert type(store.get(name)) is int
        elif ftype == "real":
            assert store.get(name) == default
        else:
            np.testing.assert_array_equal(store.get(name), zeros3)
            assert store.get(name).shape == (3,)
    for name in EXTERNALS:
        assert name not in store.names()


def test_initialize_and_terminate_are_pass():
    vehicle, intercept, ctx = _ready()
    intercept.initialize(vehicle, ctx)
    intercept.terminate(vehicle, ctx)
    assert vehicle.health == 1
    assert vehicle.store.get("write") == 1
    assert ctx.combus[0].status == 1


def test_halt_1_write_1_sets_health_and_packet_status_0():
    vehicle, intercept, ctx = _ready(halt=1, write=1)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert ctx.combus[1].status == 1


def test_halt_0_sbel3_neg_7000_no_kill():
    vehicle, intercept, ctx = _ready(halt=0, sbel=SBEL_AIR)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 1
    np.testing.assert_allclose(vehicle.store.get("SBEL")[2], -7000.0, rtol=RTOL, atol=ATOL)


def test_ground_sbel3_plus_1_kills_once_write_0():
    vehicle, intercept, ctx = _ready(halt=0, write=1, sbel=SBEL_GROUND)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert vehicle.store.get("write") == 0
    assert type(vehicle.store.get("write")) is int

    vehicle.health = 1
    ctx.combus[ctx.vehicle_slot].status = 1
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 0


def test_halt_absent_treated_as_0():
    vehicle, intercept, ctx = _ready(plant_halt=False, stop=0, sbel=SBEL_AIR)
    assert "halt" not in vehicle.store.names()
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1


def test_trcond_and_stop_kills():
    vehicle, intercept, ctx = _ready(halt=0, trcond=4, stop=1, sbel=SBEL_AIR)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert ctx.combus[1].status == 1


def test_trcond_without_stop_does_not_kill():
    vehicle, intercept, ctx = _ready(halt=0, trcond=4, stop=0, sbel=SBEL_AIR)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1


def test_stop_absent_treated_as_0():
    vehicle, intercept, ctx = _ready(trcond=4, plant_stop=False, sbel=SBEL_AIR)
    assert "stop" not in vehicle.store.names()
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1


def test_target_plane_mguid_40_tilted_interpolates_and_kills():
    vehicle, intercept, ctx = _ready(
        halt=0,
        write=1,
        mguid=MGUID_MID4,
        sbel=SBEL_HIT,
        stel=STEL_HIT,
        sbmtp=SBMTP_HIT,
        thtplx=THTPLX_HIT,
        time_m=TIME_M,
    )
    intercept.execute(vehicle, ctx)
    store = vehicle.store
    tpl = mat2tr(0.0 * RAD, THTPLX_HIT * RAD)
    sbtp = tpl @ (SBEL_HIT - STEL_HIT)
    sbbmp = sbtp - SBMTP_HIT
    stbmp = SBMTP_HIT * (-1.0)
    dum = stbmp[2] / sbbmp[2]
    want_miss_p = sbbmp * dum - stbmp
    want_miss = sqrt(
        float(want_miss_p[0] ** 2 + want_miss_p[1] ** 2 + want_miss_p[2] ** 2)
    )
    want_hit_time = dum * INT_STEP + TIME_M
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert ctx.combus[1].status == 1
    assert store.get("write") == 0
    assert store.get("hit_time") == pytest.approx(want_hit_time, rel=RTOL, abs=ATOL)
    assert store.get("miss") == pytest.approx(want_miss, rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(store.get("MISS_P"), want_miss_p, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBMTP"), sbtp, rtol=RTOL, atol=ATOL)
    assert store.get("time_m") == pytest.approx(TIME, rel=RTOL, abs=ATOL)
    assert store.get("dbt") == pytest.approx(DBT_HIT, rel=RTOL, abs=ATOL)
    assert store.get("mode") == MODE_MID4
    # Identity TPL would leave sbtp3==0 and not kill; tilt must be applied.
    assert sbtp[2] > 0


def test_target_plane_mguid_6_identity_lateral_miss():
    vehicle, intercept, ctx = _ready(
        halt=0,
        write=1,
        mguid=MGUID_TERM6,
        sbel=SBEL_LAT,
        stel=STEL_LAT,
        sbmtp=SBMTP_LAT,
        psiplx=0.0,
        thtplx=0.0,
        time_m=TIME_M,
    )
    intercept.execute(vehicle, ctx)
    store = vehicle.store
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert store.get("write") == 0
    assert store.get("miss") == pytest.approx(MISS_LAT, rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(store.get("MISS_P"), MISS_P_LAT, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBMTP"), SBTP_LAT, rtol=RTOL, atol=ATOL)
    assert store.get("hit_time") == pytest.approx(HIT_TIME_LAT, rel=RTOL, abs=ATOL)
    assert store.get("dbt") == pytest.approx(DBT_LAT, rel=RTOL, abs=ATOL)


def test_target_plane_mguid_5_kills():
    vehicle, intercept, ctx = _ready(
        halt=0,
        write=1,
        mguid=MGUID_TERM5,
        sbel=SBEL_HIT,
        stel=STEL_HIT,
        sbmtp=SBMTP_HIT,
        thtplx=THTPLX_HIT,
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert vehicle.store.get("write") == 0


def test_target_plane_mguid_30_does_not_hit():
    vehicle, intercept, ctx = _ready(
        halt=0,
        write=1,
        mguid=MGUID_MID3,
        sbel=SBEL_HIT,
        stel=STEL_HIT,
        sbmtp=SBMTP_HIT,
        thtplx=THTPLX_HIT,
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 1
    assert vehicle.store.get("miss") == 0.0
    np.testing.assert_allclose(vehicle.store.get("SBMTP"), SBMTP_HIT, rtol=RTOL, atol=ATOL)


def test_target_plane_dbt_not_under_100_does_not_hit():
    sbel = np.array([200.0, 0.0, -7000.0], dtype=float)
    vehicle, intercept, ctx = _ready(
        halt=0,
        write=1,
        mguid=MGUID_MID4,
        sbel=sbel,
        stel=STEL_HIT,
        sbmtp=SBMTP_HIT,
        thtplx=THTPLX_HIT,
        time_m=TIME_M,
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 1
    np.testing.assert_allclose(vehicle.store.get("SBMTP"), SBMTP_HIT, rtol=RTOL, atol=ATOL)
    assert vehicle.store.get("time_m") == pytest.approx(TIME_M, rel=RTOL, abs=ATOL)
    assert vehicle.store.get("dbt") == pytest.approx(200.0, rel=RTOL, abs=ATOL)


def test_target_plane_sbtp3_not_positive_saves_previous_without_kill():
    sbel = np.array([-10.0, 0.0, -7000.0], dtype=float)
    sbtp = mat2tr(0.0 * RAD, THTPLX_HIT * RAD) @ (sbel - STEL_HIT)
    assert sbtp[2] < 0
    vehicle, intercept, ctx = _ready(
        halt=0,
        write=1,
        mguid=MGUID_MID4,
        sbel=sbel,
        stel=STEL_HIT,
        sbmtp=SBMTP_HIT,
        thtplx=THTPLX_HIT,
        time_m=TIME_M,
    )
    intercept.execute(vehicle, ctx)
    store = vehicle.store
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert store.get("write") == 1
    assert store.get("miss") == 0.0
    np.testing.assert_allclose(store.get("SBMTP"), sbtp, rtol=RTOL, atol=ATOL)
    assert store.get("time_m") == pytest.approx(TIME, rel=RTOL, abs=ATOL)


def test_target_from_type_TARGET3_and_tgt_num_not_id():
    decoy = _packet(
        "t1",
        "AIRCRAFT3",
        SAEL=np.array([999.0, 999.0, 999.0], dtype=float),
    )
    first = _packet(
        "later",
        "TARGET3",
        SAEL=np.array([50.0, 0.0, 0.0], dtype=float),
        VAEL=np.zeros(3),
    )
    second = _packet(
        "t1",
        "TARGET3",
        SAEL=STEL_HIT.copy(),
        VAEL=np.zeros(3),
    )
    combus = [_packet("m1", "MISSILE6"), decoy, first, second]
    vehicle, intercept, ctx = _ready(
        halt=0,
        write=1,
        mguid=MGUID_MID4,
        sbel=SBEL_HIT,
        sbmtp=SBMTP_HIT,
        thtplx=THTPLX_HIT,
        tgt_num=2,
        combus=combus,
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert vehicle.store.get("dbt") == pytest.approx(DBT_HIT, rel=RTOL, abs=ATOL)


def test_sbel_fallback_when_sael_absent():
    combus = [
        _packet("m1", "MISSILE6"),
        _packet("t1", "TARGET3", SBEL=STEL_HIT.copy(), VBEL=np.zeros(3)),
    ]
    vehicle, intercept, ctx = _ready(
        halt=0,
        write=1,
        mguid=MGUID_MID4,
        sbel=SBEL_HIT,
        sbmtp=SBMTP_HIT,
        thtplx=THTPLX_HIT,
        combus=combus,
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert vehicle.store.get("dbt") == pytest.approx(DBT_HIT, rel=RTOL, abs=ATOL)


def test_does_not_sys_exit_or_print():
    src = inspect.getsource(Agm6Intercept)
    assert "sys.exit" not in src
    assert "print(" not in src
    vehicle, intercept, ctx = _ready(halt=1, write=1)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
