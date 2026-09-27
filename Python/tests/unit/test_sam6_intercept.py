from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr, polar_from_cart
from cadac.vehicles.flat6.sam6.intercept import Sam6Intercept

RTOL = 1e-12
ATOL = 1e-14

# Closest-approach fixture (hand-derived from C++ linear interpolation).
# Integer-friendly steps so IEEE doubles match the exact miss [0, -3, 0].
TIME = 11.0
TIME_M = 10.0
INT_STEP = 1.0
SBEL = np.array([20.0, 0.0, 0.0], dtype=float)
SBMEL = np.array([10.0, 0.0, 0.0], dtype=float)
STEL = np.array([25.0, 3.0, 0.0], dtype=float)
STMEL = np.array([25.0, 3.0, 0.0], dtype=float)
SBTLM = np.array([-15.0, -3.0, 0.0], dtype=float)
HIT_TIME = 11.5
MISS_L = np.array([0.0, -3.0, 0.0], dtype=float)
MISS_MAG = 3.0
SBTL = np.array([-5.0, -3.0, 0.0], dtype=float)
# Receding: range-rate UTBL·(VTEL-VBEL) > 0 so C++ closest-approach fires.
VBEL_RECEDE = np.array([-10.0, 0.0, 0.0], dtype=float)
VTEL_STILL = np.array([0.0, 0.0, 0.0], dtype=float)
VBEL_CLOSE = np.array([10.0, 0.0, 0.0], dtype=float)
DBT = sqrt(34.0)
MODE_RF_LOCK = 140000
ZEROS3 = (0.0, 0.0, 0.0)

DEFINED = (
    "mterm",
    "write",
    "miss",
    "hit_time",
    "MISS",
    "time_m",
    "SBTLM",
    "STMEL",
    "SBMEL",
    "mode",
    "dbt",
    "psiptx",
    "thtptx",
)
ROLES = {
    "mterm": "data",
    "write": "init",
    "miss": "diag",
    "hit_time": "diag",
    "MISS": "diag",
    "time_m": "save",
    "SBTLM": "save",
    "STMEL": "save",
    "SBMEL": "save",
    "mode": "diag",
    "dbt": "diag",
    "psiptx": "diag/data",
    "thtptx": "diag/data",
}
OUTPUTS = {
    "mterm": (),
    "write": (),
    "miss": ("plot",),
    "hit_time": (),
    "MISS": ("plot",),
    "time_m": (),
    "SBTLM": (),
    "STMEL": (),
    "SBMEL": (),
    "mode": ("scrn", "plot"),
    "dbt": ("scrn", "plot"),
    "psiptx": ("plot",),
    "thtptx": ("plot",),
}
INT_FIELDS = ("mterm", "write", "mode")
VEC_FIELDS = ("MISS", "SBTLM", "STMEL", "SBMEL")
NOT_DEFINED = (
    "time",
    "stop",
    "lconv",
    "TBL",
    "SBEL",
    "VBEL",
    "alt",
    "hbe",
    "dvbe",
    "psivlx",
    "thtvlx",
    "pdynmc",
    "STEL",
    "VTEL",
    "tgt_slot",
    "mseek",
    "dta",
    "mguide",
    "ip_sltrange",
    "SIBLC",
    "maut",
    "mprop",
    "trcond",
    "halt",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()
        self.health = 1


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ctx(combus=None, int_step=INT_STEP, vehicle_slot=0):
    if combus is None:
        combus = [
            Packet(name="SAM", type="MISSILE6", status=1, vars={}),
            Packet(name="A1", type="AIRCRAFT3", status=1, vars={}),
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
    stop=0,
    alt=1000.0,
    hbe=1000.0,
    sbel=None,
    vbel=None,
    stel=None,
    vtel=None,
    tgt_slot=1,
    mseek=0,
    trcond=0,
    mguide=0,
    ip_sltrange=999999.0,
    siblc=None,
    maut=0,
    mprop=0,
):
    if sbel is None:
        sbel = np.zeros(3)
    if vbel is None:
        vbel = np.zeros(3)
    if stel is None:
        stel = np.zeros(3)
    if vtel is None:
        vtel = np.zeros(3)
    if siblc is None:
        siblc = np.zeros(3)
    store.define(Field("time", time, "real", "exec", "kinematics"))
    store.define(Field("stop", stop, "int", "exec", "kinematics"))
    store.define(Field("alt", alt, "real", "out", "newton"))
    store.define(Field("hbe", hbe, "real", "out", "newton"))
    store.define(Field("SBEL", sbel, "vec", "state", "newton"))
    store.define(Field("VBEL", vbel, "vec", "out", "newton"))
    store.define(Field("STEL", stel, "vec", "out", "sensor"))
    store.define(Field("VTEL", vtel, "vec", "out", "sensor"))
    store.define(Field("tgt_slot", tgt_slot, "int", "out", "sensor"))
    store.define(Field("mseek", mseek, "int", "data", "sensor"))
    store.define(Field("trcond", trcond, "int", "diag", "aerodynamics"))
    store.define(Field("mguide", mguide, "int", "data", "guidance"))
    store.define(Field("ip_sltrange", ip_sltrange, "real", "diag", "guidance"))
    store.define(Field("SIBLC", siblc, "vec", "out", "guidance"))
    store.define(Field("maut", maut, "int", "data", "control"))
    store.define(Field("mprop", mprop, "int", "data", "propulsion"))


def _ready(
    *,
    mterm=0,
    write=1,
    time_m=TIME_M,
    sbtlm=None,
    stmel=None,
    sbmel=None,
    psiptx=0.0,
    thtptx=0.0,
    stop=0,
    trcond=0,
    alt=1000.0,
    hbe=1000.0,
    sbel=None,
    vbel=None,
    stel=None,
    vtel=None,
    mseek=0,
    mguide=0,
    ip_sltrange=999999.0,
    siblc=None,
    tgt_slot=1,
    combus=None,
    int_step=INT_STEP,
):
    if sbtlm is None:
        sbtlm = SBTLM
    if stmel is None:
        stmel = STMEL
    if sbmel is None:
        sbmel = SBMEL
    vehicle = _Vehicle()
    intercept = Sam6Intercept()
    intercept.define(vehicle)
    store = vehicle.store
    store.set("mterm", mterm)
    store.set("write", write)
    store.set("time_m", time_m)
    store.set("SBTLM", sbtlm)
    store.set("STMEL", stmel)
    store.set("SBMEL", sbmel)
    store.set("psiptx", psiptx)
    store.set("thtptx", thtptx)
    _plant(
        store,
        stop=stop,
        trcond=trcond,
        alt=alt,
        hbe=hbe,
        sbel=sbel,
        vbel=vbel,
        stel=stel,
        vtel=vtel,
        mseek=mseek,
        mguide=mguide,
        ip_sltrange=ip_sltrange,
        siblc=siblc,
        tgt_slot=tgt_slot,
    )
    return vehicle, intercept, _ctx(combus=combus, int_step=int_step)


def _mterm1_replica(stel, sbel, vtel, vbel, sbtlm, stmel, sbmel, time_m, int_step):
    stbl = stel - sbel
    sbtl = -stbl
    sbbml = sbel - sbmel
    sttml = stel - stmel
    hit_time = time_m - int_step * float((sbbml - sttml) @ sbtlm) / float(
        sbbml @ sbbml
    )
    vtbel = vtel - vbel
    polar = polar_from_cart(vtel)
    ttl = mat2tr(float(polar[1]), float(polar[2]))
    vtbet = ttl @ vtbel
    vbtet = -vtbet
    polar_asp = polar_from_cart(vbtet)
    psiptx = float(polar_asp[1]) * DEG
    thtptx = float(polar_asp[2]) * DEG - 90.0
    tpt = mat2tr(psiptx * RAD, thtptx * RAD)
    tpl = tpt @ ttl
    sbtp = tpl @ sbtl
    sbbmp = tpl @ sbbml
    stbmp = sbbmp - sbtp
    ww = float(stbmp[2] / sbbmp[2])
    miss_vec = sbbmp * ww - stbmp
    miss = float(np.sqrt(miss_vec[0] ** 2 + miss_vec[1] ** 2 + miss_vec[2] ** 2))
    return hit_time, miss_vec, miss, psiptx, thtptx


def test_name_is_intercept():
    assert Sam6Intercept().name == "intercept"


def test_define_cpp_fields():
    vehicle = _Vehicle()
    Sam6Intercept().define(vehicle)
    store = vehicle.store
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        assert field.module == "intercept"
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        if name in INT_FIELDS:
            assert field.type == "int"
            if name == "write":
                assert store.get(name) == 1
            else:
                assert store.get(name) == 0
            assert type(store.get(name)) is int
        elif name in VEC_FIELDS:
            assert field.type == "vec"
            np.testing.assert_array_equal(store.get(name), np.zeros(3))
        else:
            assert field.type == "real"
            assert store.get(name) == 0.0


def test_define_does_not_register_plant_names():
    vehicle = _Vehicle()
    Sam6Intercept().define(vehicle)
    for name in NOT_DEFINED:
        with pytest.raises(KeyError):
            vehicle.store.get(name)


def test_write_default_is_one():
    vehicle = _Vehicle()
    Sam6Intercept().define(vehicle)
    assert vehicle.store.get("write") == 1
    assert type(vehicle.store.get("write")) is int


def test_initialize_and_terminate_are_pass():
    vehicle, intercept, ctx = _ready()
    assert intercept.initialize(vehicle, ctx) is None
    assert intercept.terminate(vehicle, ctx) is None
    assert vehicle.store.get("write") == 1
    assert vehicle.health == 1


def test_halt_stop_1_trcond_4_write_1_sets_health_and_packet_status_0():
    vehicle, intercept, ctx = _ready(stop=1, trcond=4, write=1)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert ctx.combus[1].status == 1


def test_stop_0_does_not_kill():
    vehicle, intercept, ctx = _ready(stop=0, trcond=4, write=1)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 1


def test_stop_1_trcond_0_does_not_kill():
    vehicle, intercept, ctx = _ready(stop=1, trcond=0, write=1)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1


def test_ground_alt_negative_write_1_kills():
    vehicle, intercept, ctx = _ready(stop=0, write=1, alt=-1.0, hbe=1000.0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert vehicle.store.get("write") == 0
    assert ctx.combus[1].status == 1


def test_ground_alt_zero_write_1_kills():
    vehicle, intercept, ctx = _ready(stop=0, write=1, alt=0.0, hbe=0.0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert vehicle.store.get("write") == 0


def test_ground_alt_positive_does_not_kill():
    vehicle, intercept, ctx = _ready(stop=0, write=1, alt=0.1, hbe=0.1)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 1


def test_ground_alt_le_0_write_0_does_not_kill():
    vehicle, intercept, ctx = _ready(stop=0, write=0, alt=-1.0, hbe=-1.0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 0


def test_ground_hbe_le_0_with_positive_alt_kills():
    vehicle, intercept, ctx = _ready(stop=0, write=1, alt=1000.0, hbe=-1.0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert vehicle.store.get("write") == 0


def test_ip_sltrange_under_500_closing_negative_kills_missile_only():
    vehicle, intercept, ctx = _ready(
        stop=0,
        write=1,
        ip_sltrange=400.0,
        siblc=np.array([400.0, 0.0, 0.0], dtype=float),
        vbel=np.array([-16.0, 0.0, 0.0], dtype=float),
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert vehicle.store.get("write") == 0
    assert ctx.combus[1].status == 1


def test_ip_sltrange_under_500_closing_nonnegative_does_not_kill():
    vehicle, intercept, ctx = _ready(
        stop=0,
        write=1,
        ip_sltrange=400.0,
        siblc=np.array([400.0, 0.0, 0.0], dtype=float),
        vbel=np.array([16.0, 0.0, 0.0], dtype=float),
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 1


def test_ip_sltrange_500_does_not_kill():
    vehicle, intercept, ctx = _ready(
        stop=0,
        write=1,
        ip_sltrange=500.0,
        siblc=np.array([500.0, 0.0, 0.0], dtype=float),
        vbel=np.array([-16.0, 0.0, 0.0], dtype=float),
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 1


def test_ip_sltrange_zero_mguide_0_does_not_raise_or_ip_kill():
    vehicle, intercept, ctx = _ready(
        stop=0,
        write=1,
        alt=1000.0,
        hbe=1000.0,
        mguide=0,
        ip_sltrange=0.0,
        siblc=np.zeros(3),
        vbel=np.array([-16.0, 0.0, 0.0], dtype=float),
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 1
    assert vehicle.store.get("mguide") == 0


def test_mterm_0_lock_dbt_under_500_closing_positive_interpolates_and_kills_both():
    vehicle, intercept, ctx = _ready(
        mterm=0,
        write=1,
        mseek=14,
        sbel=SBEL,
        stel=STEL,
        vbel=VBEL_RECEDE,
        vtel=VTEL_STILL,
    )
    intercept.execute(vehicle, ctx)
    store = vehicle.store
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert ctx.combus[1].status == 0
    assert store.get("write") == 0
    assert _approx(store.get("hit_time"), HIT_TIME)
    assert _approx(store.get("miss"), MISS_MAG)
    np.testing.assert_allclose(store.get("MISS"), MISS_L, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBTLM"), SBTL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("STMEL"), STEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBMEL"), SBEL, rtol=RTOL, atol=ATOL)
    assert _approx(store.get("time_m"), TIME)
    assert _approx(store.get("dbt"), DBT)
    assert store.get("mode") == MODE_RF_LOCK


def test_skr_mode_4_closing_nonpositive_updates_previous_without_kill():
    vehicle, intercept, ctx = _ready(
        mterm=0,
        write=1,
        mseek=14,
        sbel=SBEL,
        stel=STEL,
        vbel=VBEL_CLOSE,
        vtel=VTEL_STILL,
    )
    intercept.execute(vehicle, ctx)
    store = vehicle.store
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert ctx.combus[1].status == 1
    assert store.get("write") == 1
    assert store.get("miss") == 0.0
    assert store.get("hit_time") == 0.0
    np.testing.assert_array_equal(store.get("MISS"), np.zeros(3))
    np.testing.assert_allclose(store.get("SBTLM"), SBTL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("STMEL"), STEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBMEL"), SBEL, rtol=RTOL, atol=ATOL)
    assert _approx(store.get("time_m"), TIME)


def test_skr_mode_not_4_does_not_intercept():
    vehicle, intercept, ctx = _ready(
        mterm=0,
        write=1,
        mseek=13,
        sbel=SBEL,
        stel=STEL,
        vbel=VBEL_RECEDE,
        vtel=VTEL_STILL,
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert ctx.combus[1].status == 1
    assert vehicle.store.get("write") == 1
    np.testing.assert_allclose(vehicle.store.get("SBTLM"), SBTLM, rtol=RTOL, atol=ATOL)


def test_dbt_500_does_not_update_previous_or_kill():
    stel = np.array([500.0, 0.0, 0.0], dtype=float)
    sbel = np.zeros(3)
    vehicle, intercept, ctx = _ready(
        mterm=0,
        write=1,
        mseek=14,
        sbel=sbel,
        stel=stel,
        vbel=VBEL_RECEDE,
        vtel=VTEL_STILL,
    )
    intercept.execute(vehicle, ctx)
    store = vehicle.store
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert store.get("write") == 1
    np.testing.assert_allclose(store.get("SBTLM"), SBTLM, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("STMEL"), STMEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBMEL"), SBMEL, rtol=RTOL, atol=ATOL)
    assert _approx(store.get("time_m"), TIME_M)
    assert _approx(store.get("dbt"), 500.0)


def test_mterm_1_miss_in_intercept_plane_kills_both():
    vehicle, intercept, ctx = _ready(
        mterm=1,
        write=1,
        mseek=14,
        sbel=SBEL,
        stel=STEL,
        vbel=VBEL_RECEDE,
        vtel=VTEL_STILL,
    )
    intercept.execute(vehicle, ctx)
    hit_time, miss_vec, miss, psiptx, thtptx = _mterm1_replica(
        STEL, SBEL, VTEL_STILL, VBEL_RECEDE, SBTLM, STMEL, SBMEL, TIME_M, INT_STEP
    )
    store = vehicle.store
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert ctx.combus[1].status == 0
    assert store.get("write") == 0
    assert _approx(store.get("hit_time"), hit_time)
    assert _approx(store.get("miss"), miss)
    np.testing.assert_allclose(store.get("MISS"), miss_vec, rtol=RTOL, atol=ATOL)
    assert _approx(store.get("psiptx"), psiptx)
    assert _approx(store.get("thtptx"), thtptx)


def test_mterm_2_raises():
    vehicle, intercept, ctx = _ready(mterm=2)
    with pytest.raises(ValueError, match="mterm"):
        intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1


def test_always_writes_write_back_when_unchanged():
    vehicle, intercept, ctx = _ready(stop=0, write=1, alt=1000.0)
    intercept.execute(vehicle, ctx)
    assert vehicle.store.get("write") == 1
    assert type(vehicle.store.get("write")) is int


def test_does_not_sys_exit_on_halt():
    vehicle, intercept, ctx = _ready(stop=1, trcond=4, write=1)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0


def test_no_sys_exit_or_print():
    import cadac.vehicles.flat6.sam6.intercept as mod

    text = Path(mod.__file__).read_text(encoding="utf-8")
    assert "sys.exit" not in text
    assert "print(" not in text


def test_no_flat6_hyper5_or_plane_imports():
    import cadac.vehicles.flat6.sam6.intercept as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "Plane5" not in src
    assert "Plane6" not in src
    assert "hyper5" not in src
    assert "Hyper5Intercept" not in src
