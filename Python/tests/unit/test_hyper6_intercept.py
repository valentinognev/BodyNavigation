"""HYPER6 intercept — Task 13: miss / closest approach (Hyper::intercept)."""

from math import sqrt
from unittest.mock import patch

import numpy as np
import pytest

from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import cadac_matmul, skew
from cadac.vehicles.round6.hyper6.intercept import Hyper6Intercept
from cadac.vehicles.round6.hyper6.vehicle import Hyper6

RTOL = 1e-12
ATOL = 1e-14

# Closest-approach fixture (hand-derived from C++ linear interpolation).
TIME = 11.0
TIME_M = 10.0
INT_STEP = 1.0
SBII = np.array([20.0, 0.0, 0.0])
SBMII = np.array([10.0, 0.0, 0.0])
STII = np.array([25.0, 3.0, 0.0])
STMII = np.array([25.0, 3.0, 0.0])
SBTIM = np.array([-15.0, -3.0, 0.0])
HIT_TIME = 11.5
MISS_I = np.array([0.0, -3.0, 0.0])
MISS = 3.0
VTII = np.array([5.0, 3.0, 0.0])
VBII = np.zeros(3)

PLOT = ("plot",)
SCRN_PLOT = ("scrn", "plot")

DEFINED = {
    "mintercept": ("int", "data", 0, ()),
    "write": ("int", "init", 1, ()),
    "miss": ("real", "diag", 0.0, PLOT),
    "hit_time": ("real", "diag", 0.0, ()),
    "MISS_I": ("vec", "diag", (0.0, 0.0, 0.0), PLOT),
    "time_m": ("real", "save", 0.0, ()),
    "SBTIM": ("vec", "save", (0.0, 0.0, 0.0), ()),
    "STMII": ("vec", "save", (0.0, 0.0, 0.0), ()),
    "SBMII": ("vec", "save", (0.0, 0.0, 0.0), ()),
    "event": ("int", "diag", 0, SCRN_PLOT),
    "dbt": ("real", "diag", 0.0, SCRN_PLOT),
    "MISS_H": ("vec", "diag", (0.0, 0.0, 0.0), PLOT),
    "MISS_L": ("vec", "diag", (0.0, 0.0, 0.0), PLOT),
}

NOT_DEFINED = (
    "time",
    "TBI",
    "alt",
    "dvbe",
    "dvbi",
    "psivdx",
    "thtvdx",
    "SBII",
    "VBII",
    "headon_flag",
    "sat_num",
    "STII",
    "VTII",
    "mprop",
    "mrcs_moment",
    "mseek",
    "mguide",
    "wp_lonx",
    "wp_latx",
    "wp_alt",
    "SWBD",
    "wp_flag",
    "maut",
)


def _univec3(vec):
    v = np.asarray(vec, dtype=float)
    scale = sqrt(float(v @ v))
    if scale == 0.0:
        return np.zeros(3)
    return v / scale


def _expected_miss_h(stii, vtii, miss_i):
    uh1 = _univec3(stii)
    uh3 = _univec3(cadac_matmul(skew(stii), vtii))
    uh2 = cadac_matmul(skew(uh3), uh1)
    thi = np.vstack((uh1, uh2, uh3))
    return cadac_matmul(thi, miss_i)


def _expected_miss_l(stii, vtii, miss_i):
    ul1i = _univec3(vtii)
    ul3i = _univec3(stii) * (-1.0)
    ul2i = cadac_matmul(skew(ul3i), ul1i)
    tli = np.vstack((ul1i, ul2i, ul3i))
    return cadac_matmul(tli, miss_i)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()
        self.health = 1


def _ctx(combus="default", int_step=INT_STEP, vehicle_slot=0):
    if combus == "default":
        combus = [Packet(name="Hypersonic", type="HYPER6", status=1, vars={})]
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
    alt=10000.0,
    mguide=0,
    maut=0,
    mprop=0,
    mseek=0,
    mrcs_moment=0,
    wp_alt=0.0,
    wp_flag=0,
    swbd=None,
    sbii=None,
    vbii=None,
    stii=None,
    vtii=None,
    headon_flag=0,
    sat_num=1,
    dvbe=1000.0,
    dvbi=1000.0,
    psivdx=0.0,
    thtvdx=0.0,
):
    if swbd is None:
        swbd = np.zeros(3)
    if sbii is None:
        sbii = np.zeros(3)
    if vbii is None:
        vbii = np.zeros(3)
    if stii is None:
        stii = np.zeros(3)
    if vtii is None:
        vtii = np.zeros(3)
    for name, value, ftype in (
        ("time", time, "real"),
        ("TBI", np.eye(3), "mat"),
        ("alt", alt, "real"),
        ("dvbe", dvbe, "real"),
        ("dvbi", dvbi, "real"),
        ("psivdx", psivdx, "real"),
        ("thtvdx", thtvdx, "real"),
        ("SBII", sbii, "vec"),
        ("VBII", vbii, "vec"),
        ("headon_flag", headon_flag, "int"),
        ("sat_num", sat_num, "int"),
        ("STII", stii, "vec"),
        ("VTII", vtii, "vec"),
        ("mprop", mprop, "int"),
        ("mrcs_moment", mrcs_moment, "int"),
        ("mseek", mseek, "int"),
        ("mguide", mguide, "int"),
        ("wp_lonx", 0.0, "real"),
        ("wp_latx", 0.0, "real"),
        ("wp_alt", wp_alt, "real"),
        ("SWBD", swbd, "vec"),
        ("wp_flag", wp_flag, "int"),
        ("maut", maut, "int"),
    ):
        store.define(Field(name, value, ftype, "data", "plant"))


def _ready(
    *,
    write=1,
    time_m=0.0,
    sbtim=None,
    stmii=None,
    sbmii=None,
    alt=10000.0,
    mguide=0,
    maut=0,
    mprop=0,
    mseek=0,
    mrcs_moment=0,
    wp_alt=0.0,
    wp_flag=0,
    swbd=None,
    sbii=None,
    vbii=None,
    stii=None,
    vtii=None,
    headon_flag=0,
    combus="default",
    int_step=INT_STEP,
):
    if sbtim is None:
        sbtim = np.zeros(3)
    if stmii is None:
        stmii = np.zeros(3)
    if sbmii is None:
        sbmii = np.zeros(3)
    vehicle = _Vehicle()
    intercept = Hyper6Intercept()
    intercept.define(vehicle)
    store = vehicle.store
    store.set("write", write)
    store.set("time_m", time_m)
    store.set("SBTIM", sbtim)
    store.set("STMII", stmii)
    store.set("SBMII", sbmii)
    _plant(
        store,
        alt=alt,
        mguide=mguide,
        maut=maut,
        mprop=mprop,
        mseek=mseek,
        mrcs_moment=mrcs_moment,
        wp_alt=wp_alt,
        wp_flag=wp_flag,
        swbd=swbd,
        sbii=sbii,
        vbii=vbii,
        stii=stii,
        vtii=vtii,
        headon_flag=headon_flag,
    )
    return vehicle, intercept, _ctx(combus=combus, int_step=int_step)


def test_name_is_intercept():
    assert Hyper6Intercept().name == "intercept"


def test_define_registers_miss_hit_diagnostics():
    vehicle = _Vehicle()
    Hyper6Intercept().define(vehicle)
    store = vehicle.store
    assert store.names() == list(DEFINED)
    for name, (ftype, role, default, outputs) in DEFINED.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "intercept"
        assert field.outputs == outputs
        got = store.get(name)
        if ftype == "vec":
            assert np.allclose(got, default)
        else:
            assert got == default
            assert type(got) is type(default)


def test_define_does_not_register_inputs():
    vehicle = _Vehicle()
    Hyper6Intercept().define(vehicle)
    for name in NOT_DEFINED:
        assert name not in vehicle.store.names()


def test_initialize_and_terminate_are_pass():
    vehicle = _Vehicle()
    intercept = Hyper6Intercept()
    intercept.define(vehicle)
    intercept.initialize(vehicle, _ctx())
    intercept.terminate(vehicle, _ctx())
    assert vehicle.store.get("write") == 1
    assert vehicle.health == 1


def test_alt_positive_health_stays_1():
    vehicle, intercept, ctx = _ready(alt=100.0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert vehicle.store.get("write") == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1


def test_ground_impact_kills():
    vehicle, intercept, ctx = _ready(alt=-1.0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert vehicle.store.get("write") == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0


def test_alt_0_kills():
    vehicle, intercept, ctx = _ready(alt=0.0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert vehicle.store.get("write") == 0


def test_write_0_skips_ground_impact():
    vehicle, intercept, ctx = _ready(alt=-1.0, write=0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert vehicle.store.get("write") == 0
    assert ctx.combus[ctx.vehicle_slot].status == 1


def test_combus_none_sets_health_without_raising():
    vehicle, intercept, ctx = _ready(alt=-1.0, combus=None)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert vehicle.store.get("write") == 0


def test_mguide_33_alt_below_wp_alt_kills_with_miss():
    swbd = np.array([3.0, 4.0, 12.0])
    vehicle, intercept, ctx = _ready(
        alt=50.0, mguide=33, wp_alt=100.0, swbd=swbd
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert vehicle.store.get("write") == 0
    assert vehicle.store.get("miss") == pytest.approx(13.0)
    assert ctx.combus[ctx.vehicle_slot].status == 0


def test_mguide_33_wp_flag_1_resets_write():
    vehicle, intercept, ctx = _ready(
        alt=200.0, mguide=33, wp_alt=100.0, wp_flag=1, write=0
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert vehicle.store.get("write") == 1


def test_mguide_4_wp_flag_minus1_clears_write_without_kill():
    vehicle, intercept, ctx = _ready(
        alt=7000.0, mguide=4, wp_flag=-1, swbd=np.array([10.0, 20.0, 0.0])
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert vehicle.store.get("write") == 0
    assert ctx.combus[ctx.vehicle_slot].status == 1


def test_mguide_6_closest_approach_miss_and_miss_h():
    vehicle, intercept, ctx = _ready(
        mguide=6,
        time_m=TIME_M,
        sbtim=SBTIM,
        stmii=STMII,
        sbmii=SBMII,
        sbii=SBII,
        vbii=VBII,
        stii=STII,
        vtii=VTII,
        alt=100000.0,
    )
    intercept.execute(vehicle, ctx)
    store = vehicle.store
    assert vehicle.health == 0
    assert store.get("write") == 0
    assert store.get("hit_time") == pytest.approx(HIT_TIME, rel=RTOL, abs=ATOL)
    assert store.get("miss") == pytest.approx(MISS, rel=RTOL, abs=ATOL)
    assert np.allclose(store.get("MISS_I"), MISS_I, rtol=RTOL, atol=ATOL)
    expected_h = _expected_miss_h(STII, VTII, MISS_I)
    assert np.allclose(store.get("MISS_H"), expected_h, rtol=RTOL, atol=ATOL)
    assert np.allclose(store.get("MISS_L"), np.zeros(3), rtol=RTOL, atol=ATOL)
    assert ctx.combus[ctx.vehicle_slot].status == 0


def test_mguide_7_closest_approach_same_as_6():
    vehicle, intercept, ctx = _ready(
        mguide=7,
        time_m=TIME_M,
        sbtim=SBTIM,
        stmii=STMII,
        sbmii=SBMII,
        sbii=SBII,
        vbii=VBII,
        stii=STII,
        vtii=VTII,
        alt=100000.0,
    )
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert vehicle.store.get("miss") == pytest.approx(MISS, rel=RTOL, abs=ATOL)
    expected_h = _expected_miss_h(STII, VTII, MISS_I)
    assert np.allclose(vehicle.store.get("MISS_H"), expected_h, rtol=RTOL, atol=ATOL)


def test_mguide_8_closest_approach_miss_l():
    vehicle, intercept, ctx = _ready(
        mguide=8,
        time_m=TIME_M,
        sbtim=SBTIM,
        stmii=STMII,
        sbmii=SBMII,
        sbii=SBII,
        vbii=VBII,
        stii=STII,
        vtii=VTII,
        alt=100000.0,
    )
    intercept.execute(vehicle, ctx)
    store = vehicle.store
    assert vehicle.health == 0
    assert store.get("miss") == pytest.approx(MISS, rel=RTOL, abs=ATOL)
    assert np.allclose(store.get("MISS_I"), MISS_I, rtol=RTOL, atol=ATOL)
    expected_l = _expected_miss_l(STII, VTII, MISS_I)
    assert np.allclose(store.get("MISS_L"), expected_l, rtol=RTOL, atol=ATOL)
    assert np.allclose(store.get("MISS_H"), np.zeros(3), rtol=RTOL, atol=ATOL)


def test_mguide_6_saves_previous_state_inside_sphere():
    # Outside closest-approach trigger (closing_speed <= 0) but inside 20 km sphere.
    stii = np.array([100.0, 0.0, 0.0])
    sbii = np.zeros(3)
    vtii = np.array([-10.0, 0.0, 0.0])  # UTBI·VTBI < 0 → approaching
    vbii = np.zeros(3)
    vehicle, intercept, ctx = _ready(
        mguide=6,
        time_m=5.0,
        sbii=sbii,
        vbii=vbii,
        stii=stii,
        vtii=vtii,
        alt=100000.0,
    )
    intercept.execute(vehicle, ctx)
    store = vehicle.store
    assert vehicle.health == 1
    assert store.get("dbt") == pytest.approx(100.0)
    assert np.allclose(store.get("SBTIM"), sbii - stii)
    assert np.allclose(store.get("STMII"), stii)
    assert np.allclose(store.get("SBMII"), sbii)
    assert store.get("time_m") == pytest.approx(TIME)


def test_event_diagnostic_packing():
    # mseek=4, mguide=6, maut=53 → mauty=5 mautp=3, mrcs=12 → type=1 mode=2, mprop=4
    # event = 4*1e6 + 6*1e5 + 5*1e4 + 3*1e3 + 1*100 + 2*10 + 4 = 4653124
    vehicle, intercept, ctx = _ready(
        alt=100.0, mseek=4, mguide=6, maut=53, mrcs_moment=12, mprop=4
    )
    # Keep outside sphere so no kill path: STII far
    vehicle.store.set("STII", np.array([1.0e6, 0.0, 0.0]))
    vehicle.store.set("SBII", np.zeros(3))
    intercept.execute(vehicle, ctx)
    assert vehicle.store.get("event") == 4653124
    assert type(vehicle.store.get("event")) is int


def test_never_calls_sys_exit():
    def boom(*args, **kwargs):
        raise AssertionError("sys.exit called")

    with patch("sys.exit", side_effect=boom):
        vehicle, intercept, ctx = _ready(alt=100.0)
        intercept.execute(vehicle, ctx)
        assert vehicle.health == 1
        vehicle, intercept, ctx = _ready(alt=-1.0)
        intercept.execute(vehicle, ctx)
        assert vehicle.health == 0


def test_hyper6_registers_intercept_after_euler():
    vehicle = Hyper6("Hypersonic", None, None)
    names = [module.name for module in vehicle.modules]
    assert "intercept" in names
    assert names.index("intercept") == names.index("euler") + 1
    assert names[-1] == "intercept"
