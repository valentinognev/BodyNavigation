import math

import numpy as np
import pytest

from cadac.constants import RAD, REARTH
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round6.rocket6.guidance import Rocket6Guidance

RTOL = 1e-12
ATOL = 1e-14
PLOT = ("plot",)
ZEROS3 = (0.0, 0.0, 0.0)

# insertion LTG data (input.asc) plus frozen INS at LTG engagement
LTG_STEP = 0.01
DT = LTG_STEP
NUM_STAGES = 2
DBI_DESIRED = 6470e3
DVBI_DESIRED = 6600.0
THTVDX_DESIRED = 1.0
DELAY_IGNITION = 0.1
AMIN = 3.0
LAMD_LIMIT = 0.01
EXHAUST_VEL1 = 2795.0
EXHAUST_VEL2 = 2785.0
BURNOUT_EPOCH1 = 51.5
BURNOUT_EPOCH2 = 126.0
CHAR_TIME1 = 81.9
CHAR_TIME2 = 112.2
FMASSR = 9552.0
MPROP = 4
GRAV = 9.81
TIME = 11.0
SBIIC = (REARTH + 50e3, 1.0e5, 2.0e5)
VBIIC = (400.0, 2500.0, 200.0)
FSPCB = (34.0, 0.1, -0.2)
TBIC = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

FIELDS = {
    "mguide": ("int", "data", 0, ()),
    "init_flag": ("int", "init", 1, ()),
    "time_ltg": ("real", "diag", 0.0, ()),
    "UTBC": ("vec", "out", ZEROS3, PLOT),
    "RBIAS": ("vec", "save", ZEROS3, ()),
    "beco_flag": ("int", "diag", 0, ()),
    "inisw_flag": ("int", "init", 1, ()),
    "skip_flag": ("int", "init", 1, ()),
    "ipas_flag": ("int", "init", 1, ()),
    "ipas2_flag": ("int", "init", 1, ()),
    "print_flag": ("int", "init", 1, ()),
    "ltg_count": ("int", "save", 0, ()),
    "ltg_step": ("real", "data", 0.0, ()),
    "dbi_desired": ("real", "data", 0.0, ()),
    "dvbi_desired": ("real", "data", 0.0, ()),
    "thtvdx_desired": ("real", "data", 0.0, ()),
    "num_stages": ("int", "data", 0, ()),
    "delay_ignition": ("real", "data", 0.0, ()),
    "amin": ("real", "data", 0.0, ()),
    "char_time1": ("real", "data", 0.0, ()),
    "char_time2": ("real", "data", 0.0, ()),
    "char_time3": ("real", "data", 0.0, ()),
    "exhaust_vel1": ("real", "data", 0.0, ()),
    "exhaust_vel2": ("real", "data", 0.0, ()),
    "exhaust_vel3": ("real", "data", 0.0, ()),
    "burnout_epoch1": ("real", "data", 0.0, ()),
    "burnout_epoch2": ("real", "data", 0.0, ()),
    "burnout_epoch3": ("real", "data", 0.0, ()),
    "lamd_limit": ("real", "data", 0.0, ()),
    "RGRAV": ("vec", "save", ZEROS3, ()),
    "RGO": ("vec", "save", ZEROS3, ()),
    "VGO": ("vec", "save", ZEROS3, ()),
    "SDII": ("vec", "save", ZEROS3, ()),
    "UD": ("vec", "save", ZEROS3, ()),
    "UY": ("vec", "save", ZEROS3, ()),
    "UZ": ("vec", "save", ZEROS3, ()),
    "vgom": ("real", "diag", 0.0, ()),
    "tgo": ("real", "save", 0.0, ()),
    "nst": ("int", "save", 0, ()),
    "ULAM": ("vec", "diag", ZEROS3, ()),
    "LAMD": ("vec", "diag", ZEROS3, ()),
    "UTIC": ("vec", "diag", ZEROS3, ()),
    "nstmax": ("int", "diag", 0, ()),
    "lamd": ("real", "diag", 0.0, PLOT),
    "dpd": ("real", "diag", 0.0, PLOT),
    "dbd": ("real", "diag", 0.0, PLOT),
    "ddb": ("real", "diag", 0.0, PLOT),
    "dvdb": ("real", "diag", 0.0, PLOT),
    "thtvddbx": ("real", "diag", 0.0, PLOT),
    "alphacomx": ("real", "out", 0.0, ()),
    "betacomx": ("real", "out", 0.0, ()),
}
DEFINED = tuple(FIELDS)
INS_NAMES = ("SBIIC", "VBIIC", "TBIC", "FSPCB")
NOT_DEFINED = INS_NAMES + (
    "time",
    "grav",
    "mprop",
    "dbi",
    "dvbi",
    "thtvdx",
    "fmassr",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(int_step=DT, sim_time=TIME):
    return SimContext(
        sim_time=sim_time,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _plant_ins(store, *, sbiic=SBIIC, vbiic=VBIIC, tbic=TBIC, fspcb=FSPCB):
    sbiic = np.asarray(sbiic, dtype=float)
    vbiic = np.asarray(vbiic, dtype=float)
    dbi = math.sqrt(sbiic[0] ** 2 + sbiic[1] ** 2 + sbiic[2] ** 2)
    dvbi = math.sqrt(vbiic[0] ** 2 + vbiic[1] ** 2 + vbiic[2] ** 2)
    for name, value, ftype, role, module in (
        ("time", TIME, "real", "exec", "kinematics"),
        ("grav", GRAV, "real", "out", "environment"),
        ("mprop", MPROP, "int", "data", "propulsion"),
        ("fmassr", FMASSR, "real", "save", "propulsion"),
        ("dbi", dbi, "real", "out", "newton"),
        ("dvbi", dvbi, "real", "out", "newton"),
        ("thtvdx", THTVDX_DESIRED, "real", "init/out", "newton"),
        ("SBIIC", sbiic, "vec", "out", "ins"),
        ("VBIIC", vbiic, "vec", "out", "ins"),
        ("FSPCB", fspcb, "vec", "out", "ins"),
        ("TBIC", tbic, "mat", "out", "ins"),
    ):
        store.define(Field(name, value, ftype, role, module))


def _ready(*, mguide=5, **overrides):
    vehicle = _Vehicle()
    guidance = Rocket6Guidance()
    guidance.define(vehicle)
    _plant_ins(vehicle.store)
    store = vehicle.store
    store.set("mguide", mguide)
    store.set("ltg_step", LTG_STEP)
    store.set("num_stages", NUM_STAGES)
    store.set("dbi_desired", DBI_DESIRED)
    store.set("dvbi_desired", DVBI_DESIRED)
    store.set("thtvdx_desired", THTVDX_DESIRED)
    store.set("delay_ignition", DELAY_IGNITION)
    store.set("amin", AMIN)
    store.set("lamd_limit", LAMD_LIMIT)
    store.set("exhaust_vel1", EXHAUST_VEL1)
    store.set("exhaust_vel2", EXHAUST_VEL2)
    store.set("burnout_epoch1", BURNOUT_EPOCH1)
    store.set("burnout_epoch2", BURNOUT_EPOCH2)
    store.set("char_time1", CHAR_TIME1)
    store.set("char_time2", CHAR_TIME2)
    for name, value in overrides.items():
        store.set(name, value)
    guidance.initialize(vehicle, _ctx())
    return guidance, vehicle


def _run(guidance, vehicle, n, int_step=DT):
    ctx = _ctx(int_step)
    for _ in range(n):
        guidance.execute(vehicle, ctx)


def _univec3_cpp(vec):
    v1, v2, v3 = (float(vec[0]), float(vec[1]), float(vec[2]))
    d = math.sqrt(v1 * v1 + v2 * v2 + v3 * v3)
    if d == 0.0:
        return np.zeros(3)
    return np.array([v1 / d, v2 / d, v3 / d], dtype=float)


def _skip_flag_cpp(skip_flag):
    # C++: if(skip_flag){ skip_flag++; if(skip_flag==10) skip_flag=0; }
    if skip_flag:
        skip_flag += 1
        if skip_flag == 10:
            skip_flag = 0
    return skip_flag


def test_name_is_guidance():
    # Break: class name token not "guidance" (module bind).
    assert Rocket6Guidance().name == "guidance"


def test_define_registers_cpp_fields_not_ins():
    # Break: def_guidance field missing, wrong role/default, or INS names defined here.
    vehicle = _Vehicle()
    Rocket6Guidance().define(vehicle)
    store = vehicle.store
    for name, (ftype, role, default, outputs) in FIELDS.items():
        assert name in store.names(), name
        field = store.field(name)
        assert field.module == "guidance"
        assert field.role == role, name
        assert field.type == ftype, name
        assert field.outputs == outputs, name
        if ftype == "int":
            assert store.get(name) == default, name
        elif ftype == "real":
            assert store.get(name) == default, name
        else:
            np.testing.assert_array_equal(store.get(name), np.zeros(3))
    for name in NOT_DEFINED:
        assert name not in store.names(), name


def test_mguide_0_zeros_utbc():
    # Break: mguide==0 does not zero UTBC.
    guidance, vehicle = _ready(mguide=0)
    vehicle.store.set("UTBC", np.array([1.0, 2.0, 3.0]))
    _run(guidance, vehicle, 1)
    np.testing.assert_allclose(
        vehicle.store.get("UTBC"), np.zeros(3), rtol=RTOL, atol=ATOL
    )


def test_mguide_6_raises():
    # Break: mguide==6 (and other non-0/5) is not ValueError.
    guidance, vehicle = _ready(mguide=6)
    with pytest.raises(ValueError, match="6"):
        _run(guidance, vehicle, 1)


def test_skip_flag_first_nine_ltg_calls_leave_utic_zero():
    # Break: skip_flag init/increment not C++ (first 9 LTG calls leave UTIC unset).
    guidance, vehicle = _ready()
    store = vehicle.store
    assert store.get("skip_flag") == 1
    _run(guidance, vehicle, 1)
    # time_ltg==0 is not > ltg_step*0, so LTG is not called.
    assert store.get("skip_flag") == 1
    assert store.get("ltg_count") == 0
    np.testing.assert_allclose(store.get("UTIC"), np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("UTBC"), np.zeros(3), rtol=RTOL, atol=ATOL)

    skip = 1
    for k in range(1, 10):
        _run(guidance, vehicle, 1)
        skip = _skip_flag_cpp(skip)
        assert store.get("skip_flag") == skip
        assert store.get("ltg_count") == k
        np.testing.assert_allclose(store.get("UTIC"), np.zeros(3), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(store.get("UTBC"), np.zeros(3), rtol=RTOL, atol=ATOL)
    assert skip == 0


def test_mguide_5_utbc_unit_after_skip_clears():
    # Break: 10th LTG call does not write finite unit-ish UTBC; or UTBC!=TBIC*UTIC.
    guidance, vehicle = _ready()
    _run(guidance, vehicle, 1 + 10)
    store = vehicle.store
    utic = np.asarray(store.get("UTIC"), dtype=float)
    utbc = np.asarray(store.get("UTBC"), dtype=float)
    tbic = np.asarray(store.get("TBIC"), dtype=float)
    assert np.all(np.isfinite(utbc))
    assert np.all(np.isfinite(utic))
    np.testing.assert_allclose(np.linalg.norm(utic), 1.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.linalg.norm(utbc), 1.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(utbc, tbic @ utic, rtol=RTOL, atol=ATOL)
    # C++ _crct: SDII = UD * dbi_desired
    sdii = np.asarray(store.get("SDII"), dtype=float)
    ud = np.asarray(store.get("UD"), dtype=float)
    np.testing.assert_allclose(sdii, ud * DBI_DESIRED, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.linalg.norm(ud), 1.0, rtol=RTOL, atol=ATOL)
    assert store.get("mprop") == 4
    assert store.get("beco_flag") == 0
    assert store.get("tgo") > 0.0
    assert np.isfinite(store.get("tgo"))


def test_ltg_igrl_a1_a2_replica():
    # Break: C++ _igrl a1/a2 (including x==2 factor 1.001) not used.
    from cadac.vehicles.round6.rocket6.guidance import _ltg_igrl_a1_a2

    x = 0.5
    a1, a2 = _ltg_igrl_a1_a2(x)
    want_a1 = 1.0 / (1.0 - 0.5 * x)
    want_a2 = 1.0 / (1.0 - x)
    assert a1 == pytest.approx(want_a1, rel=RTOL, abs=ATOL)
    assert a2 == pytest.approx(want_a2, rel=RTOL, abs=ATOL)
    a1_two, _a2_two = _ltg_igrl_a1_a2(2.0)
    want_a1_two = 1.0 / (1.0 - 0.5 * 2.0 * 1.001)
    assert a1_two == pytest.approx(want_a1_two, rel=RTOL, abs=ATOL)


def test_ltg_x_equals_1_raises_valueerror():
    # Break: C++ exit(1) on x==1 not mapped to ValueError (or sys.exit used).
    guidance, vehicle = _ready(dvbi_desired=1.0e9)
    with pytest.raises(ValueError, match="LTG Terminator"):
        _run(guidance, vehicle, 2)


def test_crct_unit_cross_replica():
    # Break: C++ operator% (unitized cross) not used for UY.
    from cadac.vehicles.round6.rocket6.guidance import _unit_cross

    vbiic = np.array(VBIIC, dtype=float)
    sbiic = np.array(SBIIC, dtype=float)
    v1 = vbiic[1] * sbiic[2] - vbiic[2] * sbiic[1]
    v2 = vbiic[2] * sbiic[0] - vbiic[0] * sbiic[2]
    v3 = vbiic[0] * sbiic[1] - vbiic[1] * sbiic[0]
    dv = math.sqrt(v1 * v1 + v2 * v2 + v3 * v3)
    want = np.array([v1 / dv, v2 / dv, v3 / dv], dtype=float)
    np.testing.assert_allclose(_unit_cross(vbiic, sbiic), want, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        _univec3_cpp(sbiic),
        np.array(SBIIC, dtype=float) / math.sqrt(sum(c * c for c in SBIIC)),
        rtol=RTOL,
        atol=ATOL,
    )
