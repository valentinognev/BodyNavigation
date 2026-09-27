from math import sqrt
from types import SimpleNamespace

import cadac.constants as cadac_constants
import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.agm6.control import SMALL, Agm6Control

RTOL = 1e-12
ATOL = 1e-14

WRCL = 5.0
ZRCL = 0.9
DPLIMX = 25.0
PHICOMX = 0.0
PHIBLCX = 2.0
WBECB_ROLL = (0.1, 0.0, 0.0)
DLP = -2.0
DLD = 20.0

ZETLAGR = 0.9
QQCOMX = 0.0
RRCOMX = 0.0
DVBE = 293.0
DNA = 40.0
DND = -80.0
DMA = -15.0
DMQ = -2.0
DMD = -50.0
WBECB_RATE = (0.1, 0.05, -0.03)

DEFINED = (
    "maut",
    "mfreeze",
    "wacl",
    "zacl",
    "pacl",
    "alimit",
    "dqlimx",
    "drlimx",
    "dplimx",
    "phicomx",
    "wrcl",
    "zrcl",
    "yyd",
    "yy",
    "zzd",
    "zz",
    "dpcx",
    "dqcx",
    "drcx",
    "GAINFB",
    "gainp",
    "gkp",
    "gkphi",
    "zetlagr",
    "qqcomx",
    "rrcomx",
    "zrate",
    "grate",
    "wnlagr",
)
INT_FIELDS = ("maut", "mfreeze")
VEC_FIELDS = ("GAINFB",)
STATE_FIELDS = ("yyd", "yy", "zzd", "zz")
PLOT_OUT = ("dpcx", "dqcx", "drcx")
PLOT_DATA = ("wacl", "phicomx", "qqcomx", "rrcomx")
DIAG_REAL = ("gkp", "gkphi", "zrate", "grate", "wnlagr")
EXTERNALS = (
    "WBECB",
    "phiblcx",
    "dlp",
    "dld",
    "dna",
    "dnd",
    "dma",
    "dmq",
    "dmd",
    "dvbe",
    "time",
    "pdynmc",
    "ancomx",
    "alcomx",
    "FSPCB",
    "phiblx",
    "ppx",
    "qqx",
    "rrx",
    "dllp",
    "dllda",
    "delacx",
    "delecx",
    "delrcx",
    "phibdcx",
    "ppcx",
)


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.001,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _externals(
    store,
    *,
    wbecb=WBECB_ROLL,
    phiblcx=PHIBLCX,
    dlp=DLP,
    dld=DLD,
    dna=DNA,
    dnd=DND,
    dma=DMA,
    dmq=DMQ,
    dmd=DMD,
    dvbe=DVBE,
):
    store.define(Field("WBECB", wbecb, "vec", "out", "ins"))
    store.define(Field("phiblcx", phiblcx, "real", "out", "ins"))
    store.define(Field("dlp", dlp, "real", "out", "aerodynamics"))
    store.define(Field("dld", dld, "real", "out", "aerodynamics"))
    store.define(Field("dna", dna, "real", "out", "aerodynamics"))
    store.define(Field("dnd", dnd, "real", "out", "aerodynamics"))
    store.define(Field("dma", dma, "real", "out", "aerodynamics"))
    store.define(Field("dmq", dmq, "real", "out", "aerodynamics"))
    store.define(Field("dmd", dmd, "real", "out", "aerodynamics"))
    store.define(Field("dvbe", dvbe, "real", "out", "newton"))


def _ready(**kw):
    vehicle = SimpleNamespace(store=StateStore())
    ctrl = Agm6Control()
    ctrl.define(vehicle)
    _externals(vehicle.store, **kw)
    store = vehicle.store
    store.set("wrcl", WRCL)
    store.set("zrcl", ZRCL)
    store.set("dplimx", DPLIMX)
    store.set("phicomx", PHICOMX)
    store.set("zetlagr", ZETLAGR)
    store.set("qqcomx", QQCOMX)
    store.set("rrcomx", RRCOMX)
    ctrl.initialize(vehicle, _ctx())
    return vehicle, ctrl


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _control_roll(store):
    dplimx = store.get("dplimx")
    phicomx = store.get("phicomx")
    wrcl = store.get("wrcl")
    zrcl = store.get("zrcl")
    dlp = store.get("dlp")
    dld = store.get("dld")
    wbecb = store.get("WBECB")
    phiblcx = store.get("phiblcx")
    gkp = (2.0 * zrcl * wrcl + dlp) / dld
    gkphi = wrcl * wrcl / dld
    pp = wbecb[0]
    ephi = gkphi * (phicomx - phiblcx) * RAD
    dpc = ephi - gkp * pp
    dpcx = dpc * DEG
    if abs(dpcx) > dplimx:
        dpcx = dplimx * _sign(dpcx)
    return dpcx, gkp, gkphi


def _control_rate(store):
    zetlagr = store.get("zetlagr")
    qqcomx = store.get("qqcomx")
    rrcomx = store.get("rrcomx")
    dvbe = store.get("dvbe")
    dna = store.get("dna")
    dnd = store.get("dnd")
    dma = store.get("dma")
    dmq = store.get("dmq")
    dmd = store.get("dmd")
    wbecb = store.get("WBECB")
    zrate = dna / dvbe - dma * dnd / (dvbe * dmd)
    aa = dna / dvbe - dmq
    bb = -dma - dmq * dna / dvbe
    dum1 = aa - 2 * zetlagr * zetlagr * zrate
    dum2 = aa * aa - 4 * zetlagr * zetlagr * bb
    radix = dum1 * dum1 - dum2
    if radix < 0:
        radix = SMALL
    if abs(dmd) < SMALL:
        dmd = SMALL * _sign(dmd)
    grate = -(-dum1 + sqrt(radix)) / dmd
    dum3 = grate * dmd * zrate
    radix = bb + dum3
    if radix < 0:
        radix = SMALL
    wnlagr = sqrt(radix)
    qq = wbecb[1]
    rr = wbecb[2]
    dqcx = DEG * grate * qq - qqcomx
    drcx = DEG * grate * rr - rrcomx
    return dqcx, drcx, zrate, grate, wnlagr


def test_name_is_control():
    assert Agm6Control.name == "control"
    assert Agm6Control().name == "control"


def test_small_is_module_level_not_in_constants():
    assert SMALL == 1.0e-7
    assert not hasattr(cadac_constants, "SMALL")


def test_define_registers_cpp_fields_not_externals():
    vehicle = SimpleNamespace(store=StateStore())
    Agm6Control().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    for name in DEFINED:
        assert store.field(name).module == "control"
    for name in INT_FIELDS:
        assert store.get(name) == 0
        assert store.field(name).type == "int"
        assert store.field(name).role == "data"
        assert store.field(name).outputs == ()
    zeros = np.zeros(3)
    for name in VEC_FIELDS:
        np.testing.assert_array_equal(store.get(name), zeros)
        assert store.get(name).shape == (3,)
        assert store.field(name).type == "vec"
        assert store.field(name).role == "diag"
        assert store.field(name).outputs == ()
    for name in STATE_FIELDS:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "state"
        assert store.field(name).outputs == ()
    for name in PLOT_OUT:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "out"
        assert store.field(name).outputs == ("plot",)
    for name in PLOT_DATA:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "data"
        assert store.field(name).outputs == ("plot",)
    for name in DIAG_REAL:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "diag"
        assert store.field(name).outputs == ()
    assert store.field("alimit").role == "data"
    assert store.field("alimit").outputs == ()
    assert store.field("dplimx").role == "data"
    assert store.field("wrcl").role == "data"
    assert store.field("zrcl").role == "data"
    assert store.field("zetlagr").role == "data"
    assert store.field("zetlagr").outputs == ()
    assert store.field("gainp").role == "data"
    assert store.field("zacl").role == "data"
    assert store.field("pacl").role == "data"
    assert store.field("dqlimx").role == "data"
    assert store.field("drlimx").role == "data"
    for name in EXTERNALS:
        assert name not in store.names()


def test_initialize_is_pass():
    vehicle, _ctrl = _ready()
    store = vehicle.store
    assert store.get("dpcx") == 0.0
    assert store.get("dqcx") == 0.0
    assert store.get("drcx") == 0.0
    assert store.get("gkp") == 0.0
    assert store.get("gkphi") == 0.0
    assert store.get("zrate") == 0.0
    assert store.get("grate") == 0.0
    assert store.get("wnlagr") == 0.0


def test_execute_maut_zero_returns_without_writing():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.set("maut", 0)
    store.set("dpcx", 7.0)
    store.set("dqcx", 8.0)
    store.set("drcx", 9.0)
    store.set("gkp", 1.5)
    store.set("gkphi", 2.5)
    store.set("zrate", 3.5)
    store.set("grate", 4.5)
    store.set("wnlagr", 5.5)
    assert ctrl.execute(vehicle, _ctx()) is None
    assert store.get("dpcx") == 7.0
    assert store.get("dqcx") == 8.0
    assert store.get("drcx") == 9.0
    assert store.get("gkp") == 1.5
    assert store.get("gkphi") == 2.5
    assert store.get("zrate") == 3.5
    assert store.get("grate") == 4.5
    assert store.get("wnlagr") == 5.5


def test_control_roll_matches_cadac_formulas():
    vehicle, ctrl = _ready(wbecb=WBECB_ROLL, phiblcx=PHIBLCX, dlp=DLP, dld=DLD)
    want, gkp, gkphi = _control_roll(vehicle.store)
    assert ctrl.control_roll(vehicle) is None
    store = vehicle.store
    assert _approx(store.get("dpcx"), want)
    assert _approx(store.get("gkp"), gkp)
    assert _approx(store.get("gkphi"), gkphi)
    assert gkp != 0.0
    assert gkphi != 0.0
    assert store.get("dpcx") != 0.0
    assert store.get("dqcx") == 0.0
    assert store.get("drcx") == 0.0
    assert abs(store.get("dpcx")) <= DPLIMX


def test_control_roll_reads_ins_wbecb_phiblcx_not_body_euler():
    vehicle, ctrl = _ready(wbecb=WBECB_ROLL, phiblcx=PHIBLCX)
    vehicle.store.define(Field("phiblx", 45.0, "real", "diag", "kinematics"))
    vehicle.store.define(Field("ppx", 99.0, "real", "out", "euler"))
    want, gkp, gkphi = _control_roll(vehicle.store)
    ctrl.control_roll(vehicle)
    assert _approx(vehicle.store.get("dpcx"), want)
    assert _approx(vehicle.store.get("gkp"), gkp)
    assert _approx(vehicle.store.get("gkphi"), gkphi)
    decoy_gkp = (2.0 * ZRCL * WRCL + DLP) / DLD
    ephi = (WRCL * WRCL / DLD) * (PHICOMX - 45.0) * RAD
    dpc = ephi - decoy_gkp * 99.0 * RAD
    assert vehicle.store.get("dpcx") != pytest.approx(dpc * DEG, rel=RTOL, abs=ATOL)


def test_control_roll_does_not_scale_wbecb_p_by_rad():
    vehicle, ctrl = _ready(wbecb=WBECB_ROLL, phiblcx=PHIBLCX)
    want, _gkp, gkphi = _control_roll(vehicle.store)
    ctrl.control_roll(vehicle)
    gkp = (2.0 * ZRCL * WRCL + DLP) / DLD
    ephi = gkphi * (PHICOMX - PHIBLCX) * RAD
    wrong = (ephi - gkp * WBECB_ROLL[0] * RAD) * DEG
    assert _approx(vehicle.store.get("dpcx"), want)
    assert vehicle.store.get("dpcx") != pytest.approx(wrong, rel=RTOL, abs=ATOL)


def test_control_roll_zero_error_still_computes_gains():
    vehicle, ctrl = _ready(wbecb=(0.0, 0.0, 0.0), phiblcx=PHICOMX)
    want, gkp, gkphi = _control_roll(vehicle.store)
    ctrl.control_roll(vehicle)
    assert _approx(vehicle.store.get("dpcx"), want)
    assert vehicle.store.get("dpcx") == 0.0
    assert _approx(vehicle.store.get("gkp"), gkp)
    assert _approx(vehicle.store.get("gkphi"), gkphi)
    assert gkp != 0.0
    assert gkphi != 0.0


def test_control_roll_limits_dpcx_cadac_sign():
    vehicle, ctrl = _ready(wbecb=(0.0, 0.0, 0.0), phiblcx=90.0)
    want, gkp, gkphi = _control_roll(vehicle.store)
    unlimited = gkphi * (PHICOMX - 90.0) * RAD * DEG
    assert abs(unlimited) > DPLIMX
    assert want == -DPLIMX
    ctrl.control_roll(vehicle)
    assert _approx(vehicle.store.get("dpcx"), want)
    assert vehicle.store.get("dpcx") == -DPLIMX
    assert _approx(vehicle.store.get("gkp"), gkp)
    assert _approx(vehicle.store.get("gkphi"), gkphi)


def test_control_rate_matches_cadac_formulas_dqcx_finite():
    vehicle, ctrl = _ready(wbecb=WBECB_RATE)
    want_q, want_r, zrate, grate, wnlagr = _control_rate(vehicle.store)
    assert np.isfinite(want_q)
    assert np.isfinite(want_r)
    assert want_q != 0.0
    assert ctrl.control_rate(vehicle) is None
    store = vehicle.store
    assert _approx(store.get("dqcx"), want_q)
    assert _approx(store.get("drcx"), want_r)
    assert _approx(store.get("zrate"), zrate)
    assert _approx(store.get("grate"), grate)
    assert _approx(store.get("wnlagr"), wnlagr)
    assert np.isfinite(store.get("dqcx"))
    assert grate != 0.0
    assert wnlagr != 0.0
    assert store.get("dpcx") == 0.0
    assert store.get("dmd") == DMD


def test_control_rate_uses_deg_times_grate_qq():
    vehicle, ctrl = _ready(wbecb=WBECB_RATE)
    _want_q, _want_r, _zrate, grate, _wnlagr = _control_rate(vehicle.store)
    ctrl.control_rate(vehicle)
    wrong = grate * (WBECB_RATE[1] - QQCOMX)
    assert vehicle.store.get("dqcx") != pytest.approx(wrong, rel=RTOL, abs=ATOL)
    assert _approx(vehicle.store.get("dqcx"), DEG * grate * WBECB_RATE[1] - QQCOMX)


def test_control_rate_clamps_negative_radix_to_small():
    vehicle, ctrl = _ready(
        wbecb=WBECB_RATE,
        dna=0.0,
        dnd=1.0,
        dma=400.0,
        dmq=-3.0,
        dmd=-2.0,
    )
    want_q, want_r, zrate, grate, wnlagr = _control_rate(vehicle.store)
    ctrl.control_rate(vehicle)
    assert _approx(vehicle.store.get("dqcx"), want_q)
    assert _approx(vehicle.store.get("drcx"), want_r)
    assert _approx(vehicle.store.get("zrate"), zrate)
    assert _approx(vehicle.store.get("grate"), grate)
    assert _approx(vehicle.store.get("wnlagr"), wnlagr)
    assert np.isfinite(vehicle.store.get("dqcx"))
    assert grate != 0.0


def test_control_rate_clamps_small_dmd_cadac_sign():
    vehicle, ctrl = _ready(wbecb=WBECB_RATE, dma=0.0, dmd=1.0e-8)
    want_pos, _, _, grate_pos, _ = _control_rate(vehicle.store)
    ctrl.control_rate(vehicle)
    assert _approx(vehicle.store.get("dqcx"), want_pos)
    assert vehicle.store.get("dmd") == 1.0e-8
    unclamped = grate_pos * (SMALL / 1.0e-8)
    assert grate_pos != pytest.approx(unclamped, rel=RTOL, abs=ATOL)
    assert grate_pos != 0.0

    vehicle_n, ctrl_n = _ready(wbecb=WBECB_RATE, dma=0.0, dmd=-1.0e-8)
    want_neg, _, _, _grate_neg, _ = _control_rate(vehicle_n.store)
    ctrl_n.control_rate(vehicle_n)
    assert _approx(vehicle_n.store.get("dqcx"), want_neg)
    assert vehicle_n.store.get("dmd") == -1.0e-8
    assert want_pos != pytest.approx(want_neg, rel=RTOL, abs=ATOL)


def test_cadac_sign_zero_is_plus_one_not_numpy_sign():
    from cadac.vehicles.flat6.agm6.control import _sign as prod_sign

    assert prod_sign(0.0) == 1
    assert prod_sign(-0.0) == 1
    assert prod_sign(-1.0) == -1
    assert prod_sign(1.0) == 1
    assert np.sign(0.0) == 0.0


def test_does_not_require_time_or_pdynmc():
    vehicle, ctrl = _ready()
    assert "time" not in vehicle.store.names()
    assert "pdynmc" not in vehicle.store.names()
    ctrl.control_roll(vehicle)
    ctrl.control_rate(vehicle)
    assert "time" not in vehicle.store.names()
    assert "pdynmc" not in vehicle.store.names()


def test_terminate_exists_and_is_pass():
    vehicle, ctrl = _ready()
    assert ctrl.terminate(vehicle, _ctx()) is None
    assert vehicle.store.get("dpcx") == 0.0
    assert vehicle.store.get("dqcx") == 0.0
