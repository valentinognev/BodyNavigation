from math import sqrt
from types import SimpleNamespace

import cadac.constants as cadac_constants
import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.hyper6.control import SMALL, Hyper6Control

RTOL = 1e-12
ATOL = 1e-14

WRCL = 8.0
ZRCL = 0.9
DLLP = -2.5
DLLDA = 18.0
PHICOMX = 10.0
PHIBDCX = 0.0
PPCX = 3.0
TP = 0.5
PCOMX = 8.0
ZETLAGR = 1.1
DLA = 8.0
DLDE = -12.0
DMA = -2.0
DMQ = -1.5
DMDE = -8.0
QQCX = 4.0
QCOMX = 0.0
DVBEC = 1000.0
DYB = -6.0
DYDR = 4.0
DNB = 1.5
DNR = -0.8
DNDR = -3.0
RRCX = 2.0
RCOMX = 0.0

DEFINED = (
    "maut",
    "mroll",
    "mfreeze",
    "waclp",
    "zaclp",
    "paclp",
    "alimitx",
    "dalimx",
    "delimx",
    "drlimx",
    "philimx",
    "wrcl",
    "zrcl",
    "yyd",
    "yy",
    "zzd",
    "zz",
    "delacx",
    "delecx",
    "delrcx",
    "alcomx",
    "ancomx",
    "GAINFP",
    "gainp",
    "gainl",
    "altcom",
    "gainalt",
    "gainaltrate",
    "altrate",
    "gkp",
    "gkphi",
    "fspb2m",
    "fspb2mt",
    "fspb3m",
    "fspb3mt",
    "qqxm",
    "qqxmt",
    "rrxm",
    "rrxmt",
    "dqcxm",
    "dqcxmt",
    "drcxm",
    "drcxmt",
    "isetc2",
    "factwacl",
    "factzacl",
    "tp",
    "zetlagr",
    "psivdcomx",
    "facthead",
    "gainpsi",
    "phicomx",
    "pcomx",
    "qcomx",
    "rcomx",
    "thtvdcomx",
    "zrate",
    "grate",
    "wnlagr",
    "pgam",
    "wgam",
    "zgam",
    "GAINGAM",
    "gainff",
)
INT_FIELDS = ("maut", "mroll", "mfreeze")
VEC_FIELDS = ("GAINFP", "GAINGAM")
STATE_FIELDS = ("yyd", "yy", "zzd", "zz")
PLOT_OUT = ("delacx", "delecx", "delrcx")
PLOT_DATA = ("alcomx", "ancomx", "phicomx")
DIAG_REAL = (
    "altrate",
    "gkp",
    "gkphi",
    "fspb2m",
    "fspb2mt",
    "fspb3m",
    "fspb3mt",
    "qqxm",
    "qqxmt",
    "rrxm",
    "rrxmt",
    "dqcxm",
    "dqcxmt",
    "drcxm",
    "drcxmt",
    "gainpsi",
    "zrate",
    "grate",
    "wnlagr",
    "gainff",
)
EXTERNALS = (
    "phibdcx",
    "ppcx",
    "qqcx",
    "rrcx",
    "dvbec",
    "dllp",
    "dllda",
    "dla",
    "dlde",
    "dma",
    "dmq",
    "dmde",
    "dyb",
    "dydr",
    "dnb",
    "dnr",
    "dndr",
    "phiblx",
    "ppx",
    "qqx",
    "rrx",
    "dvbe",
    "anlimpx",
    "anlimnx",
    "psivlcomx",
    "thtvlcomx",
)


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.01,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _externals(
    store,
    *,
    phibdcx=PHIBDCX,
    ppcx=PPCX,
    dllp=DLLP,
    dllda=DLLDA,
    dla=DLA,
    dlde=DLDE,
    dma=DMA,
    dmq=DMQ,
    dmde=DMDE,
    qqcx=QQCX,
    dvbec=DVBEC,
    dyb=DYB,
    dydr=DYDR,
    dnb=DNB,
    dnr=DNR,
    dndr=DNDR,
    rrcx=RRCX,
):
    store.define(Field("phibdcx", phibdcx, "real", "out", "ins"))
    store.define(Field("ppcx", ppcx, "real", "out", "ins"))
    store.define(Field("dllp", dllp, "real", "out", "aerodynamics"))
    store.define(Field("dllda", dllda, "real", "out", "aerodynamics"))
    store.define(Field("dla", dla, "real", "out", "aerodynamics"))
    store.define(Field("dlde", dlde, "real", "out", "aerodynamics"))
    store.define(Field("dma", dma, "real", "out", "aerodynamics"))
    store.define(Field("dmq", dmq, "real", "out", "aerodynamics"))
    store.define(Field("dmde", dmde, "real", "out", "aerodynamics"))
    store.define(Field("qqcx", qqcx, "real", "out", "ins"))
    store.define(Field("dvbec", dvbec, "real", "out", "ins"))
    store.define(Field("dyb", dyb, "real", "out", "aerodynamics"))
    store.define(Field("dydr", dydr, "real", "out", "aerodynamics"))
    store.define(Field("dnb", dnb, "real", "out", "aerodynamics"))
    store.define(Field("dnr", dnr, "real", "out", "aerodynamics"))
    store.define(Field("dndr", dndr, "real", "out", "aerodynamics"))
    store.define(Field("rrcx", rrcx, "real", "out", "ins"))


def _ready(**kw):
    vehicle = SimpleNamespace(store=StateStore())
    ctrl = Hyper6Control()
    ctrl.define(vehicle)
    _externals(vehicle.store, **kw)
    store = vehicle.store
    store.set("wrcl", WRCL)
    store.set("zrcl", ZRCL)
    store.set("tp", TP)
    store.set("zetlagr", ZETLAGR)
    ctrl.initialize(vehicle, _ctx())
    return vehicle, ctrl


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _control_roll(store, phicomx):
    wrcl = store.get("wrcl")
    zrcl = store.get("zrcl")
    phibdcx = store.get("phibdcx")
    ppcx = store.get("ppcx")
    dllp = store.get("dllp")
    dllda = store.get("dllda")
    gkp = (2.0 * zrcl * wrcl + dllp) / dllda
    gkphi = wrcl * wrcl / dllda
    ephi = gkphi * (phicomx - phibdcx) * RAD
    dpc = ephi - gkp * ppcx * RAD
    delacx = dpc * DEG
    return delacx, gkp, gkphi


def _control_roll_rate(store, pcomx):
    tp = store.get("tp")
    ppcx = store.get("ppcx")
    dllp = store.get("dllp")
    dllda = store.get("dllda")
    kp = (1 / tp + dllp) / dllda
    return kp * (pcomx - ppcx)


def _control_pitch_rate(store, qcomx):
    zetlagr = store.get("zetlagr")
    dla = store.get("dla")
    dlde = store.get("dlde")
    dma = store.get("dma")
    dmq = store.get("dmq")
    dmde = store.get("dmde")
    qqcx = store.get("qqcx")
    dvbec = store.get("dvbec")
    zrate = dla / dvbec - dma * dlde / (dvbec * dmde)
    aa = dla / dvbec - dmq
    bb = -dma - dmq * dla / dvbec
    dum1 = aa - 2.0 * zetlagr * zetlagr * zrate
    dum2 = aa * aa - 4.0 * zetlagr * zetlagr * bb
    radix = dum1 * dum1 - dum2
    if radix < 0.0:
        radix = 0.0
    if abs(dmde) < SMALL:
        dmde = SMALL * _sign(dmde)
    grate = (-dum1 + sqrt(radix)) / (-dmde)
    delecx = grate * (qqcx - qcomx)
    return delecx, zrate, grate, radix


def _control_yaw_rate(store, rcomx):
    zetlagr = store.get("zetlagr")
    dyb = store.get("dyb")
    dydr = store.get("dydr")
    dnb = store.get("dnb")
    dnr = store.get("dnr")
    dndr = store.get("dndr")
    rrcx = store.get("rrcx")
    dvbec = store.get("dvbec")
    zrate = -dyb / dvbec + dnb * dydr / (dvbec * dndr)
    aa = -dyb / dvbec - dnr
    bb = dnb + dyb * dnr / dvbec
    dum1 = aa - 2.0 * zetlagr * zetlagr * zrate
    dum2 = aa * aa - 4.0 * zetlagr * zetlagr * bb
    radix = dum1 * dum1 - dum2
    if radix < 0.0:
        radix = 0.0
    if abs(dndr) < SMALL:
        dndr = SMALL * _sign(dndr)
    grate = (-dum1 + sqrt(radix)) / (-dndr)
    dum3 = grate * dndr * zrate
    radix = bb + dum3
    if radix < 0.0:
        radix = 0.0
    wnlagr = sqrt(radix)
    delrcx = grate * (rrcx - rcomx)
    return delrcx, zrate, grate, wnlagr


def test_name_is_control():
    assert Hyper6Control().name == "control"


def test_small_is_module_level_not_in_constants():
    assert SMALL == 1.0e-7
    assert not hasattr(cadac_constants, "SMALL")


def test_define_registers_cpp_fields_not_externals():
    vehicle = SimpleNamespace(store=StateStore())
    Hyper6Control().define(vehicle)
    store = vehicle.store
    for name in DEFINED:
        assert name in store.names()
        assert store.field(name).module == "control"
    for name in INT_FIELDS:
        assert store.get(name) == 0
        assert store.field(name).type == "int"
        assert store.field(name).role == "data"
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
    for name in PLOT_OUT:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "out"
        assert store.field(name).outputs == ("plot",)
    for name in PLOT_DATA:
        assert store.get(name) == 0.0
        assert store.field(name).role == "data"
        assert store.field(name).outputs == ("plot",)
    assert store.field("altcom").role == "data"
    assert store.field("altcom").outputs == ()
    assert store.field("zetlagr").role == "data"
    assert store.field("zetlagr").outputs == ()
    assert store.field("alimitx").role == "data"
    assert store.field("psivdcomx").role == "data"
    assert store.field("thtvdcomx").role == "data"
    assert store.field("isetc2").type == "real"
    assert store.field("isetc2").role == "init"
    for name in DIAG_REAL:
        assert store.field(name).type == "real"
        assert store.field(name).role == "diag"
        assert store.field(name).outputs == ()
    for name in EXTERNALS:
        assert name not in store.names()


def test_initialize_is_pass():
    vehicle, _ctrl = _ready()
    store = vehicle.store
    assert store.get("delacx") == 0.0
    assert store.get("delecx") == 0.0
    assert store.get("delrcx") == 0.0
    assert store.get("gkp") == 0.0
    assert store.get("gkphi") == 0.0
    assert store.get("zrate") == 0.0
    assert store.get("grate") == 0.0
    assert store.get("wnlagr") == 0.0


def test_execute_dispatches_maut_24():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.define(Field("thtbdcx", 2.5, "real", "out", "ins"))
    store.define(Field("thtvdcx", 0.0, "real", "out", "ins"))
    store.define(Field("dvbe", 1000.0, "real", "out", "newton"))
    store.set("pgam", 4.0)
    store.set("wgam", 2.0)
    store.set("zgam", 0.7)
    store.set("dalimx", 1.0e6)
    store.set("delimx", 1.0e6)
    store.set("drlimx", 1.0e6)
    store.set("philimx", 60.0)
    store.set("thtvdcomx", 0.0)
    store.set("maut", 24)
    store.set("phicomx", PHICOMX)
    want, gkp, gkphi = _control_roll(store, PHICOMX)
    ctrl.execute(vehicle, _ctx())
    assert _approx(store.get("delacx"), want)
    assert store.get("delacx") != 0.0
    assert store.get("delecx") != 0.0
    assert store.get("delrcx") != 0.0
    assert _approx(store.get("gkp"), gkp)
    assert _approx(store.get("gkphi"), gkphi)


def test_control_roll_matches_cadac_formulas():
    vehicle, ctrl = _ready()
    want, gkp, gkphi = _control_roll(vehicle.store, PHICOMX)
    got = ctrl.control_roll(vehicle, PHICOMX)
    assert _approx(got, want)
    assert _approx(vehicle.store.get("gkp"), gkp)
    assert _approx(vehicle.store.get("gkphi"), gkphi)
    assert vehicle.store.get("delacx") == 0.0
    assert gkp != 0.0
    assert gkphi != 0.0
    assert got != 0.0


def test_control_roll_reads_ins_phibdcx_not_body_euler():
    vehicle, ctrl = _ready(phibdcx=4.0, ppcx=1.0)
    vehicle.store.define(Field("phiblx", 45.0, "real", "diag", "kinematics"))
    vehicle.store.define(Field("ppx", 99.0, "real", "out", "euler"))
    want, gkp, gkphi = _control_roll(vehicle.store, PHICOMX)
    got = ctrl.control_roll(vehicle, PHICOMX)
    assert _approx(got, want)
    assert _approx(vehicle.store.get("gkp"), gkp)
    assert _approx(vehicle.store.get("gkphi"), gkphi)
    decoy = (2.0 * ZRCL * WRCL + DLLP) / DLLDA
    ephi = (WRCL * WRCL / DLLDA) * (PHICOMX - 45.0) * RAD
    dpc = ephi - decoy * 99.0 * RAD
    assert got != pytest.approx(dpc * DEG, rel=RTOL, abs=ATOL)


def test_control_roll_zero_error_still_computes_gains():
    vehicle, ctrl = _ready(phibdcx=PHICOMX, ppcx=0.0)
    want, gkp, gkphi = _control_roll(vehicle.store, PHICOMX)
    got = ctrl.control_roll(vehicle, PHICOMX)
    assert _approx(got, want)
    assert got == 0.0
    assert _approx(vehicle.store.get("gkp"), gkp)
    assert _approx(vehicle.store.get("gkphi"), gkphi)
    assert gkp != 0.0
    assert gkphi != 0.0


def test_control_roll_rate_matches_cadac_formulas():
    vehicle, ctrl = _ready()
    want = _control_roll_rate(vehicle.store, PCOMX)
    got = ctrl.control_roll_rate(vehicle, PCOMX)
    assert _approx(got, want)
    assert want != 0.0
    assert vehicle.store.get("gkp") == 0.0
    assert vehicle.store.get("delacx") == 0.0


def test_control_pitch_rate_matches_cadac_formulas():
    vehicle, ctrl = _ready()
    want, _zrate, grate, radix = _control_pitch_rate(vehicle.store, QCOMX)
    assert radix > 0.0
    got = ctrl.control_pitch_rate(vehicle, QCOMX)
    assert _approx(got, want)
    assert grate != 0.0
    assert want != 0.0
    assert vehicle.store.get("zrate") == 0.0
    assert vehicle.store.get("grate") == 0.0
    assert vehicle.store.get("wnlagr") == 0.0
    assert vehicle.store.get("delecx") == 0.0
    assert vehicle.store.get("dmde") == DMDE


def test_control_pitch_rate_reads_ins_qqcx_dvbec():
    vehicle, ctrl = _ready(qqcx=5.0, dvbec=800.0)
    vehicle.store.define(Field("qqx", 40.0, "real", "out", "euler"))
    vehicle.store.define(Field("dvbe", 180.0, "real", "out", "newton"))
    want, _zrate, grate, _radix = _control_pitch_rate(vehicle.store, QCOMX)
    got = ctrl.control_pitch_rate(vehicle, QCOMX)
    assert _approx(got, want)
    assert grate != 0.0


def test_control_pitch_rate_clamps_negative_radix():
    vehicle, ctrl = _ready(dla=0.0, dlde=1.0, dma=400.0, dmq=-3.0, dmde=-2.0)
    want, _zrate, grate, radix = _control_pitch_rate(vehicle.store, QCOMX)
    assert radix == 0.0
    got = ctrl.control_pitch_rate(vehicle, QCOMX)
    assert _approx(got, want)
    assert grate != 0.0


def test_control_pitch_rate_clamps_small_dmde_cadac_sign():
    vehicle, ctrl = _ready(dmde=1.0e-8)
    want_pos, _, grate_pos, _ = _control_pitch_rate(vehicle.store, QCOMX)
    got_pos = ctrl.control_pitch_rate(vehicle, QCOMX)
    assert _approx(got_pos, want_pos)
    assert vehicle.store.get("dmde") == 1.0e-8
    unclamped = grate_pos * (SMALL / 1.0e-8)
    assert grate_pos != pytest.approx(unclamped, rel=RTOL, abs=ATOL)

    vehicle_n, ctrl_n = _ready(dmde=-1.0e-8)
    want_neg, _, _grate_neg, _ = _control_pitch_rate(vehicle_n.store, QCOMX)
    got_neg = ctrl_n.control_pitch_rate(vehicle_n, QCOMX)
    assert _approx(got_neg, want_neg)
    assert vehicle_n.store.get("dmde") == -1.0e-8
    assert want_pos != pytest.approx(want_neg, rel=RTOL, abs=ATOL)


def test_control_yaw_rate_matches_cadac_formulas():
    vehicle, ctrl = _ready()
    want, zrate, grate, wnlagr = _control_yaw_rate(vehicle.store, RCOMX)
    got = ctrl.control_yaw_rate(vehicle, RCOMX)
    assert _approx(got, want)
    assert _approx(vehicle.store.get("zrate"), zrate)
    assert _approx(vehicle.store.get("grate"), grate)
    assert _approx(vehicle.store.get("wnlagr"), wnlagr)
    assert grate != 0.0
    assert wnlagr != 0.0
    assert want != 0.0
    assert vehicle.store.get("delrcx") == 0.0
    assert vehicle.store.get("dndr") == DNDR


def test_control_yaw_rate_reads_ins_rrcx_dvbec():
    vehicle, ctrl = _ready(rrcx=3.0, dvbec=800.0)
    vehicle.store.define(Field("rrx", 20.0, "real", "out", "euler"))
    vehicle.store.define(Field("dvbe", 180.0, "real", "out", "newton"))
    want, zrate, grate, wnlagr = _control_yaw_rate(vehicle.store, RCOMX)
    got = ctrl.control_yaw_rate(vehicle, RCOMX)
    assert _approx(got, want)
    assert _approx(vehicle.store.get("zrate"), zrate)
    assert _approx(vehicle.store.get("grate"), grate)
    assert _approx(vehicle.store.get("wnlagr"), wnlagr)


def test_control_yaw_rate_clamps_negative_radix():
    vehicle, ctrl = _ready(dyb=0.0, dydr=1.0, dnb=-400.0, dnr=3.0, dndr=-2.0)
    want, zrate, grate, wnlagr = _control_yaw_rate(vehicle.store, RCOMX)
    assert wnlagr == 0.0
    got = ctrl.control_yaw_rate(vehicle, RCOMX)
    assert _approx(got, want)
    assert _approx(vehicle.store.get("zrate"), zrate)
    assert _approx(vehicle.store.get("grate"), grate)
    assert vehicle.store.get("wnlagr") == 0.0
    assert grate != 0.0


def test_control_yaw_rate_clamps_small_dndr_cadac_sign():
    vehicle, ctrl = _ready(dndr=1.0e-8)
    want_pos, _, grate_pos, _ = _control_yaw_rate(vehicle.store, RCOMX)
    got_pos = ctrl.control_yaw_rate(vehicle, RCOMX)
    assert _approx(got_pos, want_pos)
    assert vehicle.store.get("dndr") == 1.0e-8
    unclamped = grate_pos * (SMALL / 1.0e-8)
    assert grate_pos != pytest.approx(unclamped, rel=RTOL, abs=ATOL)

    vehicle_n, ctrl_n = _ready(dndr=-1.0e-8)
    want_neg, _, _grate_neg, _ = _control_yaw_rate(vehicle_n.store, RCOMX)
    got_neg = ctrl_n.control_yaw_rate(vehicle_n, RCOMX)
    assert _approx(got_neg, want_neg)
    assert vehicle_n.store.get("dndr") == -1.0e-8
    assert want_pos != pytest.approx(want_neg, rel=RTOL, abs=ATOL)


def test_cadac_sign_zero_is_plus_one_not_numpy_sign():
    from cadac.vehicles.hyper6.control import _sign as prod_sign

    assert prod_sign(0.0) == 1
    assert prod_sign(-0.0) == 1
    assert prod_sign(-1.0) == -1
    assert prod_sign(1.0) == 1
    assert np.sign(0.0) == 0.0


def test_terminate_exists_and_is_pass():
    vehicle, ctrl = _ready()
    assert ctrl.terminate(vehicle, _ctx()) is None
    assert vehicle.store.get("delacx") == 0.0
