from math import sqrt
from pathlib import Path

import cadac.constants as cadac_constants
import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.sam6.control import SMALL, Sam6Control

RTOL = 1e-12
ATOL = 1e-14

DVBE = 16.0
ZETLAGR = 1.2
ZRCL = 0.9
DLP = -8.0
DLD = 12.0
DNA = 40.0
DND = 8.0
DMA = -6.0
DMQ = -1.5
DMD = -15.0
WBECB = (0.1, 0.05, -0.02)
THTBLCX = 80.0
PHIBLCX = 0.0
PHICOMX = 0.0
FACTWRCL = 0.0
TP = 0.5

DEFINED = (
    "maut",
    "mfreeze",
    "wacl",
    "zacl",
    "pacl",
    "alimitx",
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
    "tp",
    "GAINFB",
    "gainp",
    "dqcx_rcs",
    "drcx_rcs",
    "qqcomx",
    "rrcomx",
    "gkp",
    "gkphi",
    "factwrcl",
    "zetlagr",
    "zrate",
    "grate",
    "wnlagr",
    "wacl_bias",
    "pacl_bias",
    "zacl_bias",
    "ancomx_test",
    "alcomx_test",
)
INT_FIELDS = ("maut", "mfreeze")
STATE_FIELDS = ("yyd", "yy", "zzd", "zz")
OUT_SCRN = ("dpcx", "dqcx", "drcx")
OUT_PLAIN = ("dqcx_rcs", "drcx_rcs")
DIAG_PLOT = ("wacl", "zacl", "pacl", "wrcl", "zrate", "wnlagr")
DATA_PLOT = ("ancomx_test", "alcomx_test")
NOT_DEFINED = (
    "factwacl",
    "twcl",
    "WBECB",
    "thtblcx",
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
    "WBEB",
    "ppcx",
    "qqcx",
    "rrcx",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ctx(int_step=0.001):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _plant(
    store,
    *,
    dlp=DLP,
    dld=DLD,
    dna=DNA,
    dnd=DND,
    dma=DMA,
    dmq=DMQ,
    dmd=DMD,
    dvbe=DVBE,
    wbecb=WBECB,
    thtblcx=THTBLCX,
    phiblcx=PHIBLCX,
):
    store.define(Field("dlp", dlp, "real", "out", "aerodynamics"))
    store.define(Field("dld", dld, "real", "out", "aerodynamics"))
    store.define(Field("dna", dna, "real", "out", "aerodynamics"))
    store.define(Field("dnd", dnd, "real", "out", "aerodynamics"))
    store.define(Field("dma", dma, "real", "out", "aerodynamics"))
    store.define(Field("dmq", dmq, "real", "out", "aerodynamics"))
    store.define(Field("dmd", dmd, "real", "out", "aerodynamics"))
    store.define(Field("dvbe", dvbe, "real", "out", "newton"))
    store.define(Field("WBECB", wbecb, "vec", "diag", "ins"))
    store.define(Field("thtblcx", thtblcx, "real", "out", "ins"))
    store.define(Field("phiblcx", phiblcx, "real", "out", "ins"))


def _ready(**kw):
    vehicle = _Vehicle()
    ctrl = Sam6Control()
    ctrl.define(vehicle)
    ctrl.initialize(vehicle, _ctx())
    _plant(vehicle.store, **kw)
    store = vehicle.store
    store.set("zrcl", ZRCL)
    store.set("zetlagr", ZETLAGR)
    store.set("phicomx", PHICOMX)
    store.set("factwrcl", FACTWRCL)
    store.set("tp", TP)
    return vehicle, ctrl


def _cpp_roll(store):
    dlp = store.get("dlp")
    dld = store.get("dld")
    factwrcl = store.get("factwrcl")
    zrcl = store.get("zrcl")
    phicomx = store.get("phicomx")
    phiblcx = store.get("phiblcx")
    thtblcx = store.get("thtblcx")
    tp = store.get("tp")
    pp = store.get("WBECB")[0]
    wrcl = -0.8 * dlp * (1 + factwrcl)
    gkp = (2 * zrcl * wrcl + dlp) / dld
    gkphi = wrcl * wrcl / dld
    ephi = gkphi * (phicomx - phiblcx) * RAD
    dpc = ephi - gkp * pp
    if abs(thtblcx) > 88:
        kp = (1 / tp + dlp) / dld
        dpcx = kp * pp * DEG
    else:
        dpcx = dpc * DEG
    return dpcx, wrcl, gkp, gkphi


def _cpp_rate(store):
    zetlagr = store.get("zetlagr")
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
        radix = 0
    if abs(dmd) < SMALL:
        dmd = SMALL * _sign(dmd)
    grate = (-dum1 + sqrt(radix)) / dmd
    dum3 = grate * dmd * zrate
    radix = bb + dum3
    if radix < 0:
        radix = 0
    wnlagr = sqrt(radix)
    qq = wbecb[1]
    rr = wbecb[2]
    dqcx = DEG * grate * qq
    drcx = DEG * grate * rr
    return dqcx, drcx, zrate, grate, wnlagr


def test_name_is_control():
    assert Sam6Control().name == "control"


def test_small_is_module_level_not_in_constants():
    assert SMALL == 1e-7
    assert not hasattr(cadac_constants, "SMALL")


def test_define_registers_cpp_fields_not_externals():
    vehicle = _Vehicle()
    Sam6Control().define(vehicle)
    store = vehicle.store
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        assert field.module == "control", name
    for name in INT_FIELDS:
        assert store.get(name) == 0
        assert store.field(name).type == "int"
        assert store.field(name).role == "data"
    for name in STATE_FIELDS:
        assert _approx(store.get(name), 0.0)
        assert store.field(name).type == "real"
        assert store.field(name).role == "state"
        assert store.field(name).outputs == ()
    for name in OUT_SCRN:
        assert _approx(store.get(name), 0.0)
        assert store.field(name).role == "out"
        assert store.field(name).outputs == ("scrn",)
    for name in OUT_PLAIN:
        assert _approx(store.get(name), 0.0)
        assert store.field(name).role == "out"
        assert store.field(name).outputs == ()
    for name in DIAG_PLOT:
        assert _approx(store.get(name), 0.0)
        assert store.field(name).role == "diag"
        assert store.field(name).outputs == ("plot",)
    assert store.field("GAINFB").type == "vec"
    assert store.field("GAINFB").role == "diag"
    assert store.field("GAINFB").outputs == ()
    np.testing.assert_array_equal(store.get("GAINFB"), np.zeros(3))
    for name in ("gkp", "gkphi", "grate"):
        assert _approx(store.get(name), 0.0)
        assert store.field(name).role == "diag"
        assert store.field(name).outputs == ()
    for name in DATA_PLOT:
        assert store.field(name).role == "data"
        assert store.field(name).outputs == ("plot",)
    assert store.field("zrcl").role == "data"
    assert store.field("zetlagr").role == "data"
    assert store.field("phicomx").role == "data"
    assert store.field("tp").role == "data"
    assert store.field("factwrcl").role == "data"
    for name in NOT_DEFINED:
        assert name not in store.names(), name


def test_initialize_is_pass():
    vehicle = _Vehicle()
    ctrl = Sam6Control()
    ctrl.define(vehicle)
    vehicle.store.set("dpcx", 7.0)
    vehicle.store.set("wrcl", 3.0)
    before = {name: vehicle.store.get(name) for name in DEFINED if name != "GAINFB"}
    assert ctrl.initialize(vehicle, _ctx()) is None
    for name in before:
        got = vehicle.store.get(name)
        if name in INT_FIELDS:
            assert got == before[name]
        else:
            assert _approx(got, before[name]), name


def test_maut_0_does_not_write_dpcx():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.set("maut", 0)
    store.set("dpcx", 0.0)
    store.set("wrcl", 5.0)
    store.set("gkp", 1.0)
    ctrl.execute(vehicle, _ctx())
    assert _approx(store.get("dpcx"), 0.0)
    assert _approx(store.get("dqcx"), 0.0)
    assert _approx(store.get("drcx"), 0.0)
    assert _approx(store.get("wrcl"), 5.0)
    assert _approx(store.get("gkp"), 1.0)
    assert _approx(store.get("zrate"), 0.0)


def test_maut_2_dummy_aero_dqcx_drcx_finite():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.set("maut", 2)
    ctrl.execute(vehicle, _ctx())
    assert np.isfinite(store.get("dqcx"))
    assert np.isfinite(store.get("drcx"))
    assert np.isfinite(store.get("dpcx"))
    assert store.get("dqcx") != 0.0
    assert store.get("drcx") != 0.0


def test_maut_4_raises():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.set("maut", 4)
    store.set("dpcx", 99.0)
    store.set("dqcx", 88.0)
    with pytest.raises(ValueError):
        ctrl.execute(vehicle, _ctx())
    assert _approx(store.get("dpcx"), 99.0)
    assert _approx(store.get("dqcx"), 88.0)


def test_maut_1_roll_only_does_not_write_rate():
    vehicle = _Vehicle()
    ctrl = Sam6Control()
    ctrl.define(vehicle)
    store = vehicle.store
    store.define(Field("dlp", DLP, "real", "out", "aerodynamics"))
    store.define(Field("dld", DLD, "real", "out", "aerodynamics"))
    store.define(Field("WBECB", WBECB, "vec", "diag", "ins"))
    store.define(Field("thtblcx", THTBLCX, "real", "out", "ins"))
    store.define(Field("phiblcx", PHIBLCX, "real", "out", "ins"))
    store.set("maut", 1)
    store.set("zrcl", ZRCL)
    store.set("phicomx", PHICOMX)
    store.set("factwrcl", FACTWRCL)
    store.set("tp", TP)
    ctrl.execute(vehicle, _ctx())
    dpcx, wrcl, gkp, gkphi = _cpp_roll(store)
    assert _approx(store.get("dpcx"), dpcx)
    assert _approx(store.get("wrcl"), wrcl)
    assert _approx(store.get("gkp"), gkp)
    assert _approx(store.get("gkphi"), gkphi)
    assert _approx(store.get("dqcx"), 0.0)
    assert _approx(store.get("drcx"), 0.0)
    assert _approx(store.get("zrate"), 0.0)


def test_maut_2_execute_matches_cadac_roll_and_rate():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.set("maut", 2)
    dpcx, wrcl, gkp, gkphi = _cpp_roll(store)
    dqcx, drcx, zrate, grate, wnlagr = _cpp_rate(store)
    ctrl.execute(vehicle, _ctx())
    assert _approx(store.get("dpcx"), dpcx)
    assert _approx(store.get("wrcl"), wrcl)
    assert _approx(store.get("gkp"), gkp)
    assert _approx(store.get("gkphi"), gkphi)
    assert _approx(store.get("dqcx"), dqcx)
    assert _approx(store.get("drcx"), drcx)
    assert _approx(store.get("dqcx_rcs"), dqcx)
    assert _approx(store.get("drcx_rcs"), drcx)
    assert _approx(store.get("zrate"), zrate)
    assert _approx(store.get("grate"), grate)
    assert _approx(store.get("wnlagr"), wnlagr)


def test_control_roll_matches_cadac_formulas():
    vehicle, ctrl = _ready()
    dpcx, wrcl, gkp, gkphi = _cpp_roll(vehicle.store)
    ctrl.control_roll(vehicle)
    store = vehicle.store
    assert _approx(store.get("dpcx"), dpcx)
    assert _approx(store.get("wrcl"), wrcl)
    assert _approx(store.get("gkp"), gkp)
    assert _approx(store.get("gkphi"), gkphi)
    assert dpcx != 0.0
    assert wrcl != 0.0


def test_control_roll_reads_ins_wbecb_thtblcx_phiblcx():
    vehicle, ctrl = _ready(wbecb=(0.2, 0.0, 0.0), thtblcx=10.0, phiblcx=3.0)
    store = vehicle.store
    store.define(Field("WBEB", (9.0, 9.0, 9.0), "vec", "diag", "euler"))
    store.define(Field("thtblx", 45.0, "real", "diag", "kinematics"))
    store.define(Field("phiblx", 45.0, "real", "diag", "kinematics"))
    store.set("phicomx", 5.0)
    dpcx, wrcl, gkp, gkphi = _cpp_roll(store)
    ctrl.control_roll(vehicle)
    assert _approx(store.get("dpcx"), dpcx)
    assert _approx(store.get("wrcl"), wrcl)
    assert _approx(store.get("gkp"), gkp)
    assert _approx(store.get("gkphi"), gkphi)
    pp_decoy = 9.0
    wrcl_d = -0.8 * DLP * (1 + FACTWRCL)
    gkp_d = (2 * ZRCL * wrcl_d + DLP) / DLD
    gkphi_d = wrcl_d * wrcl_d / DLD
    ephi = gkphi_d * (5.0 - 45.0) * RAD
    decoy = (ephi - gkp_d * pp_decoy) * DEG
    assert store.get("dpcx") != pytest.approx(decoy, rel=RTOL, abs=ATOL)


def test_control_roll_vertical_switches_to_rate():
    vehicle, ctrl = _ready(thtblcx=89.0, wbecb=(0.1, 0.0, 0.0))
    dpcx, wrcl, gkp, gkphi = _cpp_roll(vehicle.store)
    ctrl.control_roll(vehicle)
    store = vehicle.store
    assert abs(store.get("thtblcx")) > 88
    assert _approx(store.get("dpcx"), dpcx)
    assert _approx(store.get("wrcl"), wrcl)
    assert _approx(store.get("gkp"), gkp)
    assert _approx(store.get("gkphi"), gkphi)
    pos = ((gkp * (store.get("phicomx") - store.get("phiblcx")) * RAD) - gkp * 0.1) * DEG
    assert store.get("dpcx") != pytest.approx(pos, rel=RTOL, abs=ATOL)


def test_control_rate_matches_cadac_formulas():
    vehicle, ctrl = _ready()
    dqcx, drcx, zrate, grate, wnlagr = _cpp_rate(vehicle.store)
    ctrl.control_rate(vehicle)
    store = vehicle.store
    assert _approx(store.get("dqcx"), dqcx)
    assert _approx(store.get("drcx"), drcx)
    assert _approx(store.get("dqcx_rcs"), dqcx)
    assert _approx(store.get("drcx_rcs"), drcx)
    assert _approx(store.get("zrate"), zrate)
    assert _approx(store.get("grate"), grate)
    assert _approx(store.get("wnlagr"), wnlagr)
    assert dqcx != 0.0
    assert grate != 0.0


def test_control_rate_ignores_unused_qqcomx_rrcomx():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.set("qqcomx", 99.0)
    store.set("rrcomx", -50.0)
    dqcx, drcx, zrate, grate, wnlagr = _cpp_rate(store)
    ctrl.control_rate(vehicle)
    assert _approx(store.get("dqcx"), dqcx)
    assert _approx(store.get("drcx"), drcx)
    assert _approx(store.get("zrate"), zrate)
    assert _approx(store.get("grate"), grate)
    assert _approx(store.get("wnlagr"), wnlagr)
    tracked = DEG * grate * (store.get("WBECB")[1] - 99.0)
    assert store.get("dqcx") != pytest.approx(tracked, rel=RTOL, abs=ATOL)


def test_control_rate_reads_dvbe_not_dvbec():
    vehicle, ctrl = _ready(dvbe=20.0)
    vehicle.store.define(Field("dvbec", 1000.0, "real", "out", "ins"))
    dqcx, drcx, zrate, grate, wnlagr = _cpp_rate(vehicle.store)
    ctrl.control_rate(vehicle)
    store = vehicle.store
    assert _approx(store.get("dqcx"), dqcx)
    assert _approx(store.get("drcx"), drcx)
    assert _approx(store.get("zrate"), zrate)
    assert _approx(store.get("grate"), grate)
    assert _approx(store.get("wnlagr"), wnlagr)


def test_control_rate_clamps_negative_radix():
    vehicle, ctrl = _ready(dna=0.0, dnd=1.0, dma=400.0, dmq=-3.0, dmd=-2.0)
    dqcx, drcx, zrate, grate, wnlagr = _cpp_rate(vehicle.store)
    ctrl.control_rate(vehicle)
    store = vehicle.store
    assert _approx(store.get("dqcx"), dqcx)
    assert _approx(store.get("drcx"), drcx)
    assert _approx(store.get("zrate"), zrate)
    assert _approx(store.get("grate"), grate)
    assert _approx(store.get("wnlagr"), wnlagr)


def test_control_rate_clamps_small_dmd_cadac_sign():
    vehicle, ctrl = _ready(dmd=1.0e-8)
    dqcx_p, drcx_p, _z, grate_p, _w = _cpp_rate(vehicle.store)
    ctrl.control_rate(vehicle)
    store = vehicle.store
    assert _approx(store.get("dqcx"), dqcx_p)
    assert _approx(store.get("drcx"), drcx_p)
    assert _approx(store.get("grate"), grate_p)
    assert store.get("dmd") == 1.0e-8
    unclamped = grate_p * (SMALL / 1.0e-8)
    assert grate_p != pytest.approx(unclamped, rel=RTOL, abs=ATOL)

    vehicle_n, ctrl_n = _ready(dmd=-1.0e-8)
    dqcx_n, drcx_n, _zn, grate_n, _wn = _cpp_rate(vehicle_n.store)
    ctrl_n.control_rate(vehicle_n)
    assert _approx(vehicle_n.store.get("dqcx"), dqcx_n)
    assert _approx(vehicle_n.store.get("drcx"), drcx_n)
    assert _approx(vehicle_n.store.get("grate"), grate_n)
    assert vehicle_n.store.get("dmd") == -1.0e-8
    assert grate_p != pytest.approx(grate_n, rel=RTOL, abs=ATOL)


def test_cadac_sign_zero_is_plus_one_not_numpy_sign():
    from cadac.vehicles.sam6.control import _sign as prod_sign

    assert prod_sign(0.0) == 1
    assert prod_sign(-0.0) == 1
    assert prod_sign(-1.0) == -1
    assert prod_sign(1.0) == 1
    assert np.sign(0.0) == 0.0


def test_execute_does_not_require_factwacl_twcl():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.set("maut", 2)
    assert "factwacl" not in store.names()
    assert "twcl" not in store.names()
    ctrl.execute(vehicle, _ctx())
    assert np.isfinite(store.get("dqcx"))


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.sam6.control as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src
    assert "Hyper6Control" not in src
    assert "Plane6Control" not in src
    assert "np.sign" not in src
    assert "_cadac_sign" not in src


def test_terminate_exists_and_is_pass():
    vehicle, ctrl = _ready()
    assert ctrl.terminate(vehicle, _ctx()) is None
    assert _approx(vehicle.store.get("dpcx"), 0.0)
    assert _approx(vehicle.store.get("dqcx"), 0.0)
