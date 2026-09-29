from math import atan2, cos, fabs, sin, sqrt
from pathlib import Path

import numpy as np
import pytest

from cadac.constants import AGRAV, DEG
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.sam6.control import Sam6Control

RTOL = 1e-12
ATOL = 1e-14
SMALL = 1e-7

DT = 0.001
ALIMITX = 50.0
ANCOMX_TEST = 1.0
ALCOMX = 0.0
ANCOMX = 0.0
REALQ1 = -10.0
REALQ2 = -5.0
DVBE = 16.0
ZRCL = 0.9
DLP = -8.0
DLD = 12.0
DNA = 40.0
DMA = -6.0
DMQ = -1.5
DMD = -15.0
DYB = 35.0
DNB = -5.0
DNR = -1.2
DLND = -12.0
FSPCB = (0.0, 1.0, -10.0)
WBECB = (0.1, 0.05, -0.02)
THTBLCX = 80.0
PHIBLCX = 0.0
PHICOMX = 0.0
FACTWRCL = 0.0
TP = 0.5
GAINP = 0.0


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ctx(int_step=DT):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _plant(store, **kw):
    values = dict(
        dlp=DLP,
        dld=DLD,
        dna=DNA,
        dma=DMA,
        dmq=DMQ,
        dmd=DMD,
        dyb=DYB,
        dnb=DNB,
        dnr=DNR,
        dlnd=DLND,
        dvbe=DVBE,
        realq1=REALQ1,
        realq2=REALQ2,
        fspcb=FSPCB,
        wbecb=WBECB,
        thtblcx=THTBLCX,
        phiblcx=PHIBLCX,
        ancomx=ANCOMX,
        alcomx=ALCOMX,
    )
    values.update(kw)
    store.define(Field("dlp", values["dlp"], "real", "out", "aerodynamics"))
    store.define(Field("dld", values["dld"], "real", "out", "aerodynamics"))
    store.define(Field("dna", values["dna"], "real", "out", "aerodynamics"))
    store.define(Field("dma", values["dma"], "real", "out", "aerodynamics"))
    store.define(Field("dmq", values["dmq"], "real", "out", "aerodynamics"))
    store.define(Field("dmd", values["dmd"], "real", "out", "aerodynamics"))
    store.define(Field("dyb", values["dyb"], "real", "out", "aerodynamics"))
    store.define(Field("dnb", values["dnb"], "real", "out", "aerodynamics"))
    store.define(Field("dnr", values["dnr"], "real", "out", "aerodynamics"))
    store.define(Field("dlnd", values["dlnd"], "real", "out", "aerodynamics"))
    store.define(Field("dvbe", values["dvbe"], "real", "out", "newton"))
    store.define(Field("realq1", values["realq1"], "real", "diag", "aerodynamics"))
    store.define(Field("realq2", values["realq2"], "real", "diag", "aerodynamics"))
    store.define(Field("FSPCB", values["fspcb"], "vec", "diag", "ins"))
    store.define(Field("WBECB", values["wbecb"], "vec", "diag", "ins"))
    store.define(Field("thtblcx", values["thtblcx"], "real", "out", "ins"))
    store.define(Field("phiblcx", values["phiblcx"], "real", "out", "ins"))
    store.define(Field("ancomx", values["ancomx"], "real", "data", "guidance"))
    store.define(Field("alcomx", values["alcomx"], "real", "data", "guidance"))


def _ready(**kw):
    vehicle = _Vehicle()
    ctrl = Sam6Control()
    ctrl.define(vehicle)
    ctrl.initialize(vehicle, _ctx())
    _plant(vehicle.store, **kw)
    store = vehicle.store
    store.set("zrcl", ZRCL)
    store.set("phicomx", PHICOMX)
    store.set("factwrcl", FACTWRCL)
    store.set("tp", TP)
    store.set("alimitx", ALIMITX)
    store.set("gainp", GAINP)
    store.set("ancomx_test", ANCOMX_TEST)
    store.set("alcomx_test", 0.0)
    return vehicle, ctrl


def _cpp_accel(store, int_step):
    wacl_bias = store.get("wacl_bias")
    pacl_bias = store.get("pacl_bias")
    zacl_bias = store.get("zacl_bias")
    alimitx = store.get("alimitx")
    gainp = store.get("gainp")
    ancomx = store.get("ancomx") + store.get("ancomx_test")
    alcomx = store.get("alcomx") + store.get("alcomx_test")
    dna = store.get("dna")
    dma = store.get("dma")
    dmq = store.get("dmq")
    dmd = store.get("dmd")
    dlnd = store.get("dlnd")
    realq1 = store.get("realq1")
    realq2 = store.get("realq2")
    dnr = store.get("dnr")
    dyb = store.get("dyb")
    dnb = store.get("dnb")
    fspcb = store.get("FSPCB")
    wbecb = store.get("WBECB")
    yyd = store.get("yyd")
    yy = store.get("yy")
    zzd = store.get("zzd")
    zz = store.get("zz")
    dvbe = store.get("dvbe")
    aa = sqrt(alcomx * alcomx + ancomx * ancomx)
    if aa > alimitx:
        aa = alimitx
    if fabs(ancomx) < SMALL and fabs(alcomx) < SMALL:
        phi = 0.0
    else:
        phi = atan2(ancomx, alcomx)
    alcomx = aa * cos(phi)
    ancomx = aa * sin(phi)
    zacl = 0.7 * (1 + zacl_bias)
    wacl = fabs(realq1) * (1 + wacl_bias)
    pacl = (fabs(realq2) + 35) * (1 + pacl_bias)
    gainfb3 = wacl * wacl * pacl / (dna * dmd)
    gainfb2 = (2 * zacl * wacl + pacl + dmq - dna / dvbe) / dmd
    gainfb1 = (
        wacl * wacl
        + 2 * zacl * wacl * pacl
        + dma
        + dmq * dna / dvbe
        - gainfb2 * dna * dmd / dvbe
    ) / (dna * dmd) - gainp
    qq = wbecb[1]
    fspb3 = fspcb[2]
    zzd_new = AGRAV * ancomx + fspb3
    zz = integrate(zzd_new, zzd, zz, int_step)
    zzd = zzd_new
    dqc = -gainfb1 * (-fspb3) - gainfb2 * qq + gainfb3 * zz + gainp * zzd
    dqcx = dqc * DEG
    gainfb3 = -wacl * wacl * pacl / (dyb * dlnd)
    gainfb2 = (2 * zacl * wacl + pacl + dnr + dyb / dvbe) / dlnd
    gainfb1 = (
        -wacl * wacl
        - 2 * zacl * wacl * pacl
        + dnb
        + dnr * dyb / dvbe
        - gainfb2 * dyb * dlnd / dvbe
    ) / (dyb * dlnd) - gainp
    gainfb = (gainfb1, gainfb2, gainfb3)
    rr = wbecb[2]
    fspb2 = fspcb[1]
    yyd_new = AGRAV * alcomx - fspb2
    yy = integrate(yyd_new, yyd, yy, int_step)
    yyd = yyd_new
    drc = -gainfb1 * fspb2 - gainfb2 * rr + gainfb3 * yy + gainp * yyd
    drcx = drc * DEG
    return dqcx, drcx, wacl, zacl, pacl, zz, zzd, yy, yyd, gainfb


def test_maut_3_ancomx_test_dqcx_finite_wacl_from_realq1():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.set("maut", 3)
    assert store.get("ancomx_test") == 1.0
    assert _approx(store.get("ancomx"), 0.0)
    assert _approx(store.get("alcomx"), 0.0)
    assert _approx(store.get("realq1"), -10.0)
    assert _approx(store.get("realq2"), -5.0)
    assert _approx(store.get("wacl_bias"), 0.0)
    ctrl.execute(vehicle, _ctx(0.001))
    assert np.isfinite(store.get("dqcx"))
    assert store.get("dqcx") != 0.0
    want_wacl = 10.0 * (1.0 + store.get("wacl_bias"))
    assert want_wacl == 10.0
    assert store.get("wacl") == pytest.approx(want_wacl, rel=RTOL, abs=ATOL)


def test_maut_3_calls_roll_and_accel_not_rate():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.set("maut", 3)
    ctrl.execute(vehicle, _ctx())
    assert np.isfinite(store.get("dpcx"))
    assert store.get("dpcx") != 0.0
    assert _approx(store.get("zrate"), 0.0)
    assert _approx(store.get("grate"), 0.0)
    assert _approx(store.get("wnlagr"), 0.0)
    assert _approx(store.get("dqcx_rcs"), 0.0)


def test_maut_3_matches_cadac_accel_and_poles():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.set("maut", 3)
    want = _cpp_accel(store, DT)
    ctrl.execute(vehicle, _ctx())
    dqcx, drcx, wacl, zacl, pacl, zz, zzd, yy, yyd, gainfb = want
    assert _approx(store.get("dqcx"), dqcx)
    assert _approx(store.get("drcx"), drcx)
    assert _approx(store.get("wacl"), wacl)
    assert _approx(store.get("zacl"), zacl)
    assert _approx(store.get("pacl"), pacl)
    assert _approx(store.get("zz"), zz)
    assert _approx(store.get("zzd"), zzd)
    assert _approx(store.get("yy"), yy)
    assert _approx(store.get("yyd"), yyd)
    np.testing.assert_allclose(store.get("GAINFB"), gainfb, rtol=RTOL, atol=ATOL)
    assert _approx(zacl, 0.7)
    assert _approx(wacl, 10.0)
    assert _approx(pacl, 40.0)
    assert _approx(store.get("ancomx"), 0.0)
    assert _approx(store.get("alcomx"), 0.0)


def test_control_accel_pole_biases():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.set("wacl_bias", 0.5)
    store.set("pacl_bias", -0.2)
    store.set("zacl_bias", 0.1)
    want = _cpp_accel(store, DT)
    ctrl.control_accel(vehicle, DT)
    assert _approx(store.get("wacl"), abs(REALQ1) * (1 + 0.5))
    assert _approx(store.get("wacl"), 15.0)
    assert _approx(store.get("zacl"), 0.7 * (1 + 0.1))
    assert _approx(store.get("pacl"), (abs(REALQ2) + 35) * (1 - 0.2))
    assert _approx(store.get("dqcx"), want[0])
    assert _approx(store.get("drcx"), want[1])


def test_control_accel_uses_realq_not_realr():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.define(Field("realr1", -99.0, "real", "diag", "aerodynamics"))
    store.define(Field("realr2", -1.0, "real", "diag", "aerodynamics"))
    ctrl.control_accel(vehicle, DT)
    assert store.get("wacl") == pytest.approx(10.0, rel=RTOL, abs=ATOL)
    assert store.get("pacl") == pytest.approx(40.0, rel=RTOL, abs=ATOL)
    decoy = abs(-99.0) * (1.0 + store.get("wacl_bias"))
    assert store.get("wacl") != pytest.approx(decoy, rel=RTOL, abs=ATOL)


def test_control_accel_integrates_zz_yy_stored_slope():
    vehicle, ctrl = _ready()
    store = vehicle.store
    ctrl.control_accel(vehicle, DT)
    zz1 = store.get("zz")
    zzd1 = store.get("zzd")
    yy1 = store.get("yy")
    yyd1 = store.get("yyd")
    assert zz1 != 0.0
    assert zzd1 != 0.0
    assert yy1 != 0.0
    want = _cpp_accel(store, DT)
    ctrl.control_accel(vehicle, DT)
    assert _approx(store.get("zz"), want[5])
    assert _approx(store.get("zzd"), want[6])
    assert _approx(store.get("yy"), want[7])
    assert _approx(store.get("yyd"), want[8])
    trapezoid = zz1 + (store.get("zzd") + zzd1) * DT / 2
    assert _approx(store.get("zz"), trapezoid)
    assert store.get("zz") != pytest.approx(zz1, rel=RTOL, abs=ATOL)
    assert store.get("yy") != pytest.approx(yy1, rel=RTOL, abs=ATOL)


def test_ancomx_test_changes_pitch_command():
    vehicle_on, ctrl_on = _ready()
    ctrl_on.control_accel(vehicle_on, DT)
    dqcx_on = vehicle_on.store.get("dqcx")
    vehicle_off, ctrl_off = _ready()
    vehicle_off.store.set("ancomx_test", 0.0)
    ctrl_off.control_accel(vehicle_off, DT)
    assert dqcx_on != pytest.approx(
        vehicle_off.store.get("dqcx"), rel=RTOL, abs=ATOL
    )


def test_execute_skips_absent_factwacl_twcl():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.set("maut", 3)
    assert "factwacl" not in store.names()
    assert "twcl" not in store.names()
    ctrl.execute(vehicle, _ctx())
    assert np.isfinite(store.get("dqcx"))
    assert store.get("dqcx") != 0.0


def test_unknown_maut_still_rolls():
    vehicle_u, ctrl_u = _ready()
    store_u = vehicle_u.store
    store_u.set("maut", 5)
    ctrl_u.execute(vehicle_u, _ctx())
    assert np.isfinite(store_u.get("dpcx"))
    assert store_u.get("dpcx") != 0.0
    assert _approx(store_u.get("dqcx"), 0.0)
    assert _approx(store_u.get("wacl"), 0.0)


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.flat6.sam6.control as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src
    assert "np.sign" not in src
    assert "_cadac_sign" not in src
