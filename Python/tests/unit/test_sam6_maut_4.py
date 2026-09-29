"""SAM6 maut=4: roll + rate + accel; RCS commands keep rate copies."""

from math import atan2, cos, fabs, sin, sqrt

import numpy as np
import pytest

from cadac.constants import AGRAV, DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.sam6.control import SMALL, Sam6Control

RTOL = 1e-12
ATOL = 1e-14

DT = 0.001
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
DYB = 35.0
DNB = -5.0
DNR = -1.2
DLND = -12.0
REALQ1 = -10.0
REALQ2 = -5.0
FSPCB = (0.0, 1.0, -10.0)
WBECB = (0.1, 0.05, -0.02)
THTBLCX = 80.0
PHIBLCX = 0.0
PHICOMX = 0.0
FACTWRCL = 0.0
TP = 0.5
ALIMITX = 50.0
GAINP = 0.0
ANCOMX_TEST = 1.0
ANCOMX = 0.0
ALCOMX = 0.0


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _sign(variable):
    if variable < 0:
        return -1
    return 1


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


def _plant(store):
    store.define(Field("dlp", DLP, "real", "out", "aerodynamics"))
    store.define(Field("dld", DLD, "real", "out", "aerodynamics"))
    store.define(Field("dna", DNA, "real", "out", "aerodynamics"))
    store.define(Field("dnd", DND, "real", "out", "aerodynamics"))
    store.define(Field("dma", DMA, "real", "out", "aerodynamics"))
    store.define(Field("dmq", DMQ, "real", "out", "aerodynamics"))
    store.define(Field("dmd", DMD, "real", "out", "aerodynamics"))
    store.define(Field("dyb", DYB, "real", "out", "aerodynamics"))
    store.define(Field("dnb", DNB, "real", "out", "aerodynamics"))
    store.define(Field("dnr", DNR, "real", "out", "aerodynamics"))
    store.define(Field("dlnd", DLND, "real", "out", "aerodynamics"))
    store.define(Field("realq1", REALQ1, "real", "diag", "aerodynamics"))
    store.define(Field("realq2", REALQ2, "real", "diag", "aerodynamics"))
    store.define(Field("dvbe", DVBE, "real", "out", "newton"))
    store.define(Field("FSPCB", FSPCB, "vec", "diag", "ins"))
    store.define(Field("WBECB", WBECB, "vec", "diag", "ins"))
    store.define(Field("thtblcx", THTBLCX, "real", "out", "ins"))
    store.define(Field("phiblcx", PHIBLCX, "real", "out", "ins"))
    store.define(Field("ancomx", ANCOMX, "real", "data", "guidance"))
    store.define(Field("alcomx", ALCOMX, "real", "data", "guidance"))


def _ready():
    vehicle = _Vehicle()
    ctrl = Sam6Control()
    ctrl.define(vehicle)
    ctrl.initialize(vehicle, _ctx())
    _plant(vehicle.store)
    store = vehicle.store
    store.set("zrcl", ZRCL)
    store.set("zetlagr", ZETLAGR)
    store.set("phicomx", PHICOMX)
    store.set("factwrcl", FACTWRCL)
    store.set("tp", TP)
    store.set("alimitx", ALIMITX)
    store.set("gainp", GAINP)
    store.set("ancomx_test", ANCOMX_TEST)
    store.set("alcomx_test", 0.0)
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


def test_maut_4_runs_roll_rate_accel_rcs_from_rate():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.set("maut", 4)

    dpcx, wrcl, gkp, gkphi = _cpp_roll(store)
    dqcx_rate, drcx_rate, zrate, grate, wnlagr = _cpp_rate(store)
    dqcx_acc, drcx_acc, wacl, zacl, pacl, zz, zzd, yy, yyd, gainfb = _cpp_accel(
        store, DT
    )

    ctrl.execute(vehicle, _ctx())

    # roll
    assert _approx(store.get("dpcx"), dpcx)
    assert _approx(store.get("wrcl"), wrcl)
    assert _approx(store.get("gkp"), gkp)
    assert _approx(store.get("gkphi"), gkphi)

    # accel overwrites flap commands
    assert _approx(store.get("dqcx"), dqcx_acc)
    assert _approx(store.get("drcx"), drcx_acc)
    assert _approx(store.get("wacl"), wacl)
    assert _approx(store.get("zacl"), zacl)
    assert _approx(store.get("pacl"), pacl)
    assert _approx(store.get("zz"), zz)
    assert _approx(store.get("zzd"), zzd)
    assert _approx(store.get("yy"), yy)
    assert _approx(store.get("yyd"), yyd)
    np.testing.assert_allclose(store.get("GAINFB"), gainfb, rtol=RTOL, atol=ATOL)

    # rate diagnostics + RCS copies survive accel (C++ control_rate then control_accel)
    assert _approx(store.get("zrate"), zrate)
    assert _approx(store.get("grate"), grate)
    assert _approx(store.get("wnlagr"), wnlagr)
    assert _approx(store.get("dqcx_rcs"), dqcx_rate)
    assert _approx(store.get("drcx_rcs"), drcx_rate)

    # flap commands differ from rate copies (accel won)
    assert store.get("dqcx") != pytest.approx(dqcx_rate, rel=RTOL, abs=ATOL)
    assert store.get("drcx") != pytest.approx(drcx_rate, rel=RTOL, abs=ATOL)
    assert dqcx_rate != 0.0
    assert dqcx_acc != 0.0


def test_maut_4_does_not_raise():
    vehicle, ctrl = _ready()
    vehicle.store.set("maut", 4)
    ctrl.execute(vehicle, _ctx())
    assert np.isfinite(vehicle.store.get("dqcx"))
    assert np.isfinite(vehicle.store.get("dqcx_rcs"))
