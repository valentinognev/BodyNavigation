from math import atan2, cos, sin, sqrt
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import AGRAV, DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.agm6.control import SMALL, Agm6Control

RTOL = 1e-12
ATOL = 1e-14

WACL = 2.0
ZACL = 0.7
PACL = 10.0
ALIMIT = 3.0
ANCOMX = 1.0
ALCOMX = 0.0
DQLIMX = 25.0
DRLIMX = 25.0
DPLIMX = 25.0
GAINP = 0.0
DT = 0.001

WRCL = 5.0
ZRCL = 0.9
PHICOMX = 0.0
PHIBLCX = 2.0
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
WBECB = (0.1, 0.05, -0.03)
FSPCB = (0.0, 0.5, -9.8)


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _ctx(int_step=DT):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _externals(
    store,
    *,
    wbecb=WBECB,
    fspcb=FSPCB,
    phiblcx=PHIBLCX,
    dlp=DLP,
    dld=DLD,
    dna=DNA,
    dnd=DND,
    dma=DMA,
    dmq=DMQ,
    dmd=DMD,
    dvbe=DVBE,
    ancomx=ANCOMX,
    alcomx=ALCOMX,
):
    store.define(Field("WBECB", wbecb, "vec", "out", "ins"))
    store.define(Field("FSPCB", fspcb, "vec", "out", "ins"))
    store.define(Field("phiblcx", phiblcx, "real", "out", "ins"))
    store.define(Field("dlp", dlp, "real", "out", "aerodynamics"))
    store.define(Field("dld", dld, "real", "out", "aerodynamics"))
    store.define(Field("dna", dna, "real", "out", "aerodynamics"))
    store.define(Field("dnd", dnd, "real", "out", "aerodynamics"))
    store.define(Field("dma", dma, "real", "out", "aerodynamics"))
    store.define(Field("dmq", dmq, "real", "out", "aerodynamics"))
    store.define(Field("dmd", dmd, "real", "out", "aerodynamics"))
    store.define(Field("dvbe", dvbe, "real", "out", "newton"))
    store.define(Field("ancomx", ancomx, "real", "data", "guidance"))
    store.define(Field("alcomx", alcomx, "real", "data", "guidance"))


def _ready(**kw):
    vehicle = SimpleNamespace(store=StateStore())
    ctrl = Agm6Control()
    ctrl.define(vehicle)
    _externals(vehicle.store)
    store = vehicle.store
    store.set("wacl", WACL)
    store.set("zacl", ZACL)
    store.set("pacl", PACL)
    store.set("alimit", ALIMIT)
    store.set("dqlimx", DQLIMX)
    store.set("drlimx", DRLIMX)
    store.set("dplimx", DPLIMX)
    store.set("phicomx", PHICOMX)
    store.set("wrcl", WRCL)
    store.set("zrcl", ZRCL)
    store.set("gainp", GAINP)
    store.set("zetlagr", ZETLAGR)
    store.set("qqcomx", QQCOMX)
    store.set("rrcomx", RRCOMX)
    for name, value in kw.items():
        store.set(name, value)
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


def _control_accel(store, int_step):
    wacl = store.get("wacl")
    zacl = store.get("zacl")
    pacl = store.get("pacl")
    alimit = store.get("alimit")
    dqlimx = store.get("dqlimx")
    drlimx = store.get("drlimx")
    gainp = store.get("gainp")
    dvbe = store.get("dvbe")
    ancomx = store.get("ancomx")
    alcomx = store.get("alcomx")
    dna = store.get("dna")
    dma = store.get("dma")
    dmq = store.get("dmq")
    dmd = store.get("dmd")
    fspcb = store.get("FSPCB")
    wbecb = store.get("WBECB")
    yyd = store.get("yyd")
    yy = store.get("yy")
    zzd = store.get("zzd")
    zz = store.get("zz")

    aa = sqrt(alcomx * alcomx + ancomx * ancomx)
    if aa > alimit:
        aa = alimit
    if abs(ancomx) < SMALL and abs(alcomx) < SMALL:
        phi = 0.0
    else:
        phi = atan2(ancomx, alcomx)
    alcomx = aa * cos(phi)
    ancomx = aa * sin(phi)

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

    rr = wbecb[2]
    fspb2 = fspcb[1]
    yyd_new = AGRAV * alcomx - fspb2
    yy = integrate(yyd_new, yyd, yy, int_step)
    yyd = yyd_new
    drc = -gainfb1 * fspb2 - gainfb2 * rr + gainfb3 * yy + gainp * yyd
    drcx = drc * DEG

    if abs(dqcx) > dqlimx:
        dqcx = dqlimx * _sign(dqcx)
    if abs(drcx) > drlimx:
        drcx = drlimx * _sign(drcx)

    gainfb = (gainfb1, gainfb2, gainfb3)
    return dqcx, drcx, yyd, yy, zzd, zz, gainfb


def test_maut_3_finite_dqcx_matches_cadac():
    vehicle, ctrl = _ready(maut=3)
    store = vehicle.store
    want_q, want_r, yyd, yy, zzd, zz, gainfb = _control_accel(store, DT)
    want_p, gkp, gkphi = _control_roll(store)
    assert np.isfinite(want_q)
    assert want_q != 0.0
    assert ctrl.execute(vehicle, _ctx()) is None
    assert _approx(store.get("dqcx"), want_q)
    assert _approx(store.get("drcx"), want_r)
    assert _approx(store.get("dpcx"), want_p)
    assert store.get("dpcx") != 0.0
    assert np.isfinite(store.get("dqcx"))
    assert abs(store.get("dqcx")) <= DQLIMX
    assert abs(store.get("drcx")) <= DRLIMX
    assert _approx(store.get("yyd"), yyd)
    assert _approx(store.get("yy"), yy)
    assert _approx(store.get("zzd"), zzd)
    assert _approx(store.get("zz"), zz)
    np.testing.assert_allclose(store.get("GAINFB"), gainfb, rtol=RTOL, atol=ATOL)
    assert _approx(store.get("gkp"), gkp)
    assert _approx(store.get("gkphi"), gkphi)
    assert store.get("zrate") == 0.0
    assert store.get("grate") == 0.0
    assert store.get("wnlagr") == 0.0
    assert store.get("ancomx") == ANCOMX
    assert store.get("alcomx") == ALCOMX


def test_maut_zero_returns_without_writing_commands():
    vehicle, ctrl = _ready(maut=0)
    store = vehicle.store
    store.set("dpcx", 7.0)
    store.set("dqcx", 8.0)
    store.set("drcx", 9.0)
    store.set("gkp", 1.5)
    store.set("GAINFB", (1.0, 2.0, 3.0))
    store.set("zz", 4.0)
    ctrl.execute(vehicle, _ctx())
    assert store.get("dpcx") == 7.0
    assert store.get("dqcx") == 8.0
    assert store.get("drcx") == 9.0
    assert store.get("gkp") == 1.5
    np.testing.assert_array_equal(store.get("GAINFB"), (1.0, 2.0, 3.0))
    assert store.get("zz") == 4.0
    assert store.get("zrate") == 0.0


def test_maut_1_writes_dpcx_only():
    vehicle, ctrl = _ready(maut=1)
    store = vehicle.store
    want_p, gkp, gkphi = _control_roll(store)
    ctrl.execute(vehicle, _ctx())
    assert _approx(store.get("dpcx"), want_p)
    assert store.get("dpcx") != 0.0
    assert store.get("dqcx") == 0.0
    assert store.get("drcx") == 0.0
    assert _approx(store.get("gkp"), gkp)
    assert _approx(store.get("gkphi"), gkphi)
    assert store.get("zrate") == 0.0
    assert store.get("zz") == 0.0
    np.testing.assert_array_equal(store.get("GAINFB"), np.zeros(3))


def test_maut_2_roll_and_rate_not_accel():
    vehicle, ctrl = _ready(maut=2)
    store = vehicle.store
    want_p, gkp, gkphi = _control_roll(store)
    want_q, want_r, zrate, grate, wnlagr = _control_rate(store)
    ctrl.execute(vehicle, _ctx())
    assert _approx(store.get("dpcx"), want_p)
    assert _approx(store.get("dqcx"), want_q)
    assert _approx(store.get("drcx"), want_r)
    assert _approx(store.get("zrate"), zrate)
    assert _approx(store.get("grate"), grate)
    assert _approx(store.get("wnlagr"), wnlagr)
    assert _approx(store.get("gkp"), gkp)
    assert _approx(store.get("gkphi"), gkphi)
    assert store.get("zz") == 0.0
    assert store.get("yy") == 0.0
    np.testing.assert_array_equal(store.get("GAINFB"), np.zeros(3))


def test_maut_4_raises():
    vehicle, ctrl = _ready(maut=4)
    with pytest.raises(ValueError, match="unknown maut"):
        ctrl.execute(vehicle, _ctx())
    assert vehicle.store.get("dpcx") == 0.0
    assert vehicle.store.get("dqcx") == 0.0
    assert vehicle.store.get("drcx") == 0.0


def test_maut_minus_one_raises():
    vehicle, ctrl = _ready(maut=-1)
    with pytest.raises(ValueError, match="unknown maut"):
        ctrl.execute(vehicle, _ctx())
    assert vehicle.store.get("dpcx") == 0.0
    assert vehicle.store.get("dqcx") == 0.0
    assert vehicle.store.get("drcx") == 0.0


def test_unknown_maut_raises():
    for maut in (5, 99):
        vehicle, ctrl = _ready(maut=maut)
        with pytest.raises(ValueError, match="unknown maut"):
            ctrl.execute(vehicle, _ctx())
        assert vehicle.store.get("dpcx") == 0.0


def test_control_accel_matches_cadac_formulas():
    vehicle, ctrl = _ready()
    store = vehicle.store
    want_q, want_r, yyd, yy, zzd, zz, gainfb = _control_accel(store, DT)
    assert np.isfinite(want_q)
    assert want_q != 0.0
    assert ctrl.control_accel(vehicle, DT) is None
    assert _approx(store.get("dqcx"), want_q)
    assert _approx(store.get("drcx"), want_r)
    assert _approx(store.get("yyd"), yyd)
    assert _approx(store.get("yy"), yy)
    assert _approx(store.get("zzd"), zzd)
    assert _approx(store.get("zz"), zz)
    np.testing.assert_allclose(store.get("GAINFB"), gainfb, rtol=RTOL, atol=ATOL)
    assert store.get("dpcx") == 0.0
    assert store.get("ancomx") == ANCOMX
    assert store.get("alcomx") == ALCOMX


def test_control_accel_reads_ins_fspcb_not_fspb():
    vehicle, ctrl = _ready()
    vehicle.store.define(Field("FSPB", (9.0, 9.0, 9.0), "vec", "out", "newton"))
    want_q, want_r, _yyd, _yy, _zzd, _zz, _gainfb = _control_accel(vehicle.store, DT)
    ctrl.control_accel(vehicle, DT)
    assert _approx(vehicle.store.get("dqcx"), want_q)
    assert _approx(vehicle.store.get("drcx"), want_r)
    fspb3 = 9.0
    dna = DNA
    dma = DMA
    dmq = DMQ
    dmd = DMD
    dvbe = DVBE
    gainfb3 = WACL * WACL * PACL / (dna * dmd)
    gainfb2 = (2 * ZACL * WACL + PACL + dmq - dna / dvbe) / dmd
    gainfb1 = (
        WACL * WACL
        + 2 * ZACL * WACL * PACL
        + dma
        + dmq * dna / dvbe
        - gainfb2 * dna * dmd / dvbe
    ) / (dna * dmd)
    zzd_new = AGRAV * ANCOMX + fspb3
    zz = integrate(zzd_new, 0.0, 0.0, DT)
    dqc = -gainfb1 * (-fspb3) - gainfb2 * WBECB[1] + gainfb3 * zz
    wrong = dqc * DEG
    assert vehicle.store.get("dqcx") != pytest.approx(wrong, rel=RTOL, abs=ATOL)


def test_circular_alimit_scales_commands():
    vehicle_lim, ctrl_lim = _ready(ancomx=4.0, alcomx=0.0, alimit=3.0)
    vehicle_un, ctrl_un = _ready(ancomx=4.0, alcomx=0.0, alimit=100.0)
    want_lim = _control_accel(vehicle_lim.store, DT)
    want_un = _control_accel(vehicle_un.store, DT)
    ctrl_lim.control_accel(vehicle_lim, DT)
    ctrl_un.control_accel(vehicle_un, DT)
    assert _approx(vehicle_lim.store.get("dqcx"), want_lim[0])
    assert _approx(vehicle_un.store.get("dqcx"), want_un[0])
    assert vehicle_lim.store.get("dqcx") != pytest.approx(
        vehicle_un.store.get("dqcx"), rel=RTOL, abs=ATOL
    )
    assert vehicle_lim.store.get("ancomx") == 4.0
    assert vehicle_lim.store.get("alcomx") == 0.0


def test_command_limiter_uses_cadac_sign():
    vehicle, ctrl = _ready(dqlimx=0.01, drlimx=0.01)
    want_q, want_r, _yyd, _yy, _zzd, _zz, _gainfb = _control_accel(vehicle.store, DT)
    ctrl.control_accel(vehicle, DT)
    assert abs(want_q) == 0.01
    assert abs(want_r) == 0.01
    assert _approx(vehicle.store.get("dqcx"), want_q)
    assert _approx(vehicle.store.get("drcx"), want_r)
    assert abs(vehicle.store.get("dqcx")) == 0.01
    assert abs(vehicle.store.get("drcx")) == 0.01
    assert _sign(0.0) == 1
    assert np.sign(0.0) == 0.0


def test_control_accel_stored_slope_second_step():
    vehicle, ctrl = _ready()
    ctrl.control_accel(vehicle, DT)
    want_q, want_r, yyd, yy, zzd, zz, gainfb = _control_accel(vehicle.store, DT)
    ctrl.control_accel(vehicle, DT)
    assert _approx(vehicle.store.get("dqcx"), want_q)
    assert _approx(vehicle.store.get("drcx"), want_r)
    assert _approx(vehicle.store.get("yyd"), yyd)
    assert _approx(vehicle.store.get("yy"), yy)
    assert _approx(vehicle.store.get("zzd"), zzd)
    assert _approx(vehicle.store.get("zz"), zz)
    np.testing.assert_allclose(vehicle.store.get("GAINFB"), gainfb, rtol=RTOL, atol=ATOL)


def test_execute_passes_int_step_to_accel():
    vehicle_a, ctrl_a = _ready(maut=3)
    vehicle_b, ctrl_b = _ready(maut=3)
    ctrl_a.execute(vehicle_a, _ctx(int_step=0.001))
    ctrl_b.execute(vehicle_b, _ctx(int_step=0.05))
    assert vehicle_a.store.get("zz") != pytest.approx(
        vehicle_b.store.get("zz"), rel=RTOL, abs=ATOL
    )
    assert vehicle_a.store.get("dqcx") != pytest.approx(
        vehicle_b.store.get("dqcx"), rel=RTOL, abs=ATOL
    )


def test_does_not_require_time_or_pdynmc():
    vehicle, ctrl = _ready(maut=3)
    assert "time" not in vehicle.store.names()
    assert "pdynmc" not in vehicle.store.names()
    ctrl.execute(vehicle, _ctx())
    assert "time" not in vehicle.store.names()
    assert "pdynmc" not in vehicle.store.names()
    assert np.isfinite(vehicle.store.get("dqcx"))
