from math import sqrt
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.falcon6.control import Plane6Control

RTOL = 1e-12
ATOL = 1e-14

WRCL = 15.0
ZRCL = 0.7
DLLP = -2.5
DLLDA = 18.0
PHICOMX = 10.0
PHIBLX = 0.0
PPX = 3.0
TP = 0.5
PCOMX = 8.0
ZETLAGR = 0.7
DLA = 8.0
DLDE = -12.0
DMA = -2.0
DMQ = -1.5
DMDE = -8.0
QQX = 4.0
DVBE = 180.0
DYB = -6.0
DYDR = 4.0
DNB = 1.5
DNR = -0.8
DNDR = -3.0
RRX = 2.0
RCOMX = 0.0
PGAM = 10.0
WGAM = 3.0
ZGAM = 0.5
THTVLCOMX = 1.0
THTBLX = 1.0
THTVLX = 0.5
DALIMX = 20.0
DELIMX = 20.0
DRLIMX = 20.0
PHILIMX = 70.0
ANCOMX = 1.5


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _ctx(int_step=0.001):
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
    phiblx=PHIBLX,
    ppx=PPX,
    dllp=DLLP,
    dllda=DLLDA,
    dla=DLA,
    dlde=DLDE,
    dma=DMA,
    dmq=DMQ,
    dmde=DMDE,
    qqx=QQX,
    dvbe=DVBE,
    dyb=DYB,
    dydr=DYDR,
    dnb=DNB,
    dnr=DNR,
    dndr=DNDR,
    rrx=RRX,
    thtblx=THTBLX,
    thtvlx=THTVLX,
):
    store.define(Field("phiblx", phiblx, "real", "diag", "kinematics"))
    store.define(Field("ppx", ppx, "real", "out", "euler"))
    store.define(Field("dllp", dllp, "real", "out", "aerodynamics"))
    store.define(Field("dllda", dllda, "real", "out", "aerodynamics"))
    store.define(Field("dla", dla, "real", "out", "aerodynamics"))
    store.define(Field("dlde", dlde, "real", "out", "aerodynamics"))
    store.define(Field("dma", dma, "real", "out", "aerodynamics"))
    store.define(Field("dmq", dmq, "real", "out", "aerodynamics"))
    store.define(Field("dmde", dmde, "real", "out", "aerodynamics"))
    store.define(Field("qqx", qqx, "real", "out", "euler"))
    store.define(Field("dvbe", dvbe, "real", "out", "newton"))
    store.define(Field("dyb", dyb, "real", "out", "aerodynamics"))
    store.define(Field("dydr", dydr, "real", "out", "aerodynamics"))
    store.define(Field("dnb", dnb, "real", "out", "aerodynamics"))
    store.define(Field("dnr", dnr, "real", "out", "aerodynamics"))
    store.define(Field("dndr", dndr, "real", "out", "aerodynamics"))
    store.define(Field("rrx", rrx, "real", "out", "euler"))
    store.define(Field("thtblx", thtblx, "real", "diag", "kinematics"))
    store.define(Field("thtvlx", thtvlx, "real", "out", "newton"))


def _ready(**store_kw):
    vehicle = SimpleNamespace(store=StateStore())
    ctrl = Plane6Control()
    ctrl.define(vehicle)
    _externals(vehicle.store)
    store = vehicle.store
    store.set("wrcl", WRCL)
    store.set("zrcl", ZRCL)
    store.set("tp", TP)
    store.set("zetlagr", ZETLAGR)
    store.set("pgam", PGAM)
    store.set("wgam", WGAM)
    store.set("zgam", ZGAM)
    store.set("dalimx", DALIMX)
    store.set("delimx", DELIMX)
    store.set("drlimx", DRLIMX)
    store.set("philimx", PHILIMX)
    store.set("phicomx", PHICOMX)
    store.set("pcomx", PCOMX)
    store.set("rcomx", RCOMX)
    store.set("thtvlcomx", THTVLCOMX)
    store.set("ancomx", ANCOMX)
    for name, value in store_kw.items():
        store.set(name, value)
    ctrl.initialize(vehicle, _ctx())
    return vehicle, ctrl


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _control_roll(store, phicomx):
    wrcl = store.get("wrcl")
    zrcl = store.get("zrcl")
    phiblx = store.get("phiblx")
    ppx = store.get("ppx")
    dllp = store.get("dllp")
    dllda = store.get("dllda")
    gkp = (2.0 * zrcl * wrcl + dllp) / dllda
    gkphi = wrcl * wrcl / dllda
    ephi = gkphi * (phicomx - phiblx) * RAD
    dpc = ephi - gkp * ppx * RAD
    delacx = dpc * DEG
    return delacx, gkp, gkphi


def _control_roll_rate(store, pcomx):
    tp = store.get("tp")
    ppx = store.get("ppx")
    dllp = store.get("dllp")
    dllda = store.get("dllda")
    kp = (1 / tp + dllp) / dllda
    return kp * (pcomx - ppx)


def _control_yaw_rate(store, rcomx):
    zetlagr = store.get("zetlagr")
    dyb = store.get("dyb")
    dydr = store.get("dydr")
    dnb = store.get("dnb")
    dnr = store.get("dnr")
    dndr = store.get("dndr")
    rrx = store.get("rrx")
    dvbe = store.get("dvbe")
    zrate = -dyb / dvbe + dnb * dydr / (dvbe * dndr)
    aa = -dyb / dvbe - dnr
    bb = dnb + dyb * dnr / dvbe
    dum1 = aa - 2.0 * zetlagr * zetlagr * zrate
    dum2 = aa * aa - 4.0 * zetlagr * zetlagr * bb
    radix = dum1 * dum1 - dum2
    if radix < 0.0:
        radix = 0.0
    if abs(dndr) < 1.0e-7:
        dndr = 1.0e-7 * _sign(dndr)
    grate = (-dum1 + sqrt(radix)) / (-dndr)
    dum3 = grate * dndr * zrate
    radix = bb + dum3
    if radix < 0.0:
        radix = 0.0
    wnlagr = sqrt(radix)
    delrcx = grate * (rrx - rcomx)
    return delrcx, zrate, grate, wnlagr


def _control_gamma(store, thtvlcomx):
    pgam = store.get("pgam")
    wgam = store.get("wgam")
    zgam = store.get("zgam")
    thtblx = store.get("thtblx")
    qqx = store.get("qqx")
    thtvlx = store.get("thtvlx")
    dvbe = store.get("dvbe")
    dla = store.get("dla")
    dlde = store.get("dlde")
    dma = store.get("dma")
    dmq = store.get("dmq")
    dmde = store.get("dmde")

    aa = np.array(
        [
            [dmq, dma, -dma],
            [1.0, 0.0, 0.0],
            [0.0, dla / dvbe, -dla / dvbe],
        ],
        dtype=float,
    )
    bb = np.array([dmde, 0.0, dlde / dvbe], dtype=float)

    am = 2.0 * zgam * wgam + pgam
    bm = wgam * wgam + 2.0 * zgam * wgam * pgam
    cm = wgam * wgam * pgam
    v11 = dmde
    v12 = 0.0
    v13 = dlde / dvbe
    v21 = dmde * dla / dvbe - dlde * dma / dvbe
    v22 = dmde
    v23 = -dmq * dlde / dvbe
    v31 = 0.0
    v32 = v21
    v33 = v21
    dp = np.array(
        [
            [v11, v12, v13],
            [v21, v22, v23],
            [v31, v32, v33],
        ],
        dtype=float,
    )
    dd = np.array(
        [am + dmq - dla / dvbe, bm + dma + dmq * dla / dvbe, cm],
        dtype=float,
    )
    gaingam = np.linalg.inv(dp) @ dd
    dum33 = aa - np.outer(bb, gaingam)
    dum3 = np.linalg.inv(dum33) @ bb
    hh = np.array([0.0, 0.0, 1.0], dtype=float)
    gainff = -1.0 / (hh @ dum3)

    thtc = gainff * thtvlcomx * RAD
    qqf = gaingam[0] * qqx * RAD
    thtblf = gaingam[1] * thtblx * RAD
    thtvlf = gaingam[2] * thtvlx * RAD
    delec = thtc - (qqf + thtblf + thtvlf)
    delecx = delec * DEG
    return delecx, gaingam, gainff


def _dispatch(store):
    delacx = 0.0
    delecx = 0.0
    delrcx = 0.0
    maut = store.get("maut")
    mroll = store.get("mroll")
    dalimx = store.get("dalimx")
    delimx = store.get("delimx")
    drlimx = store.get("drlimx")
    philimx = store.get("philimx")
    ancomx = store.get("ancomx")
    phicomx = store.get("phicomx")
    pcomx = store.get("pcomx")
    rcomx = store.get("rcomx")
    thtvlcomx = store.get("thtvlcomx")
    if maut == 0:
        return None
    if maut not in (1, 24):
        raise ValueError(f"unknown maut {maut}")
    mauty = maut // 10
    mautp = maut % 10
    if mauty == 2:
        delrcx = _control_yaw_rate(store, rcomx)[0]
    if mautp == 4:
        delecx = _control_gamma(store, thtvlcomx)[0]
    if mroll == 0:
        if abs(phicomx) > philimx:
            phicomx = philimx * _sign(phicomx)
        delacx = _control_roll(store, phicomx)[0]
    elif mroll == 1:
        delacx = _control_roll_rate(store, pcomx)
    else:
        raise ValueError(f"unknown mroll {mroll}")
    if abs(delacx) > dalimx:
        delacx = dalimx * _sign(delacx)
    if abs(delecx) > delimx:
        delecx = delimx * _sign(delecx)
    if abs(delrcx) > drlimx:
        delrcx = drlimx * _sign(delrcx)
    return delacx, delecx, delrcx, ancomx, phicomx


def test_maut_24_runs_without_error_and_matches_cadac():
    vehicle, ctrl = _ready(maut=24)
    store = vehicle.store
    want = _dispatch(store)
    ctrl.execute(vehicle, _ctx())
    assert want is not None
    delacx, delecx, delrcx, ancomx, phicomx = want
    assert _approx(store.get("delacx"), delacx)
    assert _approx(store.get("delecx"), delecx)
    assert _approx(store.get("delrcx"), delrcx)
    assert _approx(store.get("ancomx"), ancomx)
    assert _approx(store.get("phicomx"), phicomx)
    assert store.get("delacx") != 0.0
    assert store.get("delecx") != 0.0
    assert store.get("delrcx") != 0.0
    np.testing.assert_allclose(
        store.get("GAINGAM"), _control_gamma(store, THTVLCOMX)[1], rtol=RTOL, atol=ATOL
    )
    assert _approx(store.get("gainff"), _control_gamma(store, THTVLCOMX)[2])
    _, zrate, grate, wnlagr = _control_yaw_rate(store, RCOMX)
    assert _approx(store.get("zrate"), zrate)
    assert _approx(store.get("grate"), grate)
    assert _approx(store.get("wnlagr"), wnlagr)
    _, gkp, gkphi = _control_roll(store, PHICOMX)
    assert _approx(store.get("gkp"), gkp)
    assert _approx(store.get("gkphi"), gkphi)


def test_maut_minus_one_raises():
    vehicle, ctrl = _ready(maut=-1)
    with pytest.raises(ValueError, match="unknown maut"):
        ctrl.execute(vehicle, _ctx())
    assert vehicle.store.get("delacx") == 0.0
    assert vehicle.store.get("delecx") == 0.0
    assert vehicle.store.get("delrcx") == 0.0


def test_maut_zero_returns_without_writing_commands():
    vehicle, ctrl = _ready(maut=0)
    store = vehicle.store
    store.set("delacx", 7.0)
    store.set("delecx", 8.0)
    store.set("delrcx", 9.0)
    store.set("phicomx", 80.0)
    store.set("ancomx", 3.0)
    ctrl.execute(vehicle, _ctx())
    assert store.get("delacx") == 7.0
    assert store.get("delecx") == 8.0
    assert store.get("delrcx") == 9.0
    assert store.get("phicomx") == 80.0
    assert store.get("ancomx") == 3.0
    np.testing.assert_array_equal(store.get("GAINGAM"), np.zeros(3))
    assert store.get("gainff") == 0.0
    assert store.get("gkp") == 0.0
    assert store.get("zrate") == 0.0


def test_maut_zero_ignores_unknown_mroll():
    vehicle, ctrl = _ready(maut=0, mroll=9)
    ctrl.execute(vehicle, _ctx())
    assert vehicle.store.get("delacx") == 0.0


def test_maut_1_is_roll_only():
    vehicle, ctrl = _ready(maut=1)
    store = vehicle.store
    want = _dispatch(store)
    ctrl.execute(vehicle, _ctx())
    delacx, delecx, delrcx, ancomx, phicomx = want
    assert _approx(store.get("delacx"), delacx)
    assert store.get("delecx") == 0.0
    assert store.get("delrcx") == 0.0
    assert _approx(store.get("delecx"), delecx)
    assert _approx(store.get("delrcx"), delrcx)
    assert _approx(store.get("ancomx"), ancomx)
    assert _approx(store.get("phicomx"), phicomx)
    assert store.get("delacx") != 0.0
    np.testing.assert_array_equal(store.get("GAINGAM"), np.zeros(3))
    assert store.get("gainff") == 0.0
    assert store.get("zrate") == 0.0
    assert store.get("grate") == 0.0
    assert store.get("wnlagr") == 0.0
    _, gkp, gkphi = _control_roll(store, PHICOMX)
    assert _approx(store.get("gkp"), gkp)
    assert _approx(store.get("gkphi"), gkphi)


def test_gamma_case_omits_mroll_defaults_to_position():
    vehicle, ctrl = _ready(maut=24)
    assert vehicle.store.get("mroll") == 0
    want = _dispatch(vehicle.store)
    ctrl.execute(vehicle, _ctx())
    assert _approx(vehicle.store.get("delacx"), want[0])
    _, gkp, gkphi = _control_roll(vehicle.store, PHICOMX)
    assert _approx(vehicle.store.get("gkp"), gkp)
    assert _approx(vehicle.store.get("gkphi"), gkphi)


def test_unknown_maut_raises():
    for maut in (2, 3, 4, 5, 20, 23, 25, 34, 44, 45):
        vehicle, ctrl = _ready(maut=maut)
        with pytest.raises(ValueError, match="unknown maut"):
            ctrl.execute(vehicle, _ctx())
        assert vehicle.store.get("delacx") == 0.0
        assert vehicle.store.get("delecx") == 0.0
        assert vehicle.store.get("delrcx") == 0.0


def test_mroll_1_uses_roll_rate():
    vehicle, ctrl = _ready(maut=1, mroll=1)
    store = vehicle.store
    want = _dispatch(store)
    ctrl.execute(vehicle, _ctx())
    delacx, delecx, delrcx, ancomx, phicomx = want
    assert _approx(store.get("delacx"), delacx)
    assert _approx(store.get("delacx"), _control_roll_rate(store, PCOMX))
    assert store.get("gkp") == 0.0
    assert store.get("gkphi") == 0.0
    assert store.get("phicomx") == PHICOMX
    assert _approx(store.get("delecx"), delecx)
    assert _approx(store.get("delrcx"), delrcx)
    assert _approx(store.get("ancomx"), ancomx)
    assert _approx(store.get("phicomx"), phicomx)


def test_unknown_mroll_raises():
    vehicle, ctrl = _ready(maut=1, mroll=2)
    with pytest.raises(ValueError, match="unknown mroll"):
        ctrl.execute(vehicle, _ctx())
    assert vehicle.store.get("delacx") == 0.0


def test_philimx_clamps_phicomx_with_cadac_sign():
    vehicle, ctrl = _ready(maut=1, phicomx=80.0)
    want = _dispatch(vehicle.store)
    ctrl.execute(vehicle, _ctx())
    assert vehicle.store.get("phicomx") == 70.0
    assert _approx(vehicle.store.get("delacx"), want[0])
    assert _approx(vehicle.store.get("phicomx"), want[4])

    vehicle_n, ctrl_n = _ready(maut=1, phicomx=-80.0)
    want_n = _dispatch(vehicle_n.store)
    ctrl_n.execute(vehicle_n, _ctx())
    assert vehicle_n.store.get("phicomx") == -70.0
    assert _approx(vehicle_n.store.get("delacx"), want_n[0])
    assert _sign(0.0) == 1
    assert np.sign(0.0) == 0.0


def test_surface_limiters_use_cadac_sign():
    vehicle, ctrl = _ready(maut=24, dalimx=0.01, delimx=0.01, drlimx=0.01)
    want = _dispatch(vehicle.store)
    ctrl.execute(vehicle, _ctx())
    assert _approx(vehicle.store.get("delacx"), want[0])
    assert _approx(vehicle.store.get("delecx"), want[1])
    assert _approx(vehicle.store.get("delrcx"), want[2])
    assert abs(vehicle.store.get("delacx")) == 0.01
    assert abs(vehicle.store.get("delecx")) == 0.01
    assert abs(vehicle.store.get("delrcx")) == 0.01


def test_dt_unused():
    vehicle_a, ctrl_a = _ready(maut=24)
    vehicle_b, ctrl_b = _ready(maut=24)
    ctrl_a.execute(vehicle_a, _ctx(int_step=0.001))
    ctrl_b.execute(vehicle_b, _ctx(int_step=0.05))
    assert _approx(vehicle_a.store.get("delacx"), vehicle_b.store.get("delacx"))
    assert _approx(vehicle_a.store.get("delecx"), vehicle_b.store.get("delecx"))
    assert _approx(vehicle_a.store.get("delrcx"), vehicle_b.store.get("delrcx"))
