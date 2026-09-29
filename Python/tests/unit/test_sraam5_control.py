"""SRAAM5 C2 control — MAUT=44 accel autopilot (Fortran MODULE.FOR C2)."""

from math import acos, asin, atan, atan2, cos, sin, tan
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import AGRAV, DEG
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat5.sraam5.control import Sraam5Control

RTOL = 1e-12
ATOL = 1e-14

# Planted MAUT=44 sample (inlar1-like gains; nonzero accel commands)
MAUT = 44
ANCOM = 3.0
ALCOM = -2.0
ANPLIM = 50.0
ANNLIM = -50.0
ALLIM = 50.0
BETLIM = 0.8
ALPPLIM = 0.8
ALPLIM = 0.8
ALNLIM = 0.8
FACTGACP = 0.5
FACTTR = 0.0
TA = 2.2
PDYNMC = 4.0e4
DVBE = 400.0
AMASS = 91.7
AREA = 0.01824
CNALP = 12.0
CYBET = -11.0
FTHALT = 0.0
FSPCB = (1.0, 5.0, -20.0)
INT_STEP = 0.0123
XI = 0.01
YI = -0.02
RATEP = 0.05
RATEY = -0.03
ALP = 0.1
BET = -0.05


def _ctx(int_step=INT_STEP):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _fortran_maut44_pitch_yaw(
    *,
    ancom=ANCOM,
    alcom=ALCOM,
    anplim=ANPLIM,
    annlim=ANNLIM,
    allim=ALLIM,
    factgacp=FACTGACP,
    facttr=FACTTR,
    ta=TA,
    pdynmc=PDYNMC,
    dvbe=DVBE,
    amass=AMASS,
    area=AREA,
    cnalp=CNALP,
    cybet=CYBET,
    fthalt=FTHALT,
    fspcb=FSPCB,
    xi=XI,
    xid=0.0,
    yi=YI,
    yid=0.0,
    ratep=RATEP,
    ratepd=0.0,
    ratey=RATEY,
    rateyd=0.0,
    alp=ALP,
    alpd=0.0,
    bet=BET,
    betd=0.0,
    alplim=ALPLIM,
    alnlim=ALNLIM,
    betlim=BETLIM,
    alpplim=ALPPLIM,
    int_step=INT_STEP,
):
    """Hand-computed Fortran C2 MAUT=44 (MAUTA=0, MAUTL=4, MAUTP=4) + C2PTCH/C2YAW."""
    tr = (-2.0e-7 * pdynmc + 0.22) * (1.0 + facttr)
    gacp = (2.0e-3 * pdynmc) ** 0.575 * (1.0 + factgacp)

    if ancom > anplim:
        ancom = anplim
    if ancom < annlim:
        ancom = annlim
    abecz = -ancom * AGRAV
    ep = abecz - fspcb[2]
    tip_outer = dvbe * amass / (pdynmc * area * cnalp)
    gr_p = gacp * tip_outer * tr / dvbe
    gi_p = gr_p / ta
    xid_new = gi_p * ep
    pitch = -(ep * gr_p + xi)
    xi = integrate(xid_new, xid, xi, int_step)

    # C2PTCH simplified (TR>0)
    ratepd_new = (pitch - ratep) / tr
    tip = dvbe * amass / (pdynmc * area * cnalp + fthalt)
    alpd_new = (tip * ratep - alp) / tip
    alph = alp
    if alph > alplim:
        alph = alplim
    if alph < -alnlim:
        alph = -alnlim
    ratep = integrate(ratepd_new, ratepd, ratep, int_step)
    alp = integrate(alpd_new, alpd, alp, int_step)

    if alcom > allim:
        alcom = allim
    if alcom < -allim:
        alcom = -allim
    abecy = alcom * AGRAV
    ey = abecy - fspcb[1]
    tiy_outer = dvbe * amass / (-pdynmc * area * cybet)
    gr_y = gacp * tiy_outer * tr / dvbe
    gi_y = gr_y / ta
    yid_new = gi_y * ey
    yaw = ey * gr_y + yi
    yi = integrate(yid_new, yid, yi, int_step)

    # C2YAW simplified (TR>0)
    rateyd_new = (yaw - ratey) / tr
    tiy = dvbe * amass / (-pdynmc * area * cybet + fthalt)
    betd_new = -(tiy * ratey + bet) / tiy
    bett = bet
    if bett > betlim:
        bett = betlim
    if bett < -betlim:
        bett = -betlim
    ratey = integrate(rateyd_new, rateyd, ratey, int_step)
    bet = integrate(betd_new, betd, bet, int_step)

    alphap = acos(cos(alph) * cos(bett))
    if alphap > alpplim:
        alphap = alpplim
    if alphap < 1.0e-10:
        phip = 0.0
    else:
        phip = atan2(tan(bett), sin(alph))
    alpha = atan(cos(phip) * tan(alphap))
    beta = asin(sin(phip) * sin(alphap))

    return {
        "tr": tr,
        "gacp": gacp,
        "pitch": pitch,
        "yaw": yaw,
        "xid": xid_new,
        "yid": yid_new,
        "xi": xi,
        "yi": yi,
        "ratepd": ratepd_new,
        "rateyd": rateyd_new,
        "ratep": ratep,
        "ratey": ratey,
        "alpd": alpd_new,
        "betd": betd_new,
        "alp": alp,
        "bet": bet,
        "alpha": alpha,
        "beta": beta,
        "alphap": alphap,
        "phip": phip,
        "alphax": DEG * alpha,
        "betax": DEG * beta,
        "phibv": 0.0,
    }


def _ready(**kw):
    vehicle = SimpleNamespace(store=StateStore())
    ctrl = Sraam5Control()
    ctrl.define(vehicle)
    store = vehicle.store
    values = dict(
        maut=MAUT,
        anplim=ANPLIM,
        annlim=ANNLIM,
        allim=ALLIM,
        betlim=BETLIM,
        alpplim=ALPPLIM,
        alplim=ALPLIM,
        alnlim=ALNLIM,
        factgacp=FACTGACP,
        facttr=FACTTR,
        ta=TA,
        ancom=ANCOM,
        alcom=ALCOM,
        pdynmc=PDYNMC,
        dvbe=DVBE,
        amass=AMASS,
        area=AREA,
        cnalp=CNALP,
        cybet=CYBET,
        fthalt=FTHALT,
        FSPCB=FSPCB,
        mturn=0,
        xi=XI,
        xid=0.0,
        yi=YI,
        yid=0.0,
        ratep=RATEP,
        ratepd=0.0,
        ratey=RATEY,
        rateyd=0.0,
        alp=ALP,
        alpd=0.0,
        bet=BET,
        betd=0.0,
    )
    values.update(kw)
    for name, value in values.items():
        if name in store:
            store.set(name, value)
        else:
            ftype = "vec" if name == "FSPCB" else ("int" if name in ("maut", "mturn") else "real")
            store.define(Field(name, value, ftype, "data", "ext"))
    ctrl.execute(vehicle, _ctx())
    return vehicle.store, ctrl


def test_name_is_control():
    assert Sraam5Control().name == "control"


def test_maut44_accel_autopilot_sets_pitch_yaw_commands():
    """Fortran C2 MAUT=44: PITCH=-(EP*GR+XI), YAW=(EY*GR+YI); rate loops set alpha/beta."""
    want = _fortran_maut44_pitch_yaw()
    store, _ = _ready()

    np.testing.assert_allclose(store.get("xid"), want["xid"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("yid"), want["yid"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ratepd"), want["ratepd"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("rateyd"), want["rateyd"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("alpha"), want["alpha"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("beta"), want["beta"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("alphax"), want["alphax"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("betax"), want["betax"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("phibv"), 0.0, rtol=RTOL, atol=ATOL)
    # Pitch/yaw commands drive nonzero rate derivatives for this plant
    assert abs(want["pitch"]) > 0.0
    assert abs(want["yaw"]) > 0.0
    assert store.get("ratepd") != pytest.approx(0.0, abs=ATOL)
    assert store.get("rateyd") != pytest.approx(0.0, abs=ATOL)
