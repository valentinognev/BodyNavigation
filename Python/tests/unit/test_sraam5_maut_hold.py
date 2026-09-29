"""SRAAM5 C2 MAUTP/MAUTL=1 α/β hold (ALPHAC/BETAC) — Fortran MODULE.FOR C2."""

from math import acos, asin, atan, atan2, cos, sin, tan
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import DEG
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat5.sraam5.control import Sraam5Control

RTOL = 1e-12
ATOL = 1e-14

MAUT = 11  # MAUTL=1 beta hold, MAUTP=1 alpha hold
MTURN = 0
ALPHAC = 0.12
BETAC = -0.07
ALPPLIM = 0.8
# Planted rate-loop states deliberately offset from commands
ALP = 0.4
BET = 0.3
PDYNMC = 4.0e4
FACTGACP = 0.5
FACTTR = 0.0
INT_STEP = 0.0123


def _ctx(int_step=INT_STEP):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _fortran_stt_polar(alph, bett, alpplim):
    """Fortran C2 total-AoA limiter → body ALPHA/BETA from ALPH/BETT."""
    alphap = acos(cos(alph) * cos(bett))
    if alphap > alpplim:
        alphap = alpplim
    if alphap < 1.0e-10:
        phip = 0.0
    else:
        phip = atan2(tan(bett), sin(alph))
    alpha = atan(cos(phip) * tan(alphap))
    beta = asin(sin(phip) * sin(alphap))
    return alpha, beta, alphap, phip


def _ready(**kw):
    vehicle = SimpleNamespace(store=StateStore())
    ctrl = Sraam5Control()
    ctrl.define(vehicle)
    store = vehicle.store
    values = dict(
        maut=MAUT,
        mturn=MTURN,
        alphac=ALPHAC,
        betac=BETAC,
        alpplim=ALPPLIM,
        alplim=0.8,
        alnlim=0.8,
        betlim=0.8,
        anplim=50.0,
        annlim=-50.0,
        allim=50.0,
        factgacp=FACTGACP,
        facttr=FACTTR,
        ta=2.2,
        pdynmc=PDYNMC,
        dvbe=400.0,
        amass=91.7,
        area=0.01824,
        cnalp=12.0,
        cybet=-11.0,
        fthalt=0.0,
        FSPCB=(1.0, 5.0, -20.0),
        ancom=3.0,
        alcom=-2.0,
        alp=ALP,
        alpd=0.0,
        bet=BET,
        betd=0.0,
        ratep=0.05,
        ratepd=0.0,
        ratey=-0.03,
        rateyd=0.0,
        xi=0.01,
        xid=0.0,
        yi=-0.02,
        yid=0.0,
    )
    values.update(kw)
    for name, value in values.items():
        if name in store:
            store.set(name, value)
        else:
            ftype = (
                "vec"
                if name == "FSPCB"
                else ("int" if name in ("maut", "mturn") else "real")
            )
            store.define(Field(name, value, ftype, "data", "ext"))
    ctrl.execute(vehicle, _ctx())
    return vehicle.store, ctrl


def test_maut11_mturn0_holds_alphac_betac():
    """MAUT=11 + MTURN=0: ALPH=ALPHAC, BETT=BETAC (not rate-loop ALP/BET)."""
    want_alpha, want_beta, want_alphap, want_phip = _fortran_stt_polar(
        ALPHAC, BETAC, ALPPLIM
    )
    # Rate-loop plant would produce a different polar pair
    plant_alpha, plant_beta, _, _ = _fortran_stt_polar(ALP, BET, ALPPLIM)
    assert plant_alpha != pytest.approx(want_alpha, abs=1e-6)
    assert plant_beta != pytest.approx(want_beta, abs=1e-6)

    store, _ = _ready()

    np.testing.assert_allclose(store.get("alpha"), want_alpha, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("beta"), want_beta, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("alphap"), want_alphap, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("phip"), want_phip, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("alphax"), DEG * want_alpha, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("betax"), DEG * want_beta, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("phibv"), 0.0, rtol=RTOL, atol=ATOL)
    # Hold path does not advance rate-loop integral states
    np.testing.assert_allclose(store.get("alp"), ALP, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("bet"), BET, rtol=RTOL, atol=ATOL)
