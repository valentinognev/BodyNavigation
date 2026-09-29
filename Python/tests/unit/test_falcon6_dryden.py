"""Task 80: FALCON6 Dryden turbulence — Fortran G2TURB (MAIR MTURB=1)."""

from math import cos, sin, sqrt
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import PI, RAD
from cadac.eom.flat6 import Flat6Environment
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.stoch import dryden_white, seed

RTOL = 1e-12
ATOL = 1e-12

DT = 0.01
HBE = 1524.0
DVBA = 183.0
TURB_LENGTH = 100.0
TURB_SIGMA = 0.5
# mair = |MTURB|MWIND|MATMO| → 100 = Dryden, no wind, US76
MAIR_MTURB1 = 100
ISEED = 12345


def _ctx(int_step=DT):
    return SimpleNamespace(int_step=int_step)


def _identity_tbl():
    return np.eye(3, dtype=float)


def _vehicle(
    *,
    mair=MAIR_MTURB1,
    dvba=DVBA,
    hbe=HBE,
    vbel=None,
    alpp=0.0,
    phip=0.0,
    turb_length=TURB_LENGTH,
    turb_sigma=TURB_SIGMA,
    tbl=None,
):
    if vbel is None:
        vbel = np.array([dvba, 0.0, 0.0], dtype=float)
    if tbl is None:
        tbl = _identity_tbl()
    s = StateStore()
    vehicle = SimpleNamespace(store=s)
    env = Flat6Environment()
    env.define(vehicle)
    s.define(Field("hbe", hbe, "real", "out", "newton"))
    s.define(Field("VBEL", vbel, "vec", "diag", "newton"))
    s.define(Field("TBL", tbl, "mat", "out", "kinematics"))
    s.define(Field("alpp", alpp, "real", "out", "kinematics"))
    s.define(Field("phip", phip, "real", "out", "kinematics"))
    s.set("mair", mair)
    s.set("dvba", dvba)
    s.set("turb_length", turb_length)
    s.set("turb_sigma", turb_sigma)
    return vehicle, env


def _g2turb_vtag(gauss_value, dvba, turb_length, turb_sigma, alpp, phip, tbl, int_step):
    """Hand transcription of FALCON6 MODULE.FOR G2TURB filter + TAB→TGA→VTAG."""
    taux1 = 0.0
    taux1d = 0.0
    taux2 = 0.0
    taux2d = 0.0
    # CADAC++ module form: integrate states then form tau (same filter ODEs as G2TURB).
    taux1d_new = taux2
    taux1 = integrate(taux1d_new, taux1d, taux1, int_step)
    taux1d = taux1d_new
    vl = dvba / turb_length
    taux2d_new = -vl * vl * taux1 - 2.0 * vl * taux2 + vl * vl * gauss_value
    taux2 = integrate(taux2d_new, taux2d, taux2, int_step)
    # Fortran: DUM1=SQRT(1/(PI*VL)); DUM2=(1/VL)*SQRT(3/(PI*VL))
    dum1 = sqrt(1.0 / (PI * vl))
    dum2 = (1.0 / vl) * sqrt(3.0 / (PI * vl))
    tau = turb_sigma * (dum1 * taux1 + dum2 * taux2)
    # VTAA = [0, 0, TAU]; TAB from ALPP/PHIP (rad); TAG=TAB*TBL; VTAG=TGA*VTAA
    cosa, sina = cos(alpp), sin(alpp)
    cosp, sinp = cos(phip), sin(phip)
    tab = np.array(
        [
            [cosa, sina * sinp, sina * cosp],
            [0.0, cosp, -sinp],
            [-sina, cosa * sinp, cosa * cosp],
        ],
        dtype=float,
    )
    vtaa = np.array([0.0, 0.0, tau], dtype=float)
    tag = tab @ tbl
    vtag = tag.T @ vtaa
    return vtag, tau, taux1, taux2


def test_falcon6_mturb1_burns_dryden_and_matches_g2turb_vtag():
    """mair=100 → MTURB=1: dryden_white draw + VTAG from G2TURB TAB path at planted DVBA."""
    alpp = 0.05  # rad
    phip = -0.02
    tbl = _identity_tbl()
    seed(ISEED)
    gauss_want = dryden_white(DT)
    vtag_want, tau_want, _, _ = _g2turb_vtag(
        gauss_want, DVBA, TURB_LENGTH, TURB_SIGMA, alpp, phip, tbl, DT
    )

    seed(ISEED)
    vehicle, env = _vehicle(alpp=alpp, phip=phip, tbl=tbl)
    env.execute(vehicle, _ctx())
    store = vehicle.store

    assert store.get("gauss_value") == pytest.approx(gauss_want, rel=RTOL, abs=ATOL)
    assert store.get("tau") == pytest.approx(tau_want, rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(store.get("VTAG"), vtag_want, rtol=RTOL, atol=ATOL)
    # Gust added into air-mass velocity (MWIND=0 → VAELS=0 → VAEL=VTAG)
    np.testing.assert_allclose(store.get("VAEL"), vtag_want, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("VBAL"),
        store.get("VBEL") - store.get("VAEL"),
        rtol=RTOL,
        atol=ATOL,
    )


def test_falcon6_mair0_no_dryden_without_mair_turb():
    """mair=0 keeps C++ mwind path: no Dryden, VAEL=0 when mwind=0."""
    vehicle, env = _vehicle(mair=0)
    env.execute(vehicle, _ctx())
    store = vehicle.store
    np.testing.assert_allclose(store.get("VAEL"), np.zeros(3), rtol=RTOL, atol=ATOL)
    assert store.get("tau") == 0.0


def test_falcon6_g2turb_uses_alpp_phip_radians_via_tab():
    """G2TURB: VTAG = (TAB*TBL)^T*[0,0,tau] with ALPP/PHIP rad — not alppx-as-deg."""
    alpp = 0.1
    phip = 0.2
    seed(ISEED)
    vehicle, env = _vehicle(alpp=alpp, phip=phip)
    env.execute(vehicle, _ctx())
    tau = vehicle.store.get("tau")
    vtag = vehicle.store.get("VTAG")
    # TGA*VTAA with TBL=I → tau * third row of TAB
    g2_body = np.array(
        [
            -tau * sin(alpp),
            tau * cos(alpp) * sin(phip),
            tau * cos(alpp) * cos(phip),
        ],
        dtype=float,
    )
    np.testing.assert_allclose(vtag, g2_body, rtol=RTOL, atol=ATOL)
    # Mis-reading planted radians as AGM6 alppx degrees (×RAD) must not match.
    agm6_misread = np.array(
        [
            -tau * sin(alpp * RAD),
            tau * sin(phip * RAD) * cos(alpp * RAD),
            tau * cos(phip * RAD) * cos(alpp * RAD),
        ],
        dtype=float,
    )
    assert not np.allclose(vtag, agm6_misread, rtol=1e-3, atol=1e-3)
