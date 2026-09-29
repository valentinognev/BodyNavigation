"""SRAAM5 MTURN=1 bank-to-turn — α+φ kinematics (C2 / D2)."""

from math import cos, sin
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import AGRAV
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat5.sraam5.control import Sraam5Control
from cadac.vehicles.flat5.sraam5.rotations import Sraam5Rotations

RTOL = 1e-12
ATOL = 1e-14

# Planted BTT kinematics (rad, rad/s) — Fortran D2 MTURN≠0
ALP = 0.25
ALPD = 0.05
PHD = 0.18
BETD_PLANTED = -0.99  # must not drive WBVB under MTURN=1

# Control plant for MAUT=44 + MTURN=1 (lateral → bank via C2ACCL + C2PHI)
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
TPHI = 0.4
PHILIM = 1.5
PH = 0.1
XI = 0.01
YI = -0.02
RATEP = 0.05
ALP0 = 0.1
BET0 = -0.05
BETD0 = -0.33


def _ctx(int_step=INT_STEP):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _c2accl_phic(alcom, allim, fspcb3):
    """Fortran CRUISE5 C2ACCL (shared CADAC BTT lateral→bank); SRAAM5 C2 stripped."""
    if alcom > allim:
        alcom = allim
    if alcom < -allim:
        alcom = -allim
    pc = (alcom * AGRAV) / (abs(fspcb3) + 0.001)
    return -np.sign(fspcb3) * pc


def test_d2_mturn1_wbvb_from_alp_phd_not_betd():
    """Fortran D2 MTURN≠0: WBVB = [PHD*cos(ALP), ALPD, PHD*sin(ALP)]."""
    vehicle = SimpleNamespace(store=StateStore())
    rot = Sraam5Rotations()
    rot.define(vehicle)
    store = vehicle.store
    store.set("mturn", 1)
    for name, value in (
        ("alp", ALP),
        ("alpd", ALPD),
        ("phd", PHD),
        ("betd", BETD_PLANTED),
    ):
        if name in store:
            store.set(name, value)
        else:
            store.define(Field(name, value, "real", "state", "control"))

    rot.execute(vehicle, SimpleNamespace(int_step=0.01))

    want = np.array(
        [PHD * cos(ALP), ALPD, PHD * sin(ALP)],
        dtype=float,
    )
    got = store.get("WBVB")
    assert got == pytest.approx(want, rel=RTOL, abs=ATOL)
    # Skid-to-turn formula must NOT match (planted BETD would dominate if used)
    stt = np.array(
        [BETD_PLANTED * sin(ALP), ALPD, -BETD_PLANTED * cos(ALP)],
        dtype=float,
    )
    assert not np.allclose(got, stt, rtol=RTOL, atol=ATOL)


def test_c2_mturn1_writes_phd_phibv_not_sideslip_rate():
    """MTURN=1: control writes PHD/PHIBV (bank); leaves BETD (sideslip rate) alone."""
    vehicle = SimpleNamespace(store=StateStore())
    ctrl = Sraam5Control()
    ctrl.define(vehicle)
    store = vehicle.store
    values = dict(
        maut=MAUT,
        mturn=1,
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
        tphi=TPHI,
        philim=PHILIM,
        ph=PH,
        phd=0.0,
        xi=XI,
        xid=0.0,
        yi=YI,
        yid=0.0,
        ratep=RATEP,
        ratepd=0.0,
        ratey=0.0,
        rateyd=0.0,
        alp=ALP0,
        alpd=0.0,
        bet=BET0,
        betd=BETD0,
    )
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

    phic = _c2accl_phic(ALCOM, ALLIM, FSPCB[2])
    if phic > PHILIM:
        phic = PHILIM
    if phic < -PHILIM:
        phic = -PHILIM
    want_phd = (phic - PH) / TPHI
    want_phibv = PH  # C2PHI returns pre-integrate PH as PHI → PHIBV

    np.testing.assert_allclose(store.get("phd"), want_phd, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("phibv"), want_phibv, rtol=RTOL, atol=ATOL)
    # No yaw-to-turn: sideslip rate untouched; beta is zero for BTT
    np.testing.assert_allclose(store.get("betd"), BETD0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("beta"), 0.0, rtol=RTOL, atol=ATOL)
    # Pitch path still live (MAUTP=4)
    assert store.get("ratepd") != pytest.approx(0.0, abs=ATOL)
    assert abs(store.get("alpha")) > 0.0
