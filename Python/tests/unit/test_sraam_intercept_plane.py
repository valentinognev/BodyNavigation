"""Task 69: SRAAM5/SRAAM6 DTCT/DBTC intercept-plane miss split at MSEEK=5."""

from math import sqrt

import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr, polar_from_cart
from cadac.vehicles.flat5.sraam5.intercept import Sraam5Intercept
from cadac.vehicles.flat6.sraam6.intercept import Sraam6Intercept

RTOL = 1e-12
ATOL = 1e-12
INT_STEP = 0.5
TIME_CLOSE = 1.0
TIME_OPEN = 1.5

# Missile flies +x past a target ~30 m ahead and 3 m east. Distances stay
# under SRAAM5 Fortran G4 DBT1<50 m gate (SRAAM6 Python uses <100).
SBEL_CLOSE = np.array([0.0, 0.0, -1000.0])
SBEL_OPEN = np.array([60.0, 0.0, -1000.0])
STEL = np.array([30.0, 3.0, -1000.0])
VBEL = np.array([100.0, 0.0, 0.0])
VTEL = np.array([0.0, 0.0, 0.0])
EXX = np.array([1.5, -2.0, 0.5], dtype=float)


def _fortran_dtct_dbtc(sbtp, tpl, exx):
    """Replica of SRAAM6 MODULE.FOR G4 nav / G&C intercept-plane miss split."""
    stctl = -np.asarray(exx, dtype=float).reshape(3)[:3]
    stctp = np.asarray(tpl, dtype=float) @ stctl
    dtct = sqrt(float(stctp[0] ** 2 + stctp[1] ** 2))
    sbtcp = np.asarray(sbtp, dtype=float) - stctp
    dbtc = sqrt(float(sbtcp[0] ** 2 + sbtcp[1] ** 2))
    return dtct, dbtc


class _Vehicle:
    def __init__(self):
        self.store = StateStore()
        self.health = 1


def _ctx(combus=None, int_step=INT_STEP, vehicle_slot=0):
    if combus is None:
        combus = [
            Packet(name="m1", type="MISSILE6", status=1, vars={}),
            Packet(name="t1", type="TARGET3", status=1, vars={}),
        ]
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=vehicle_slot,
    )


def _ensure(store, name, value, kind):
    if name not in store:
        store.define(Field(name, value, kind, "data", "plant"))
    store.set(name, value)


def _plant_common(store, *, time, sbel, stel, vbel, vtel, mseek, mterm, exx):
    _ensure(store, "time", time, "real")
    _ensure(store, "halt", 0, "int")
    _ensure(store, "stop", 0, "int")
    _ensure(store, "lconv", 0, "int")
    _ensure(store, "SBEL", sbel, "vec")
    _ensure(store, "VBEL", vbel, "vec")
    _ensure(store, "mprop", 0, "int")
    _ensure(store, "trcond", 0, "int")
    _ensure(store, "mseek", mseek, "int")
    _ensure(store, "mguid", 6, "int")
    _ensure(store, "maut", 3, "int")
    _ensure(store, "EXX", exx, "vec")
    _ensure(store, "mterm", mterm, "int")
    # Target body TM: identity so intercept plane is relative-velocity framed.
    _ensure(store, "TT1L", np.eye(3), "mat")
    _ensure(store, "TTL", np.eye(3), "mat")


def _ready_sraam6(*, mseek=5, mterm=1, time=TIME_CLOSE, sbel=None, stel=None, exx=None):
    if sbel is None:
        sbel = SBEL_CLOSE
    if stel is None:
        stel = STEL
    if exx is None:
        exx = EXX
    vehicle = _Vehicle()
    intercept = Sraam6Intercept()
    intercept.define(vehicle)
    _plant_common(
        vehicle.store,
        time=time,
        sbel=sbel,
        stel=stel,
        vbel=VBEL,
        vtel=VTEL,
        mseek=mseek,
        mterm=mterm,
        exx=exx,
    )
    _ensure(vehicle.store, "STEL", stel, "vec")
    _ensure(vehicle.store, "VTEL", VTEL, "vec")
    _ensure(vehicle.store, "tgt_com_slot", 1, "int")
    return vehicle, intercept, _ctx()


def _ready_sraam5(*, mseek=5, mterm=1, time=TIME_CLOSE, sbel=None, stel=None, exx=None):
    if sbel is None:
        sbel = SBEL_CLOSE
    if stel is None:
        stel = STEL
    if exx is None:
        exx = EXX
    vehicle = _Vehicle()
    intercept = Sraam5Intercept()
    intercept.define(vehicle)
    _plant_common(
        vehicle.store,
        time=time,
        sbel=sbel,
        stel=stel,
        vbel=VBEL,
        vtel=VTEL,
        mseek=mseek,
        mterm=mterm,
        exx=exx,
    )
    _ensure(vehicle.store, "ST1EL", stel, "vec")
    _ensure(vehicle.store, "VT1EL", VTEL, "vec")
    return vehicle, intercept, _ctx(
        combus=[Packet(name="m1", type="SRAAM5", status=1, vars={})]
    )


def _two_step_cpa(vehicle, intercept, ctx):
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    vehicle.store.set("time", TIME_OPEN)
    vehicle.store.set("SBEL", SBEL_OPEN)
    intercept.execute(vehicle, ctx)


def _expected_sbtp_tpl(sbel_open, sbel_close, stel, vbel, vtel, int_step=INT_STEP):
    """Fortran-style intercept-plane SBTP and TPL at opening step (MTERM=1)."""
    sbt1l = sbel_open - stel
    vbt1l = vbel - vtel
    tt1l = np.eye(3)
    vbt1t1 = tt1l @ vbt1l
    _dv, psiyt1, thtyt1 = polar_from_cart(vbt1t1)
    psiptx = float(psiyt1) * DEG
    thtptx = float(thtyt1) * DEG - 90.0
    tpt = mat2tr(psiptx * RAD, thtptx * RAD)
    tpl = tpt @ tt1l
    sbt1p = tpl @ sbt1l
    sbbml = sbel_open - sbel_close
    sbbmp = tpl @ sbbml
    stbmp = sbbmp - sbt1p
    ww = float(stbmp[2] / sbbmp[2])
    sbtp = sbbmp * ww - stbmp
    return sbtp, tpl


@pytest.mark.parametrize(
    "ready",
    [_ready_sraam5, _ready_sraam6],
    ids=["sraam5", "sraam6"],
)
def test_define_includes_dtct_dbtc(ready):
    vehicle, _intercept, _ctx_ = ready()
    names = vehicle.store.names()
    assert "dtct" in names
    assert "dbtc" in names


@pytest.mark.parametrize(
    "ready",
    [_ready_sraam5, _ready_sraam6],
    ids=["sraam5", "sraam6"],
)
@pytest.mark.parametrize("mseek", [5, 4], ids=["mseek5", "mseek4"])
def test_cpa_writes_distinct_dtct_dbtc(ready, mseek):
    """Fortran G4 writes DTCT/DBTC on every CPA; MSEEK is not a production gate."""
    vehicle, intercept, ctx = ready(mseek=mseek, mterm=1)
    _two_step_cpa(vehicle, intercept, ctx)

    dtct = float(vehicle.store.get("dtct"))
    dbtc = float(vehicle.store.get("dbtc"))
    assert np.isfinite(dtct)
    assert np.isfinite(dbtc)
    # Distinct Fortran formulas: |STCTP_xy| vs |SBTP-STCTP|_xy with nonzero EXX.
    assert dtct != dbtc

    sbtp, tpl = _expected_sbtp_tpl(SBEL_OPEN, SBEL_CLOSE, STEL, VBEL, VTEL)
    want_dtct, want_dbtc = _fortran_dtct_dbtc(sbtp, tpl, EXX)
    np.testing.assert_allclose(dtct, want_dtct, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(dbtc, want_dbtc, rtol=RTOL, atol=ATOL)
    # Sanity: both formulas are the plane-horizontal miss magnitudes.
    assert want_dtct > 0.0
    assert want_dbtc > 0.0
