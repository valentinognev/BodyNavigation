from math import atan2, fabs, sin, sqrt
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import DEG
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr

from tests.unit.test_agm6_sensor_kin import (
    ATOL,
    RTOL,
    SAEL,
    SBEL,
    TBL,
    _ctx,
    _ready,
    _sensor_ir_thb,
    _sensor_ir_uthpb,
)

DT = 0.001
WNK = 100.0
ZETAK = 0.9
GK = 10.0
DBLIND_HOLD = 6000.0
DBLIND_FAR = 10.0
TRTHT = 1.2
TRTHTD = 20.0
TRPHID = 20.0
TRATE = 2.0
WBECB = np.array([0.1, 0.05, -0.03], dtype=float)
FOVYAW = 0.035
FOVPITCH = 0.035


def _plant_dyn(store, *, wbecb=WBECB, mguid=0, trcond=0, dblind=DBLIND_FAR):
    store.define(Field("WBECB", wbecb, "vec", "out", "ins"))
    store.define(Field("trcond", trcond, "int", "diag", "aerodynamics"))
    store.define(Field("mguid", mguid, "int", "data", "guidance"))
    store.define(Field("trtht", TRTHT, "real", "data", "aerodynamics"))
    store.define(Field("trthtd", TRTHTD, "real", "data", "aerodynamics"))
    store.define(Field("trphid", TRPHID, "real", "data", "aerodynamics"))
    store.define(Field("trate", TRATE, "real", "data", "aerodynamics"))
    store.set("dblind", dblind)
    store.set("gk", GK)
    store.set("zetak", ZETAK)
    store.set("wnk", WNK)
    store.set("fovyaw", FOVYAW)
    store.set("fovpitch", FOVPITCH)
    store.set("THB", np.eye(3))
    store.set("TPB", np.eye(3))


def _sensor_ir_aimp(thl, ttl, dbtk, daim, biasai, biassc, randsc):
    tht = thl @ ttl.T
    if dbtk < daim:
        return tht @ biasai
    return tht @ (biassc + randsc)


def _sensor_ir_dyn(store, sbtl, dbtk, int_step, mseek, mguid, thb, trcond):
    ibreak = store.get("ibreak")
    dblind = store.get("dblind")
    gk = store.get("gk")
    zetak = store.get("zetak")
    wnk = store.get("wnk")
    biast = store.get("biast")
    randt = store.get("randt")
    biasp = store.get("biasp")
    randp = store.get("randp")
    biaseh = store.get("biaseh")
    randeh = store.get("randeh")
    tpb = np.array(store.get("TPB"), dtype=float, copy=True)
    trtht = store.get("trtht")
    trthtd = store.get("trthtd")
    trphid = store.get("trphid")
    trate = store.get("trate")
    tbl = np.asarray(store.get("TBL"), dtype=float)
    ttl = np.eye(3)
    wbecb = np.asarray(store.get("WBECB"), dtype=float)
    wlq1d = store.get("wlq1d")
    wlq1 = store.get("wlq1")
    wlqd = store.get("wlqd")
    wlq = store.get("wlq")
    wlr1d = store.get("wlr1d")
    wlr1 = store.get("wlr1")
    wlrd = store.get("wlrd")
    wlr = store.get("wlr")
    wlq2d = store.get("wlq2d")
    wlq2 = store.get("wlq2")
    wlr2d = store.get("wlr2d")
    wlr2 = store.get("wlr2")
    daim = store.get("daim")
    biasai = np.asarray(store.get("BIASAI"), dtype=float)
    biassc = np.asarray(store.get("BIASSC"), dtype=float)
    randsc = np.asarray(store.get("RANDSC"), dtype=float)

    thl = thb @ tbl
    sbth = thl @ sbtl
    sath = _sensor_ir_aimp(thl, ttl, dbtk, daim, biasai, biassc, randsc)
    sabh = sath - sbth
    ey = atan2(-sabh[2], sabh[0])
    ez = atan2(sabh[1], sabh[0])
    ehy = ey + biaseh + randeh
    ehz = ez + biaseh + randeh
    eahh = np.array([0.0, ehz, -ehy])
    tbh = thb.T
    tph = tpb @ tbh
    thp = tph.T
    u1pp = np.array([1.0, 0.0, 0.0])
    u1hh = np.array([1.0, 0.0, 0.0])
    ephh = thp @ u1pp - u1hh
    eaph = eahh - ephh
    eapp = tph @ eaph
    epy = -eapp[2]
    epz = eapp[1]

    wsq = wnk * wnk
    gg = gk * wsq
    wlr1d_new = wlr2
    wlr1 = integrate(wlr1d_new, wlr1d, wlr1, int_step)
    wlr1d = wlr1d_new
    wlr2d_new = gg * epz - 2.0 * zetak * wnk * wlr1d - wsq * wlr1
    wlr2 = integrate(wlr2d_new, wlr2d, wlr2, int_step)
    wlr2d = wlr2d_new
    wlq1d_new = wlq2
    wlq1 = integrate(wlq1d_new, wlq1d, wlq1, int_step)
    wlq1d = wlq1d_new
    wlq2d_new = gg * epy - 2.0 * zetak * wnk * wlq1d - wsq * wlq1
    wlq2 = integrate(wlq2d_new, wlq2d, wlq2, int_step)
    wlq2d = wlq2d_new
    sigdz = wlr1
    sigdy = wlq1

    wbep = tpb @ wbecb
    wlrd_new = wlr1 - wbep[2]
    wlr = integrate(wlrd_new, wlrd, wlr, int_step)
    wlrd = wlrd_new
    psipb = wlr
    psipbd = wlrd
    wlqd_new = wlq1 - wbep[1]
    wlq = integrate(wlqd_new, wlqd, wlq, int_step)
    wlqd = wlqd_new
    thtpb = wlq
    thtpbd = wlqd
    tpb = mat2tr(psipb, thtpb)
    ththbc, phihbc = _sensor_ir_uthpb(psipb, thtpb)
    ththb = ththbc + biast + randt
    phihb = phihbc + biasp + randp
    thb = _sensor_ir_thb(ththb, phihb)

    if mseek == 4:
        ibreak = 0
        phihbd = -thtpbd * sin(psipb)
        eh = sqrt(ehy * ehy + ehz * ehz)
        if fabs(ththb) > trtht:
            trcond = 6
            ibreak = 1
        elif fabs(thtpbd) > trthtd:
            trcond = 7
            ibreak = 1
        elif fabs(phihbd) > trphid:
            trcond = 8
            ibreak = 1
        elif eh > trate:
            trcond = 9
            ibreak = 1
        if ibreak == 1:
            mseek = 2
            mguid = 40
        if dbtk < dblind:
            mseek = 5

    return SimpleNamespace(
        mseek=mseek,
        mguid=mguid,
        thtpb=thtpb,
        psipb=psipb,
        sigdy=sigdy,
        sigdz=sigdz,
        ehz=ehz,
        ehy=ehy,
        thb=thb,
        trcond=trcond,
        tpb=tpb,
        epy=epy,
        epz=epz,
        ththb=ththb,
        phihb=phihb,
        eahh=eahh,
        ephh=ephh,
        eaph=eaph,
        wlq1d=wlq1d,
        wlq1=wlq1,
        wlqd=wlqd,
        wlq=wlq,
        wlr1d=wlr1d,
        wlr1=wlr1,
        wlrd=wlrd,
        wlr=wlr,
        wlq2d=wlq2d,
        wlq2=wlq2,
        wlr2d=wlr2d,
        wlr2=wlr2,
    )


def _assert_dyn_match(store, want):
    assert store.get("mseek") == want.mseek
    assert store.get("mguid") == want.mguid
    assert store.get("trcond") == want.trcond
    np.testing.assert_allclose(store.get("thtpb"), want.thtpb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("psipb"), want.psipb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("sigdpy"), want.sigdy, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("sigdpz"), want.sigdz, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("epy"), want.epy, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("epz"), want.epz, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ththb"), want.ththb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("phihb"), want.phihb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("THB"), want.thb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("TPB"), want.tpb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("EAHH"), want.eahh, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("EPHH"), want.ephh, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("EAPH"), want.eaph, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("wlq1"), want.wlq1, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("wlr1"), want.wlr1, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("wlq2"), want.wlq2, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("wlr2"), want.wlr2, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("thtpbx"), want.thtpb * DEG, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("psipbx"), want.psipb * DEG, rtol=RTOL, atol=ATOL)


def test_lock_on_dyn_sigdpy_finite_vs_replica():
    vehicle, sensor, combus = _ready(mseek=4, skr_dyn=1)
    store = vehicle.store
    _plant_dyn(store)
    sbtl = SBEL - SAEL
    dbtk = float(np.linalg.norm(sbtl))
    assert dbtk == pytest.approx(5000.0, rel=RTOL, abs=ATOL)
    want = _sensor_ir_dyn(
        store, sbtl, dbtk, DT, 4, 0, np.eye(3), 0
    )
    assert np.isfinite(want.sigdy)
    assert np.isfinite(want.sigdz)

    sensor.execute(vehicle, _ctx(combus, int_step=DT))
    np.testing.assert_allclose(store.get("SBTL"), sbtl, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("dbtk"), dbtk, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("TBL"), TBL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("WBECB"), WBECB, rtol=RTOL, atol=ATOL)
    assert np.isfinite(store.get("sigdpy"))
    assert np.isfinite(store.get("sigdpz"))
    _assert_dyn_match(store, want)


def test_blind_range_sets_mseek_5():
    vehicle, sensor, combus = _ready(mseek=4, skr_dyn=1)
    store = vehicle.store
    _plant_dyn(store, dblind=DBLIND_HOLD)
    sbtl = SBEL - SAEL
    dbtk = float(np.linalg.norm(sbtl))
    assert dbtk < DBLIND_HOLD
    want = _sensor_ir_dyn(
        store, sbtl, dbtk, DT, 4, 0, np.eye(3), 0
    )
    assert want.mseek == 5
    sensor.execute(vehicle, _ctx(combus, int_step=DT))
    assert store.get("mseek") == 5
    _assert_dyn_match(store, want)


def test_break_lock_sets_mseek_2_and_mguid_40():
    vehicle, sensor, combus = _ready(mseek=4, skr_dyn=1)
    store = vehicle.store
    _plant_dyn(store, mguid=6, dblind=DBLIND_FAR)
    store.set("trate", 0.01)
    sbtl = SBEL - SAEL
    dbtk = float(np.linalg.norm(sbtl))
    want = _sensor_ir_dyn(
        store, sbtl, dbtk, DT, 4, 6, np.eye(3), 0
    )
    assert want.mseek == 2
    assert want.mguid == 40
    assert want.trcond == 9
    sensor.execute(vehicle, _ctx(combus, int_step=DT))
    assert store.get("mseek") == 2
    assert store.get("mguid") == 40
    assert store.get("trcond") == 9
    _assert_dyn_match(store, want)


def test_acquire_dyn_locks_when_fov_and_dtimac_elapsed():
    vehicle, sensor, combus = _ready(mseek=3, skr_dyn=1)
    store = vehicle.store
    _plant_dyn(store)
    store.set("isets1", 0)
    store.set("epchac", 0.0)
    store.set("time", 0.31)
    store.set("fovyaw", 2.0)
    store.set("fovpitch", 2.0)
    sensor.execute(vehicle, _ctx(combus, sim_time=0.31, int_step=DT))
    assert store.get("mseek") == 4
    np.testing.assert_allclose(store.get("timeac"), 0.31, rtol=RTOL, atol=ATOL)


def test_acquire_dyn_sets_trcond_5_when_outside_fov():
    vehicle, sensor, combus = _ready(mseek=3, skr_dyn=1)
    store = vehicle.store
    _plant_dyn(store)
    store.set("isets1", 0)
    store.set("epchac", 0.0)
    store.set("time", 0.31)
    store.set("fovyaw", 1e-9)
    store.set("fovpitch", 1e-9)
    sensor.execute(vehicle, _ctx(combus, sim_time=0.31, int_step=DT))
    assert store.get("mseek") == 3
    assert store.get("trcond") == 5


def test_skr_dyn_2_still_raises():
    vehicle, sensor, combus = _ready(mseek=4, skr_dyn=2)
    _plant_dyn(vehicle.store)
    with pytest.raises(ValueError):
        sensor.execute(vehicle, _ctx(combus, int_step=DT))
