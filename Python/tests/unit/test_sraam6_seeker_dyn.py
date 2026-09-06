from math import acos, atan2, cos, sin, sqrt, tan

import numpy as np
import pytest

from cadac.constants import DEG
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr
from cadac.vehicles.sraam6.seeker import Sraam6Seeker

RTOL = 1e-12
ATOL = 1e-14
DBLIND = 3.0
DT = 0.001
TRTHT = 1.0
TRTHTD = 10.0
TRPHID = 14.0
TRATE = 1.0
GK = 10.0
ZETAK = 0.9
WNK = 60.0
FOVYAW = 0.03140
FOVPITCH = 0.03140


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _missile_packet(sbel):
    return Packet(name="m1", type="MISSILE6", status=1, vars={"SBEL": np.asarray(sbel, dtype=float)})


def _target_packet(sael, vael=None):
    if vael is None:
        vael = np.zeros(3)
    return Packet(
        name="t1",
        type="TARGET3",
        status=1,
        vars={"SAEL": np.asarray(sael, dtype=float), "VAEL": np.asarray(vael, dtype=float)},
    )


def _ctx(combus, int_step=DT, sim_time=0.0):
    return SimContext(
        sim_time=sim_time,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=0,
    )


def _plant_externals(
    store,
    sbel,
    *,
    tbl=None,
    vbel=None,
    wbeb=None,
    trcond=0,
    trtht=TRTHT,
    trthtd=TRTHTD,
    trphid=TRPHID,
    trate=TRATE,
):
    if tbl is None:
        tbl = np.eye(3)
    if vbel is None:
        vbel = np.array([250.0, 0.0, 0.0])
    if wbeb is None:
        wbeb = np.zeros(3)
    store.define(Field("SBEL", sbel, "vec", "state", "newton"))
    store.define(Field("VBEL", vbel, "vec", "state", "newton"))
    store.define(Field("TBL", tbl, "mat", "out", "kinematics"))
    store.define(Field("WBEB", wbeb, "vec", "diag", "euler"))
    store.define(Field("mguid", 0, "int", "data", "guidance"))
    store.define(Field("trcond", trcond, "int", "diag", "aerodynamics"))
    store.define(Field("trtht", trtht, "real", "data", "aerodynamics"))
    store.define(Field("trthtd", trthtd, "real", "data", "aerodynamics"))
    store.define(Field("trphid", trphid, "real", "data", "aerodynamics"))
    store.define(Field("trate", trate, "real", "data", "aerodynamics"))


def _ready(
    *,
    sbtl,
    mseek=4,
    ms1dyn=1,
    dblind=DBLIND,
    trtht=TRTHT,
    wlq=0.0,
    wlr=0.0,
    int_step=DT,
    thb=None,
    tpb=None,
):
    sbtl = np.asarray(sbtl, dtype=float)
    sael = np.zeros(3)
    sbel = sael + sbtl
    vehicle = _Vehicle()
    seeker = Sraam6Seeker()
    seeker.define(vehicle)
    _plant_externals(vehicle.store, sbel, trtht=trtht)
    store = vehicle.store
    store.set("tgt_num", 1)
    store.set("mseek", mseek)
    store.set("ms1dyn", ms1dyn)
    store.set("dblind", dblind)
    store.set("gk", GK)
    store.set("zetak", ZETAK)
    store.set("wnk", WNK)
    store.set("fovyaw", FOVYAW)
    store.set("fovpitch", FOVPITCH)
    store.set("wlq", wlq)
    store.set("wlr", wlr)
    if thb is None:
        thb = np.eye(3)
    if tpb is None:
        tpb = np.eye(3)
    store.set("THB", thb)
    store.set("TPB", tpb)
    combus = [_missile_packet(sbel), _target_packet(sael)]
    return vehicle, seeker, _ctx(combus, int_step=int_step)


def _cpp_aimp(thl, ttl, dbt, daim, biasai, biassc, randsc):
    tht = thl @ ttl.T
    if dbt < daim:
        return tht @ biasai
    return tht @ (biassc + randsc)


def _cpp_uthpb(psipb, thtpb):
    ththb = acos(cos(thtpb) * cos(psipb))
    sinpsi = sin(psipb)
    tantht = tan(thtpb)
    if abs(sinpsi) and abs(tantht) < 1e-7:
        phihb = 0.0
    else:
        phihb = atan2(sinpsi, tantht)
    return ththb, phihb


def _cpp_thb(tht, phi):
    thb = np.zeros((3, 3))
    thb[0, 0] = cos(tht)
    thb[2, 0] = sin(tht)
    thb[1, 1] = cos(phi)
    thb[1, 2] = sin(phi)
    thb[0, 1] = thb[2, 0] * thb[1, 2]
    thb[0, 2] = (-thb[2, 0]) * thb[1, 1]
    thb[2, 1] = (-thb[0, 0]) * thb[1, 2]
    thb[2, 2] = thb[0, 0] * thb[1, 1]
    thb[1, 0] = 0.0
    return thb


def _cpp_seeker_dyn(store, mseek, mguid, thb, sbtl, dbt, int_step):
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
    ttl = np.array(store.get("TTL"), dtype=float, copy=True)
    tbl = np.array(store.get("TBL"), dtype=float, copy=True)
    wbeb = np.asarray(store.get("WBEB"), dtype=float)
    trcond = store.get("trcond")
    trtht = store.get("trtht")
    trthtd = store.get("trthtd")
    trphid = store.get("trphid")
    trate = store.get("trate")
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
    sath = _cpp_aimp(thl, ttl, dbt, daim, biasai, biassc, randsc)
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

    wbep = tpb @ wbeb
    wlrd_new = wlr1 - wbep[2]
    wlr = integrate(wlrd_new, wlrd, wlr, int_step)
    wlrd = wlrd_new
    psipb = wlr
    wlqd_new = wlq1 - wbep[1]
    wlq = integrate(wlqd_new, wlqd, wlq, int_step)
    wlqd = wlqd_new
    thtpb = wlq
    thtpbd = wlqd
    tpb = mat2tr(psipb, thtpb)
    ththbc, phihbc = _cpp_uthpb(psipb, thtpb)
    ththb = ththbc + biast + randt
    phihb = phihbc + biasp + randp
    thb = _cpp_thb(ththb, phihb)

    if mseek == 4:
        ibreak = 0
        phihbd = -thtpbd * sin(psipb)
        eh = sqrt(ehy * ehy + ehz * ehz)
        if abs(ththb) > trtht:
            trcond = 6
            ibreak = 1
        elif abs(thtpbd) > trthtd:
            trcond = 7
            ibreak = 1
        elif abs(phihbd) > trphid:
            trcond = 8
            ibreak = 1
        elif eh > trate:
            trcond = 9
            ibreak = 1
        if ibreak == 1:
            mseek = 2
            mguid = 3
        if dbt < dblind:
            mseek = 5
    return {
        "mseek": mseek,
        "mguid": mguid,
        "thtpb": thtpb,
        "psipb": psipb,
        "sigdy": sigdy,
        "sigdz": sigdz,
        "ehy": ehy,
        "ehz": ehz,
        "thb": thb,
        "tpb": tpb,
        "ththb": ththb,
        "phihb": phihb,
        "epy": epy,
        "epz": epz,
        "trcond": trcond,
        "wlq1": wlq1,
        "wlr1": wlr1,
        "eahh": eahh,
        "ephh": ephh,
        "eaph": eaph,
    }


def _sbtl_norm(norm, direction=(-100.0, 20.0, 10.0)):
    raw = np.asarray(direction, dtype=float)
    return raw * (norm / float(np.linalg.norm(raw)))


def test_lock_on_sbtl_100_sigdpy_sigdpz_finite():
    sbtl = _sbtl_norm(100.0)
    assert float(np.linalg.norm(sbtl)) == pytest.approx(100.0, rel=RTOL, abs=ATOL)
    vehicle, seeker, ctx = _ready(sbtl=sbtl, mseek=4, ms1dyn=1, dblind=DBLIND)
    seeker.execute(vehicle, ctx)
    store = vehicle.store
    assert ctx.int_step == DT
    assert np.isfinite(store.get("sigdpy"))
    assert np.isfinite(store.get("sigdpz"))
    np.testing.assert_allclose(store.get("SBTL"), sbtl, rtol=RTOL, atol=ATOL)
    assert store.get("mseek") == 4
    np.testing.assert_allclose(store.get("TPB"), mat2tr(store.get("psipb"), store.get("thtpb")), rtol=RTOL, atol=ATOL)
    assert store.get("sigdpy") == pytest.approx(store.get("wlq1"), rel=RTOL, abs=ATOL)
    assert store.get("sigdpz") == pytest.approx(store.get("wlr1"), rel=RTOL, abs=ATOL)


def test_sbtl_1_below_dblind_sets_mseek_5():
    sbtl = np.array([-1.0, 0.0, 0.0])
    assert float(np.linalg.norm(sbtl)) == pytest.approx(1.0, rel=RTOL, abs=ATOL)
    assert 1.0 < DBLIND
    vehicle, seeker, ctx = _ready(sbtl=sbtl, mseek=4, ms1dyn=1, dblind=DBLIND)
    seeker.execute(vehicle, ctx)
    assert vehicle.store.get("mseek") == 5


def test_ththb_beyond_trtht_break_lock():
    sbtl = np.array([-100.0, 0.0, 0.0])
    vehicle, seeker, ctx = _ready(sbtl=sbtl, mseek=4, ms1dyn=1, dblind=DBLIND, trtht=TRTHT, wlq=2.0)
    seeker.execute(vehicle, ctx)
    store = vehicle.store
    assert abs(store.get("ththb")) > TRTHT
    assert store.get("mseek") == 2
    assert store.get("trcond") == 6
    assert store.get("mguid") == 3


def test_seeker_dyn_one_step_matches_cpp_replica():
    sbtl = _sbtl_norm(100.0)
    vehicle, seeker, ctx = _ready(sbtl=sbtl, mseek=4, ms1dyn=1, dblind=DBLIND)
    want = _cpp_seeker_dyn(
        vehicle.store,
        4,
        6,
        np.array(vehicle.store.get("THB"), dtype=float, copy=True),
        sbtl,
        float(np.linalg.norm(sbtl)),
        DT,
    )
    seeker.execute(vehicle, ctx)
    store = vehicle.store
    assert store.get("mseek") == want["mseek"]
    assert store.get("mguid") == want["mguid"]
    assert store.get("trcond") == want["trcond"]
    assert store.get("thtpb") == pytest.approx(want["thtpb"], rel=RTOL, abs=ATOL)
    assert store.get("psipb") == pytest.approx(want["psipb"], rel=RTOL, abs=ATOL)
    assert store.get("sigdpy") == pytest.approx(want["sigdy"], rel=RTOL, abs=ATOL)
    assert store.get("sigdpz") == pytest.approx(want["sigdz"], rel=RTOL, abs=ATOL)
    assert store.get("ththb") == pytest.approx(want["ththb"], rel=RTOL, abs=ATOL)
    assert store.get("epy") == pytest.approx(want["epy"], rel=RTOL, abs=ATOL)
    assert store.get("epz") == pytest.approx(want["epz"], rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(store.get("THB"), want["thb"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("TPB"), want["tpb"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("EAHH"), want["eahh"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("EPHH"), want["ephh"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("EAPH"), want["eaph"], rtol=RTOL, atol=ATOL)
    assert store.get("ththbx") == pytest.approx(want["ththb"] * DEG, rel=RTOL, abs=ATOL)
    assert store.get("phihbx") == pytest.approx(want["phihb"] * DEG, rel=RTOL, abs=ATOL)


def test_seeker_aimp_hot_spot_default_zero_and_aimpoint_bias():
    seeker = Sraam6Seeker()
    vehicle = _Vehicle()
    seeker.define(vehicle)
    store = vehicle.store
    thl = np.eye(3)
    ttl = np.eye(3)
    store.set("daim", 0.0)
    sath = seeker.seeker_aimp(vehicle, thl, ttl, 100.0)
    np.testing.assert_allclose(sath, np.zeros(3), rtol=RTOL, atol=ATOL)
    store.set("daim", 200.0)
    sath = seeker.seeker_aimp(vehicle, thl, ttl, 100.0)
    np.testing.assert_allclose(sath, np.array([1.0, 0.5, 0.2]), rtol=RTOL, atol=ATOL)


def test_noise_fields_default_zero():
    vehicle = _Vehicle()
    Sraam6Seeker().define(vehicle)
    store = vehicle.store
    for name in ("biast", "randt", "biasp", "randp", "biaseh", "randeh"):
        assert store.get(name) == 0.0
    np.testing.assert_allclose(store.get("BIASSC"), np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("RANDSC"), np.zeros(3), rtol=RTOL, atol=ATOL)
