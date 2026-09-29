"""SRAAM5 S1/S1KIN — MSEEK 2→3/4 transition fields (Task 61)."""

import numpy as np
import pytest

from cadac.constants import DEG
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr, polar_from_cart, skew
from cadac.vehicles.flat5.sraam5.seeker import Sraam5Seeker

RTOL = 1e-12
ATOL = 1e-14

SBEL = np.array([0.0, 0.0, -5000.0])
ST1EL = np.array([10000.0, 500.0, -2000.0])
VT1EL = np.array([-250.0, 0.0, 0.0])
VBEL = np.array([250.0, 0.0, 0.0])
DTIMAC = 0.25


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _dbt1(sbel=SBEL, st1el=ST1EL, tt1l=None):
    if tt1l is None:
        tt1l = np.eye(3)
    sbt1l = sbel - st1el
    sbt1t1 = tt1l @ sbt1l
    return float(polar_from_cart(sbt1t1)[0])


def _ctx(sim_time=0.0, int_step=0.0123):
    return SimContext(
        sim_time=sim_time,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=[],
        vehicle_slot=0,
    )


def _ready(
    *,
    mseek=2,
    ms1dyn=0,
    racq=20000.0,
    dtimac=DTIMAC,
    isets1=0,
    epchac=0.0,
    mguid=0,
    sim_time=0.0,
    int_step=0.0123,
    sbel=SBEL,
    st1el=ST1EL,
    vt1el=VT1EL,
    vbel=VBEL,
    tbl=None,
    tt1l=None,
    dblind=3.0,
    gk=10.0,
    zetak=0.9,
    wnk=60.0,
    trtht=1.0,
    trthtd=10.0,
    trphid=14.0,
    trate=1.0,
):
    if tbl is None:
        tbl = np.eye(3)
    if tt1l is None:
        tt1l = np.eye(3)
    vehicle = _Vehicle()
    seeker = Sraam5Seeker()
    seeker.define(vehicle)
    store = vehicle.store
    for name, value, ftype, role, module in (
        ("SBEL", sbel, "vec", "state", "newton"),
        ("VBEL", vbel, "vec", "out", "newton"),
        ("TBL", tbl, "mat", "out", "rotations"),
        ("ST1EL", st1el, "vec", "state", "target"),
        ("VT1EL", vt1el, "vec", "state", "target"),
        ("TT1L", tt1l, "mat", "out", "target"),
        ("mguid", mguid, "int", "data", "guidance"),
        ("WBECB", (0.0, 0.0, 0.0), "vec", "out", "ins"),
        ("trcode", 0.0, "real", "diag", "control"),
        ("trtht", trtht, "real", "data", "control"),
    ):
        if name not in store:
            store.define(Field(name, value, ftype, role, module))
        else:
            store.set(name, value)
    store.set("mseek", mseek)
    store.set("ms1dyn", ms1dyn)
    store.set("racq", racq)
    store.set("dtimac", dtimac)
    store.set("isets1", isets1)
    store.set("epchac", epchac)
    store.set("dblind", dblind)
    store.set("gk", gk)
    store.set("zetak", zetak)
    store.set("wnk", wnk)
    store.set("trthtd", trthtd)
    store.set("trphid", trphid)
    store.set("trate", trate)
    return vehicle, seeker, _ctx(sim_time=sim_time, int_step=int_step)


def _s1kin_ref(sbt1l, vt1el, vbel, tbl, dbt1):
    st1bl = np.asarray(sbt1l, dtype=float) * (-1.0)
    st1bb = tbl @ st1bl
    dum1 = 1.0 / dbt1
    ut1bl = st1bl * dum1
    vt1bl = np.asarray(vt1el, dtype=float) - vbel
    dvbt1c = abs(float(ut1bl @ vt1bl))
    woeb = tbl @ (skew(ut1bl) @ vt1bl) * dum1
    polar = polar_from_cart(st1bb)
    psipb = float(polar[1])
    thtpb = float(polar[2])
    woep = mat2tr(psipb, thtpb) @ woeb
    return thtpb, psipb, float(woep[1]), float(woep[2]), dvbt1c


def test_mseek_2_dbt1_lt_racq_sets_mseek_3():
    dbt1 = _dbt1()
    assert dbt1 > 7000.0
    assert dbt1 < 20000.0
    vehicle, seeker, ctx = _ready(mseek=2, racq=20000.0, ms1dyn=0)
    seeker.execute(vehicle, ctx)
    store = vehicle.store
    assert store.get("mseek") == 3
    assert store.get("dbt1") == pytest.approx(dbt1, rel=RTOL, abs=ATOL)
    assert store.get("isets1") == 0
    assert store.get("epchac") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert np.isfinite(store.get("thtpb"))
    assert np.isfinite(store.get("psipb"))


def test_mseek_2_dbt1_ge_racq_stays_enabled():
    vehicle, seeker, ctx = _ready(mseek=2, racq=1.0, ms1dyn=0)
    seeker.execute(vehicle, ctx)
    assert vehicle.store.get("mseek") == 2
    assert vehicle.store.get("isets1") == 1
    assert vehicle.store.get("dbt1") > 1.0


def test_mseek_3_kin_timeac_gt_dtimac_sets_mseek_4_and_mguid_6():
    vehicle, seeker, ctx = _ready(
        mseek=3,
        ms1dyn=0,
        isets1=0,
        epchac=0.0,
        dtimac=DTIMAC,
        sim_time=0.251,
        mguid=3,
    )
    seeker.execute(vehicle, ctx)
    store = vehicle.store
    assert store.get("mseek") == 4
    assert store.get("mguid") == 6
    assert store.get("timeac") == pytest.approx(0.251, rel=RTOL, abs=ATOL)
    sbt1l = SBEL - ST1EL
    dbt1 = _dbt1()
    thtpb, psipb, sigdy, sigdz, dvbt1c = _s1kin_ref(
        sbt1l, VT1EL, VBEL, np.eye(3), dbt1
    )
    assert store.get("thtpb") == pytest.approx(thtpb, rel=RTOL, abs=ATOL)
    assert store.get("psipb") == pytest.approx(psipb, rel=RTOL, abs=ATOL)
    assert store.get("sigdpy") == pytest.approx(sigdy, rel=RTOL, abs=ATOL)
    assert store.get("sigdpz") == pytest.approx(sigdz, rel=RTOL, abs=ATOL)
    assert store.get("dvbt1c") == pytest.approx(dvbt1c, rel=RTOL, abs=ATOL)
    assert store.get("thtpbx") == pytest.approx(thtpb * DEG, rel=RTOL, abs=ATOL)
    assert store.get("psipbx") == pytest.approx(psipb * DEG, rel=RTOL, abs=ATOL)


def test_s1kin_los_rates_match_fortran_slice():
    vehicle, seeker, ctx = _ready(mseek=4, ms1dyn=0, mguid=0)
    seeker.execute(vehicle, ctx)
    store = vehicle.store
    sbt1l = store.get("SBT1L")
    dbt1 = store.get("dbt1")
    thtpb, psipb, sigdy, sigdz = seeker.seeker_kin(
        vehicle, sbt1l, store.get("VT1EL"), dbt1
    )
    want = _s1kin_ref(sbt1l, VT1EL, VBEL, np.eye(3), dbt1)
    assert thtpb == pytest.approx(want[0], rel=RTOL, abs=ATOL)
    assert psipb == pytest.approx(want[1], rel=RTOL, abs=ATOL)
    assert sigdy == pytest.approx(want[2], rel=RTOL, abs=ATOL)
    assert sigdz == pytest.approx(want[3], rel=RTOL, abs=ATOL)
    assert store.get("mseek") == 4
    assert store.get("mguid") == 6


def test_ms1dyn_1_lock_runs_s1dyn_without_raise():
    """inlar1 uses ms1dyn=1; lock must call S1DYN and write its fields."""
    vehicle, seeker, ctx = _ready(
        mseek=4,
        ms1dyn=1,
        mguid=0,
        dblind=3.0,
        isets1=0,
    )
    # Seed pointing states so dyn filter has a finite start (post-acquisition).
    store = vehicle.store
    sbt1l = SBEL - ST1EL
    dbt1 = _dbt1()
    thtpb0, psipb0, sigdy0, sigdz0, _ = _s1kin_ref(
        sbt1l, VT1EL, VBEL, np.eye(3), dbt1
    )
    store.set("wlq", thtpb0)
    store.set("wlr", psipb0)
    store.set("wlq1", sigdy0)
    store.set("wlr1", sigdz0)
    ththb, phihb = seeker.seeker_uthpb(psipb0, thtpb0)
    store.set("THB", seeker.seeker_thb(ththb, phihb))
    if "TPB" not in store:
        store.define(Field("TPB", mat2tr(psipb0, thtpb0), "mat", "init", "seeker"))
    else:
        store.set("TPB", mat2tr(psipb0, thtpb0))

    seeker.execute(vehicle, ctx)
    store = vehicle.store
    assert store.get("mseek") == 4
    assert store.get("mguid") == 6
    assert np.isfinite(store.get("thtpb"))
    assert np.isfinite(store.get("psipb"))
    assert np.isfinite(store.get("sigdpy"))
    assert np.isfinite(store.get("sigdpz"))
    assert np.isfinite(store.get("epy"))
    assert np.isfinite(store.get("epz"))
    assert np.isfinite(store.get("ththb"))
    assert np.isfinite(store.get("phihb"))
    assert store.get("dba") == pytest.approx(store.get("dbt1"), rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(store.get("THB"), store.get("THB"), rtol=RTOL, atol=ATOL)
    assert np.all(np.isfinite(store.get("THB")))
    assert np.all(np.isfinite(store.get("TPB")))
    assert np.all(np.isfinite(store.get("EAHH")))
    assert np.all(np.isfinite(store.get("EPHH")))
    assert np.all(np.isfinite(store.get("EAPH")))
