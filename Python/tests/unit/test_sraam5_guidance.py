"""SRAAM5 Fortran C1 / C1MID / C1TERM — MGUID mid/term acceleration commands."""

from math import atan2, cos, fabs, sin, sqrt
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import AGRAV, DEG
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import cart_from_pol, polar_from_cart, skew
from cadac.vehicles.flat5.sraam5.guidance import Sraam5Guidance

RTOL = 1e-12
ATOL = 1e-14
DT = 0.01
GNAV = 3.75
GMAX = 40.0
TRCVEL = 91.44  # 300 ft/s * OPTMET path ≈ m/s scale for plant
IDENTITY = np.eye(3, dtype=float)

# Midcourse plant (INS + extrapolated target)
SBELC = np.array([100.0, -50.0, -2000.0], dtype=float)
VBELC = np.array([400.0, 10.0, -5.0], dtype=float)
ST1CEL = np.array([5000.0, 200.0, -1800.0], dtype=float)
VT1CEL = np.array([250.0, -20.0, 0.0], dtype=float)
TBLC = np.array(
    [
        [0.98, 0.05, -0.19],
        [-0.04, 0.997, 0.06],
        [0.192, -0.052, 0.98],
    ],
    dtype=float,
)

# Terminal plant (truth + seeker)
SBEL = np.array([120.0, -40.0, -1900.0], dtype=float)
VBEL = np.array([380.0, 8.0, -3.0], dtype=float)
ST1EL = np.array([800.0, 30.0, -1850.0], dtype=float)
VT1EL = np.array([240.0, -15.0, 1.0], dtype=float)
PSIPB = 0.04
THTPB = -0.025
SIGDPY = 0.015
SIGDPZ = -0.012
FSPCB = np.array([25.0, 0.5, -9.5], dtype=float)


def _ctx(sim_time=1.0, int_step=DT):
    return SimContext(
        sim_time=sim_time,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _fortran_c1_mid(st1blc, vt1elc, vbelc, tblc, gnav):
    """Replica of MODULE.FOR C1 midcourse + C1MID."""
    st1blc = np.asarray(st1blc, dtype=float)
    vt1elc = np.asarray(vt1elc, dtype=float)
    vbelc = np.asarray(vbelc, dtype=float)
    tblc = np.asarray(tblc, dtype=float)
    dt1bc = float(np.linalg.norm(st1blc))
    ut1blc = st1blc * (1.0 / dt1bc)
    ut1bbc = tblc @ ut1blc
    polar = polar_from_cart(ut1bbc)
    psiobc = float(polar[1])
    thtobc = float(polar[2])
    psiobcx = psiobc * DEG
    thtobcx = thtobc * DEG
    vt1blc = vt1elc - vbelc
    dvt1bc = fabs(float(ut1blc @ vt1blc))
    tgoc = dt1bc / dvt1bc
    woelc = skew(ut1blc) @ vt1blc * (1.0 / dt1bc)
    uobb = cart_from_pol(1.0, psiobc, thtobc)
    uobl = tblc.T @ uobb
    apnl = skew(woelc) @ uobl * (gnav * dvt1bc)
    aapnb = tblc @ apnl
    ancomx = -float(aapnb[2]) / AGRAV
    alcomx = float(aapnb[1]) / AGRAV
    return (
        ancomx,
        alcomx,
        woelc,
        ut1blc,
        tgoc,
        dt1bc,
        dvt1bc,
        psiobcx,
        thtobcx,
    )


def _fortran_c1_term(
    sbel, st1el, vbel, vt1el, psipb, thtpb, sigdpy, sigdpz, fspcb, gnav, gmax
):
    """Replica of MODULE.FOR C1TERM (circular limiter included)."""
    sbel = np.asarray(sbel, dtype=float)
    st1el = np.asarray(st1el, dtype=float)
    vbel = np.asarray(vbel, dtype=float)
    vt1el = np.asarray(vt1el, dtype=float)
    fspcb = np.asarray(fspcb, dtype=float)
    sbt1l = sbel - st1el
    dbt1 = float(np.linalg.norm(sbt1l))
    vbt1l = vbel - vt1el
    dum = float(sbt1l @ vbt1l)
    dcvel = fabs(dum / dbt1)
    adely = sin(psipb) * float(fspcb[0])
    adelz = sin(thtpb) * cos(psipb) * float(fspcb[0])
    gn = gnav * dcvel
    apny = gn * sigdpz
    apnz = gn * sigdpy
    cththb = fabs(cos(thtpb) * cos(psipb))
    all_ = (apny + adely) / (cththb * AGRAV)
    ann = (apnz + adelz) / (cththb * AGRAV)
    aa = sqrt(all_ * all_ + ann * ann)
    if aa > gmax:
        aa = gmax
    if max(fabs(ann), fabs(all_)) < 1.0e-10:
        phi = 0.0
    else:
        phi = atan2(ann, all_)
    alcomx = aa * cos(phi)
    ancomx = aa * sin(phi)
    return alcomx, ancomx, gn, apny, apnz, adely, adelz, all_, ann, dcvel, dbt1


def _ready(*, mguid, mnav=0, sim_time=1.0, epchta=0.0):
    vehicle = SimpleNamespace(store=StateStore())
    guid = Sraam5Guidance()
    guid.define(vehicle)
    store = vehicle.store
    for name, value, ftype, role, module in (
        ("mnav", mnav, "int", "out", "ai_radar"),
        ("ST1CEL", ST1CEL, "vec", "out", "ai_radar"),
        ("VT1CEL", VT1CEL, "vec", "out", "ai_radar"),
        ("SBELC", SBELC, "vec", "out", "ins"),
        ("VBELC", VBELC, "vec", "out", "ins"),
        ("TBLC", TBLC, "mat", "out", "ins"),
        ("FSPCB", FSPCB, "vec", "out", "ins"),
        ("ST1EL", ST1EL, "vec", "out", "target"),
        ("VT1EL", VT1EL, "vec", "out", "target"),
        ("SBEL", SBEL, "vec", "out", "newton"),
        ("VBEL", VBEL, "vec", "out", "newton"),
        ("thtpb", THTPB, "real", "out", "seeker"),
        ("psipb", PSIPB, "real", "out", "seeker"),
        ("sigdpy", SIGDPY, "real", "out", "seeker"),
        ("sigdpz", SIGDPZ, "real", "out", "seeker"),
        ("gmax", GMAX, "real", "out", "aerodynamics"),
        ("trcode", 0.0, "real", "out", "intercept"),
        ("trcvel", TRCVEL, "real", "data", "intercept"),
    ):
        store.define(Field(name, value, ftype, role, module))
        store.set(name, value)
    store.set("mguid", mguid)
    store.set("gnav", GNAV)
    store.set("epchta", epchta)
    # Pre-seed saved extrapolation state as if a prior MNAV=3 update occurred at t=epchta
    if mnav != 3:
        store.set("ST1ELM", ST1CEL.copy())
        store.set("VT1ELC", VT1CEL.copy())
    return vehicle, guid, _ctx(sim_time=sim_time)


def test_mguid_3_midcourse_sets_ancomx_alcomx():
    """MGUID=3: ANCOM/ALCOM from C1 mid + C1MID match Fortran equations."""
    vehicle, guid, ctx = _ready(mguid=3, mnav=0, sim_time=1.0, epchta=0.5)
    dtimex = ctx.sim_time - 0.5
    st1elc = ST1CEL + VT1CEL * dtimex
    st1blc = st1elc - SBELC
    want = _fortran_c1_mid(st1blc, VT1CEL, VBELC, TBLC, GNAV)

    guid.execute(vehicle, ctx)
    store = vehicle.store

    assert store.get("ancomx") == pytest.approx(want[0], rel=RTOL, abs=ATOL)
    assert store.get("alcomx") == pytest.approx(want[1], rel=RTOL, abs=ATOL)
    assert np.allclose(store.get("WOELC"), want[2], rtol=RTOL, atol=ATOL)
    assert np.allclose(store.get("UT1BLC"), want[3], rtol=RTOL, atol=ATOL)
    assert store.get("tgoc") == pytest.approx(want[4], rel=RTOL, abs=ATOL)
    assert store.get("dt1bc") == pytest.approx(want[5], rel=RTOL, abs=ATOL)
    assert store.get("dvt1bc") == pytest.approx(want[6], rel=RTOL, abs=ATOL)
    assert store.get("psiobcx") == pytest.approx(want[7], rel=RTOL, abs=ATOL)
    assert store.get("thtobcx") == pytest.approx(want[8], rel=RTOL, abs=ATOL)
    assert np.allclose(store.get("ST1ELC"), st1elc, rtol=RTOL, atol=ATOL)
    assert np.allclose(store.get("ST1BLC"), st1blc, rtol=RTOL, atol=ATOL)


def test_mguid_6_terminal_sets_ancomx_alcomx():
    """MGUID=6: C1TERM circular-limited ALCOM/ANCOM match Fortran."""
    vehicle, guid, ctx = _ready(mguid=6, mnav=0, sim_time=2.0, epchta=0.0)
    want = _fortran_c1_term(
        SBEL, ST1EL, VBEL, VT1EL, PSIPB, THTPB, SIGDPY, SIGDPZ, FSPCB, GNAV, GMAX
    )

    guid.execute(vehicle, ctx)
    store = vehicle.store

    assert store.get("alcomx") == pytest.approx(want[0], rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(want[1], rel=RTOL, abs=ATOL)
    assert store.get("gn") == pytest.approx(want[2], rel=RTOL, abs=ATOL)
    assert store.get("apny") == pytest.approx(want[3], rel=RTOL, abs=ATOL)
    assert store.get("apnz") == pytest.approx(want[4], rel=RTOL, abs=ATOL)
    assert store.get("adely") == pytest.approx(want[5], rel=RTOL, abs=ATOL)
    assert store.get("adelz") == pytest.approx(want[6], rel=RTOL, abs=ATOL)
    assert store.get("all") == pytest.approx(want[7], rel=RTOL, abs=ATOL)
    assert store.get("ann") == pytest.approx(want[8], rel=RTOL, abs=ATOL)
    # Closing range ~750 m < 1000 and dcvel > trcvel → trcode stays 0
    assert store.get("trcode") == pytest.approx(0.0, abs=ATOL)


def test_mguid_6_sets_trcode_when_closing_too_slow():
    """C1TERM: DBT1<=1000 and DCVEL<TRCVEL → TRCODE=1."""
    vehicle, guid, ctx = _ready(mguid=6, mnav=0, sim_time=2.0, epchta=0.0)
    store = vehicle.store
    # Place target almost co-located with low closing speed
    close_stel = SBEL + np.array([50.0, 0.0, 0.0], dtype=float)
    slow_vtel = VBEL.copy()  # relative velocity ~0 along LOS
    store.set("ST1EL", close_stel)
    store.set("VT1EL", slow_vtel)
    store.set("trcvel", 100.0)

    guid.execute(vehicle, ctx)

    sbt1l = SBEL - close_stel
    dbt1 = float(np.linalg.norm(sbt1l))
    assert dbt1 <= 1000.0
    dum = float(sbt1l @ (VBEL - slow_vtel))
    dcvel = fabs(dum / dbt1)
    assert dcvel < 100.0
    assert store.get("trcode") == pytest.approx(1.0, abs=ATOL)


def test_mnav_3_latches_epoch_and_resets_mnav():
    """MNAV=3 copies ST1CEL/VT1CEL, sets EPCHTA=T, resets MNAV=0."""
    vehicle, guid, ctx = _ready(mguid=0, mnav=3, sim_time=4.5, epchta=0.0)
    guid.execute(vehicle, ctx)
    store = vehicle.store
    assert store.get("mnav") == 0
    assert store.get("epchta") == pytest.approx(4.5, abs=ATOL)
    assert np.allclose(store.get("ST1ELM"), ST1CEL, rtol=RTOL, atol=ATOL)
    assert np.allclose(store.get("VT1ELC"), VT1CEL, rtol=RTOL, atol=ATOL)
