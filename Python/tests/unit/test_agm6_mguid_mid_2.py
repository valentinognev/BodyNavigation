"""AGM6 guidance_mid_line (mguid mid digit 2)."""
from math import atan2, cos, exp, sin, sqrt
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import AGRAV, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr, mat3tr, polar_from_cart
from cadac.vehicles.flat6.agm6.guidance import Agm6Guidance

RTOL = 1e-12
ATOL = 1e-14
DT = 0.001
SMALL = 1.0e-7

GMAX = 20.0
LINE_GAIN = 1.5
NL_GAIN_FACT = 0.4
DECREMENT = 800.0
THTFLX = -30.0  # air-to-ground: fixed vertical LOA angle (deg)
PSIBLx = 10.0
THTBLx = 3.0
PHIBLx = 2.0
PSIVLCX = 8.0
THTVLCX = -2.0
GRAV = AGRAV

SBELC = np.array([100.0, -50.0, -7000.0], dtype=float)
VBELC = np.array([250.0, 10.0, 5.0], dtype=float)
STELM = np.array([33000.0, 10000.0, -100.0], dtype=float)
VTELC = np.array([0.0, -5.0, 0.0], dtype=float)
SAEL = np.array([0.0, 0.0, -7000.0], dtype=float)
SBTL = np.array([50.0, -10.0, 20.0], dtype=float)


def _tblc():
    return mat3tr(PSIBLx * RAD, THTBLx * RAD, PHIBLx * RAD)


def _ctx(sim_time=0.0, int_step=DT):
    return SimContext(
        sim_time=sim_time,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _cpp_mid_line(
    stalc,
    stblc,
    vbelc,
    tblc,
    line_gain,
    nl_gain_fact,
    decrement,
    thtflx,
    grav,
    thtvlcx,
    psivlcx,
    sbtl,
):
    """Replica of Missile::guidance_mid_line (AGM6 guidance.cpp)."""
    stalc = np.asarray(stalc, dtype=float)
    stblc = np.asarray(stblc, dtype=float)
    vbelc = np.asarray(vbelc, dtype=float)
    tblc = np.asarray(tblc, dtype=float)
    sbtl = np.asarray(sbtl, dtype=float)

    polar = polar_from_cart(stalc)
    dtac = float(polar[0])
    az_loa = float(polar[1])
    el_loa = float(polar[2])
    if thtflx == 0:
        tfl = mat2tr(az_loa, el_loa)
    else:
        tfl = mat2tr(az_loa, thtflx * RAD)

    polar = polar_from_cart(stblc)
    dtbc = float(polar[0])
    az_los = float(polar[1])
    el_los = float(polar[2])
    tol = mat2tr(az_los, el_los)

    tvl = mat2tr(psivlcx * RAD, thtvlcx * RAD)
    tbv = tblc @ tvl.T

    vbeo = tol @ vbelc
    vbef = tfl @ vbelc
    nl_gain = nl_gain_fact * (1.0 - exp(-dtbc / decrement))

    algv1 = grav * sin(thtvlcx * RAD) / AGRAV
    algv2 = line_gain * (-float(vbeo[1]) + nl_gain * float(vbef[1])) / AGRAV
    algv3 = (
        line_gain
        * ((-float(vbeo[2]) + nl_gain * float(vbef[2])) - grav * cos(thtvlcx * RAD))
        / AGRAV
    )
    acvx = np.array([algv1, algv2, algv3], dtype=float)
    acbx = tbv @ acvx
    sbto = tol @ sbtl
    return acbx, dtac, dtbc, vbeo, vbef, sbto, nl_gain


def _limit_commands(acbx, gmax):
    all_ = float(acbx[1])
    ann = -float(acbx[2])
    aa = sqrt(all_ * all_ + ann * ann)
    if aa > gmax:
        aa = gmax
    if abs(ann) < SMALL and abs(all_) < SMALL:
        phi = 0.0
    else:
        phi = atan2(ann, all_)
    return aa * cos(phi), aa * sin(phi)


def _ready(
    *,
    mguid=20,
    line_gain=LINE_GAIN,
    nl_gain_fact=NL_GAIN_FACT,
    decrement=DECREMENT,
    thtflx=THTFLX,
    gmax=GMAX,
    epchta=0.0,
):
    vehicle = SimpleNamespace(store=StateStore())
    guid = Agm6Guidance()
    guid.define(vehicle)
    store = vehicle.store
    tblc = _tblc()
    for name, value, ftype, role, module in (
        ("mnav", 0, "int", "out", "datalink"),
        ("STCEL", np.zeros(3), "vec", "out", "datalink"),
        ("VTCEL", np.zeros(3), "vec", "out", "datalink"),
        ("SAEL", SAEL, "vec", "out", "datalink"),
        ("VAEL", np.zeros(3), "vec", "out", "datalink"),
        ("SBELC", SBELC, "vec", "out", "ins"),
        ("VBELC", VBELC, "vec", "out", "ins"),
        ("TBLC", tblc, "mat", "out", "ins"),
        ("thtvlcx", THTVLCX, "real", "out", "ins"),
        ("psivlcx", PSIVLCX, "real", "out", "ins"),
        ("FSPCB", np.zeros(3), "vec", "out", "ins"),
        ("STEL", STELM, "vec", "out", "sensor"),
        ("VTEL", VTELC, "vec", "out", "sensor"),
        ("psipb", 0.0, "real", "out", "sensor"),
        ("thtpb", 0.0, "real", "out", "sensor"),
        ("ththb", 0.0, "real", "diag", "sensor"),
        ("sigdpy", 0.0, "real", "out", "sensor"),
        ("sigdpz", 0.0, "real", "out", "sensor"),
        ("SBTL", SBTL, "vec", "diag", "sensor"),
        ("SBEL", SBELC, "vec", "state", "newton"),
        ("VBEL", VBELC, "vec", "out", "newton"),
        ("gmax", gmax, "real", "diag", "aerodynamics"),
        ("grav", GRAV, "real", "out", "environment"),
    ):
        store.define(Field(name, value, ftype, role, module))
    store.set("mguid", mguid)
    store.set("line_gain", line_gain)
    store.set("nl_gain_fact", nl_gain_fact)
    store.set("decrement", decrement)
    store.set("thtflx", thtflx)
    store.set("epchta", epchta)
    store.set("STELM", STELM)
    store.set("VTELC", VTELC)
    store.set("ancomx", 123.0)
    store.set("alcomx", 456.0)
    guid.initialize(vehicle, _ctx())
    return vehicle, guid


def test_agm6_mguid_mid_2():
    tblc = _tblc()
    stalc = STELM - SAEL
    stblc = STELM - SBELC
    want_acbx, want_dtac, want_dtbc, want_vbeo, want_vbef, want_sbto, _ = _cpp_mid_line(
        stalc,
        stblc,
        VBELC,
        tblc,
        LINE_GAIN,
        NL_GAIN_FACT,
        DECREMENT,
        THTFLX,
        GRAV,
        THTVLCX,
        PSIVLCX,
        SBTL,
    )

    vehicle, guid = _ready(mguid=20)
    store = vehicle.store
    acbx = guid.guidance_mid_line(vehicle, stalc, stblc, VBELC, NL_GAIN_FACT)
    np.testing.assert_allclose(acbx, want_acbx, rtol=RTOL, atol=ATOL)
    assert acbx.shape == (3,)
    assert store.get("dtac") == pytest.approx(want_dtac, rel=RTOL, abs=ATOL)
    assert store.get("dtbc") == pytest.approx(want_dtbc, rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(store.get("VBEO"), want_vbeo, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VBEF"), want_vbef, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBTO"), want_sbto, rtol=RTOL, atol=ATOL)

    guid.execute(vehicle, _ctx())
    alcomx, ancomx = _limit_commands(want_acbx, GMAX)
    assert store.get("alcomx") == pytest.approx(alcomx, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(ancomx, rel=RTOL, abs=ATOL)
    assert abs(store.get("alcomx")) > 0.0 or abs(store.get("ancomx")) > 0.0
