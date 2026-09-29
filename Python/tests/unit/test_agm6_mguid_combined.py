"""AGM6 combined mid|term mguid codes (26, 36, 46).

C++ guidance.cpp runs mid and term as independent if-branches; term overwrites
ACBX last, but mid side-effects still land in the store.
"""
from math import atan2, cos, exp, sin, sqrt, tan
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import AGRAV, DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr, mat3tr, polar_from_cart
from cadac.vehicles.flat6.agm6.guidance import Agm6Guidance

RTOL = 1e-12
ATOL = 1e-14
DT = 0.001
SMALL = 1.0e-7

GNAV = 3.0
GRAV_BIAS = 1.5
GMAX = 20.0
LINE_GAIN = 1.5
NL_GAIN_FACT = 0.4
DECREMENT = 800.0
THTFLX = -30.0
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
STEL = np.array([34000.0, 11000.0, -200.0], dtype=float)
VTEL = np.array([0.0, -5.0, 0.0], dtype=float)
SBEL = np.array([120.0, -40.0, -6980.0], dtype=float)
VBEL = np.array([248.0, 12.0, 4.0], dtype=float)
FSPCB = np.array([12.0, 0.5, -9.8], dtype=float)
PSIPB = 0.05
THTPB = -0.03
THTHB = 0.21
SIGDPY = 0.02
SIGDPZ = -0.01


def _skew(vec):
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


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
    stalc = np.asarray(stalc, dtype=float)
    stblc = np.asarray(stblc, dtype=float)
    vbelc = np.asarray(vbelc, dtype=float)
    tblc = np.asarray(tblc, dtype=float)
    sbtl = np.asarray(sbtl, dtype=float)

    polar = polar_from_cart(stalc)
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
    acbx = tbv @ np.array([algv1, algv2, algv3], dtype=float)
    return acbx, dtbc, vbeo


def _cpp_mid_pronav(stblc, vtelc, tblc, vbelc, gnav, grav_bias, grav):
    dtbc = float(np.linalg.norm(stblc))
    utblc = stblc * (1.0 / dtbc)
    utbbc = tblc @ utblc
    polar = polar_from_cart(utbbc)
    psiobcx = float(polar[1]) * DEG
    thtobcx = float(polar[2]) * DEG
    vtblc = vtelc - vbelc
    dvtbc = abs(float(utblc @ vtblc))
    tgoc = dtbc / dvtbc
    woelc = _skew(utblc) @ vtblc * (1.0 / dtbc)
    grav_comp = np.array([0.0, 0.0, grav_bias * grav], dtype=float)
    acbx = tblc @ ((_skew(woelc) @ utblc * gnav * dvtbc - grav_comp) * (1.0 / AGRAV))
    return acbx, woelc, tgoc, dtbc, psiobcx, thtobcx


def _cpp_term_comp(sbel, stel, vbel, vtel, fspcb, tblc, gnav, psipb, thtpb, sigdpy, sigdpz):
    sbtl = sbel - stel
    dbt = float(np.linalg.norm(sbtl))
    dum = float(sbtl @ (vbel - vtel))
    dcvel = abs(dum / dbt)
    fspcb1 = float(fspcb[0])
    adely = fspcb1 * tan(psipb) / AGRAV
    adelz = fspcb1 * tan(thtpb) / (cos(psipb) * AGRAV)
    gravl = np.array([0.0, 0.0, 1.0], dtype=float)
    gravb = tblc @ gravl
    gn = gnav * dcvel
    apny = gn * sigdpz / (cos(psipb) * AGRAV)
    apnz = gn * (sigdpz * tan(thtpb) * tan(psipb) + sigdpy / cos(thtpb)) / AGRAV
    all_ = apny + adely - float(gravb[1])
    ann = apnz + adelz + float(gravb[2])
    acbx = np.array([0.0, all_, -ann], dtype=float)
    return acbx, gn, apny, apnz, adely, adelz


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


def _ready(*, mguid):
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
        ("FSPCB", FSPCB, "vec", "out", "ins"),
        ("STEL", STEL, "vec", "out", "sensor"),
        ("VTEL", VTEL, "vec", "out", "sensor"),
        ("psipb", PSIPB, "real", "out", "sensor"),
        ("thtpb", THTPB, "real", "out", "sensor"),
        ("ththb", THTHB, "real", "diag", "sensor"),
        ("sigdpy", SIGDPY, "real", "out", "sensor"),
        ("sigdpz", SIGDPZ, "real", "out", "sensor"),
        ("SBTL", SBTL, "vec", "diag", "sensor"),
        ("SBEL", SBEL, "vec", "state", "newton"),
        ("VBEL", VBEL, "vec", "out", "newton"),
        ("gmax", GMAX, "real", "diag", "aerodynamics"),
        ("grav", GRAV, "real", "out", "environment"),
    ):
        store.define(Field(name, value, ftype, role, module))
    store.set("mguid", mguid)
    store.set("gnav", GNAV)
    store.set("grav_bias", GRAV_BIAS)
    store.set("line_gain", LINE_GAIN)
    store.set("nl_gain_fact", NL_GAIN_FACT)
    store.set("decrement", DECREMENT)
    store.set("thtflx", THTFLX)
    store.set("epchta", 0.0)
    store.set("STELM", STELM)
    store.set("VTELC", VTELC)
    store.set("ancomx", 123.0)
    store.set("alcomx", 456.0)
    guid.initialize(vehicle, _ctx())
    return vehicle, guid


def test_agm6_mguid_26_runs_mid_and_term():
    """mguid=26: mid_line then term_comp; commands from term, mid diags set."""
    tblc = _tblc()
    stalc = STELM - SAEL
    stblc = STELM - SBELC
    mid_acbx, want_dtbc, want_vbeo = _cpp_mid_line(
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
    term_acbx, *_ = _cpp_term_comp(
        SBEL, STEL, VBEL, VTEL, FSPCB, tblc, GNAV, PSIPB, THTHB, SIGDPY, SIGDPZ
    )
    mid_al, mid_an = _limit_commands(mid_acbx, GMAX)
    term_al, term_an = _limit_commands(term_acbx, GMAX)
    assert (mid_al, mid_an) != pytest.approx((term_al, term_an), rel=RTOL, abs=ATOL)

    vehicle, guid = _ready(mguid=26)
    store = vehicle.store
    guid.execute(vehicle, _ctx())

    assert store.get("dtbc") == pytest.approx(want_dtbc, rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(store.get("VBEO"), want_vbeo, rtol=RTOL, atol=ATOL)
    assert store.get("alcomx") == pytest.approx(term_al, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(term_an, rel=RTOL, abs=ATOL)
    assert store.get("alcomx") != pytest.approx(mid_al, rel=RTOL, abs=ATOL)


def test_agm6_mguid_36_runs_mid_and_term():
    """mguid=36: mid_pronav(STELC) then term_comp."""
    tblc = _tblc()
    stblc = STELM - SBELC
    mid_acbx, want_woelc, want_tgoc, want_dtbc, *_ = _cpp_mid_pronav(
        stblc, VTELC, tblc, VBELC, GNAV, GRAV_BIAS, GRAV
    )
    term_acbx, *_ = _cpp_term_comp(
        SBEL, STEL, VBEL, VTEL, FSPCB, tblc, GNAV, PSIPB, THTHB, SIGDPY, SIGDPZ
    )
    mid_al, mid_an = _limit_commands(mid_acbx, GMAX)
    term_al, term_an = _limit_commands(term_acbx, GMAX)
    assert (mid_al, mid_an) != pytest.approx((term_al, term_an), rel=RTOL, abs=ATOL)

    vehicle, guid = _ready(mguid=36)
    store = vehicle.store
    guid.execute(vehicle, _ctx())

    np.testing.assert_allclose(store.get("WOELC"), want_woelc, rtol=RTOL, atol=ATOL)
    assert store.get("tgoc") == pytest.approx(want_tgoc, rel=RTOL, abs=ATOL)
    assert store.get("dtbc") == pytest.approx(want_dtbc, rel=RTOL, abs=ATOL)
    assert store.get("alcomx") == pytest.approx(term_al, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(term_an, rel=RTOL, abs=ATOL)
    assert store.get("alcomx") != pytest.approx(mid_al, rel=RTOL, abs=ATOL)


def test_agm6_mguid_46_runs_mid_and_term():
    """mguid=46: mid_pronav(STEL-SBELC) then term_comp; STBLC from mid digit 4."""
    tblc = _tblc()
    stblc_true = STEL - SBELC
    stblc_link = STELM - SBELC
    assert not np.allclose(stblc_true, stblc_link, rtol=RTOL, atol=ATOL)
    mid_acbx, want_woelc, want_tgoc, want_dtbc, *_ = _cpp_mid_pronav(
        stblc_true, VTELC, tblc, VBELC, GNAV, GRAV_BIAS, GRAV
    )
    term_acbx, *_ = _cpp_term_comp(
        SBEL, STEL, VBEL, VTEL, FSPCB, tblc, GNAV, PSIPB, THTHB, SIGDPY, SIGDPZ
    )
    mid_al, mid_an = _limit_commands(mid_acbx, GMAX)
    term_al, term_an = _limit_commands(term_acbx, GMAX)
    assert (mid_al, mid_an) != pytest.approx((term_al, term_an), rel=RTOL, abs=ATOL)

    vehicle, guid = _ready(mguid=46)
    store = vehicle.store
    guid.execute(vehicle, _ctx())

    np.testing.assert_allclose(store.get("STBLC"), stblc_true, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("WOELC"), want_woelc, rtol=RTOL, atol=ATOL)
    assert store.get("tgoc") == pytest.approx(want_tgoc, rel=RTOL, abs=ATOL)
    assert store.get("dtbc") == pytest.approx(want_dtbc, rel=RTOL, abs=ATOL)
    assert store.get("alcomx") == pytest.approx(term_al, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(term_an, rel=RTOL, abs=ATOL)
    assert store.get("alcomx") != pytest.approx(mid_al, rel=RTOL, abs=ATOL)
