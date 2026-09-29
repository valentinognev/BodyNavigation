"""AGM6 guidance_term_pronav (mguid terminal digit 5)."""
from math import atan2, cos, sin, sqrt, tan
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import AGRAV, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat3tr
from cadac.vehicles.flat6.agm6.guidance import Agm6Guidance

RTOL = 1e-12
ATOL = 1e-14
DT = 0.001
SMALL = 1.0e-7

GNAV = 3.0
GRAV_BIAS = 1.5
GMAX = 20.0
PSIBLx = 10.0
THTBLx = 3.0
PHIBLx = 2.0
STEL = np.array([34000.0, 11000.0, -200.0], dtype=float)
VTEL = np.array([0.0, -5.0, 0.0], dtype=float)
SBEL = np.array([120.0, -40.0, -6980.0], dtype=float)
VBEL = np.array([248.0, 12.0, 4.0], dtype=float)
PSIPB = 0.05
THTPB = -0.03
SIGDPY = 0.02
SIGDPZ = -0.01
SBELC = np.array([100.0, -50.0, -7000.0], dtype=float)
STELM = np.array([33000.0, 10000.0, -100.0], dtype=float)
VTELC = np.array([0.0, -5.0, 0.0], dtype=float)


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


def _cpp_term_pronav(sbel, stel, vbel, vtel, tblc, gnav, grav_bias, psipb, thtpb, sigdpy, sigdpz):
    # AGM6 Missile::guidance_term_pronav — kinematic LOS-rate, no compensation.
    sbtl = sbel - stel
    dbt = float(np.linalg.norm(sbtl))
    dum = float(sbtl @ (vbel - vtel))
    dcvel = abs(dum / dbt)
    gravl = np.array([0.0, 0.0, grav_bias], dtype=float)
    gravb = tblc @ gravl
    gn = gnav * dcvel
    apny = gn * sigdpz / (cos(psipb) * AGRAV)
    apnz = gn * (sigdpz * tan(thtpb) * tan(psipb) + sigdpy / cos(thtpb)) / AGRAV
    all_ = apny - float(gravb[1])
    ann = apnz + float(gravb[2])
    acbx = np.array([0.0, all_, -ann], dtype=float)
    return acbx, gn, apny, apnz


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


def _ready(*, mguid=5, gnav=GNAV, grav_bias=GRAV_BIAS, gmax=GMAX):
    vehicle = SimpleNamespace(store=StateStore())
    guid = Agm6Guidance()
    guid.define(vehicle)
    store = vehicle.store
    tblc = _tblc()
    for name, value, ftype, role, module in (
        ("mnav", 0, "int", "out", "datalink"),
        ("STCEL", np.zeros(3), "vec", "out", "datalink"),
        ("VTCEL", np.zeros(3), "vec", "out", "datalink"),
        ("SBELC", SBELC, "vec", "out", "ins"),
        ("VBELC", np.zeros(3), "vec", "out", "ins"),
        ("TBLC", tblc, "mat", "out", "ins"),
        ("FSPCB", np.zeros(3), "vec", "out", "ins"),
        ("STEL", STEL, "vec", "out", "sensor"),
        ("VTEL", VTEL, "vec", "out", "sensor"),
        ("psipb", PSIPB, "real", "out", "sensor"),
        ("thtpb", THTPB, "real", "out", "sensor"),
        ("ththb", THTPB, "real", "diag", "sensor"),
        ("sigdpy", SIGDPY, "real", "out", "sensor"),
        ("sigdpz", SIGDPZ, "real", "out", "sensor"),
        ("SBEL", SBEL, "vec", "state", "newton"),
        ("VBEL", VBEL, "vec", "out", "newton"),
        ("gmax", gmax, "real", "diag", "aerodynamics"),
        ("grav", AGRAV, "real", "out", "environment"),
    ):
        store.define(Field(name, value, ftype, role, module))
    store.set("mguid", mguid)
    store.set("gnav", gnav)
    store.set("grav_bias", grav_bias)
    store.set("epchta", 0.0)
    store.set("STELM", STELM)
    store.set("VTELC", VTELC)
    store.set("ancomx", 123.0)
    store.set("alcomx", 456.0)
    guid.initialize(vehicle, _ctx())
    return vehicle, guid


def test_agm6_mguid_5_term_pronav_sets_ancomx():
    tblc = _tblc()
    vehicle, guid = _ready(mguid=5)
    store = vehicle.store
    want_acbx, want_gn, want_apny, want_apnz = _cpp_term_pronav(
        SBEL, STEL, VBEL, VTEL, tblc, GNAV, GRAV_BIAS, PSIPB, THTPB, SIGDPY, SIGDPZ
    )
    acbx = guid.guidance_term_pronav(vehicle)
    np.testing.assert_allclose(acbx, want_acbx, rtol=RTOL, atol=ATOL)
    assert acbx.shape == (3,)
    assert store.get("gn") == pytest.approx(want_gn, rel=RTOL, abs=ATOL)
    assert store.get("apny") == pytest.approx(want_apny, rel=RTOL, abs=ATOL)
    assert store.get("apnz") == pytest.approx(want_apnz, rel=RTOL, abs=ATOL)

    guid.execute(vehicle, _ctx())
    alcomx, ancomx = _limit_commands(want_acbx, GMAX)
    assert store.get("alcomx") == pytest.approx(alcomx, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(ancomx, rel=RTOL, abs=ATOL)
    # Must not be compensated term_comp (no FSPCB / unit-g path).
    assert abs(store.get("ancomx")) > 0.0 or abs(store.get("alcomx")) > 0.0
