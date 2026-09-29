"""SAM6 mguide mid digit 3: guidance_mid_pronav toward radar IP."""

from math import atan2, cos, fabs, sin, sqrt

import numpy as np
import pytest

from cadac.constants import AGRAV, DEG
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import polar_from_cart, skew
from cadac.vehicles.flat6.sam6.guidance import Sam6Guidance

RTOL = 1e-12
ATOL = 1e-14
SMALL = 1e-7
DT = 0.001
GMAX = 40.0
GNAV = 3.1
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
SIEL_NORTH = np.array([1000.0, 0.0, 0.0], dtype=float)
VBELC = np.array([16.0, 5.0, -1.0], dtype=float)
SBELC = np.array([0.0, 0.0, 0.0], dtype=float)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ctx(int_step=DT, combus=None, vehicle_slot=0):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=vehicle_slot,
    )


def _circular_limit(allx, annx, gmax):
    aa = sqrt(allx * allx + annx * annx)
    if aa > gmax:
        aa = gmax
    if fabs(annx) < SMALL and fabs(allx) < SMALL:
        phi = 0.0
    else:
        phi = atan2(annx, allx)
    return aa * cos(phi), aa * sin(phi)


def _cpp_mid_pronav(siblc, tblc, vbelc, gnav):
    """Replica of Missile::guidance_mid_pronav (SAM6 guidance.cpp)."""
    siblc = np.asarray(siblc, dtype=float)
    tblc = np.asarray(tblc, dtype=float)
    vbelc = np.asarray(vbelc, dtype=float)
    dtbc = float(np.linalg.norm(siblc))
    utblc = siblc * (1.0 / dtbc)
    utbbc = tblc @ utblc
    polar = polar_from_cart(utbbc)
    psiobcx = float(polar[1]) * DEG
    thtobcx = float(polar[2]) * DEG
    dvtbc = fabs(float(utblc @ vbelc))
    tgoc = dtbc / dvtbc
    woelc = skew(utblc) @ vbelc * (1.0 / dtbc)
    acbx = tblc @ (skew(woelc) @ utblc) * gnav * dvtbc * (1.0 / AGRAV)
    return acbx, woelc, utblc, tgoc, dtbc, dvtbc, psiobcx, thtobcx


def _radar(siel1, name="R"):
    return Packet(
        name=name,
        type="RADAR0",
        status=1,
        vars={"SIEL1": np.asarray(siel1, dtype=float)},
    )


def _ready_mid3(siel=SIEL_NORTH):
    vehicle = _Vehicle()
    guid = Sam6Guidance()
    guid.define(vehicle)
    store = vehicle.store
    store.define(Field("gmax", GMAX, "real", "diag", "aerodynamics", ("plot",)))
    store.define(Field("TBLC", IDENTITY, "mat", "out", "ins"))
    store.define(Field("VBELC", VBELC, "vec", "out", "ins"))
    store.define(Field("SBELC", SBELC, "vec", "out", "ins"))
    store.set("mguide", 30)
    store.set("gnav", GNAV)
    combus = [
        Packet(name="SAM", type="MISSILE6", status=1, vars={}),
        _radar(siel),
    ]
    return vehicle, guid, _ctx(combus=combus, vehicle_slot=0)


def test_sam6_guid_mid_3():
    vehicle, guid, ctx = _ready_mid3()
    siblc = SIEL_NORTH - SBELC
    want_acbx, woelc, utblc, tgoc, dtbc, dvtbc, psiobcx, thtobcx = _cpp_mid_pronav(
        siblc, IDENTITY, VBELC, GNAV
    )
    alcomx, ancomx = _circular_limit(float(want_acbx[1]), float(-want_acbx[2]), GMAX)

    guid.execute(vehicle, ctx)

    store = vehicle.store
    assert np.isfinite(store.get("ancomx"))
    assert np.isfinite(store.get("alcomx"))
    assert _approx(store.get("ancomx"), ancomx)
    assert _approx(store.get("alcomx"), alcomx)
    np.testing.assert_allclose(store.get("SIBLC"), siblc, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("WOELC"), woelc, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("UTBLC"), utblc, rtol=RTOL, atol=ATOL)
    assert _approx(store.get("tgoc"), tgoc)
    assert _approx(store.get("dtbc"), dtbc)
    assert _approx(store.get("dvtbc"), dvtbc)
    assert _approx(store.get("psiobcx"), psiobcx)
    assert _approx(store.get("thtobcx"), thtobcx)
    assert _approx(store.get("allx"), float(want_acbx[1]))
    assert _approx(store.get("annx"), float(-want_acbx[2]))
