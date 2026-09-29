"""CRUISE5 GPS module (MGPS 0/1/2) — Fortran S2 / S2I from MODULE.FOR."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cadac.constants import AGRAV, DEG, REARTH
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round3.cruise5.vehicle import Cruise5

RTOL = 1e-12
ATOL = 1e-9
INT_STEP = 0.05
RGPS = 20183.0e3

# INGPS.ASC satellite geometry and filter scales (deterministic means)
AZGEX = (60.0, 152.0, -14.0, -81.0)
ELGEX = (48.0, 20.0, 8.0, 65.0)
PSPOS = 5.0
PSVEL = 0.2
PSTIL = 0.5e-3
PSACC = 1.0e-2
PSGYR = 1.0e-5
PSCBI = 3.0
PSCFR = 1.0
QSPOS = 0.1
QSVEL = 0.01
QSTIL = 2.0e-4
QSACC = 4.0e-4
QSGYR = 1.0e-6
QSCBI = 0.5
QSCFR = 0.1
RSPOS = 1.0
RSVEL = 0.5
DTIMGPS = 1.0
TC = 100.0
SBEL0 = np.array([-10000.0, 17320.0, -150.0], dtype=float)
VBEL0 = np.array([136.0, -235.5, 0.0], dtype=float)  # ~272 m/s at -60 deg


def _matcar(magnitude: float, azimuth: float, elevation: float) -> np.ndarray:
    cel = math.cos(elevation)
    return np.array(
        [
            magnitude * cel * math.cos(azimuth),
            magnitude * cel * math.sin(azimuth),
            -magnitude * math.sin(elevation),
        ],
        dtype=float,
    )


def _expected_sge(i: int) -> np.ndarray:
    el = ELGEX[i] / DEG
    az = AZGEX[i] / DEG
    angl = 1.570796 + el
    dum = REARTH * math.cos(angl)
    dge = dum + math.sqrt(dum * dum + RGPS * RGPS - REARTH * REARTH)
    return _matcar(dge, az, el)


def _ctx(sim_time=0.0, int_step=INT_STEP):
    return SimContext(sim_time, int_step, 0.0, 0.0, None, 0)


def _gps_module():
    from cadac.vehicles.round3.cruise5.gps import Cruise5Gps

    return Cruise5Gps()


def _plant_deck(store: StateStore):
    for name, value in (
        ("mgps", 0),
        ("ms2prt", 0),
        ("azgex1", AZGEX[0]),
        ("azgex2", AZGEX[1]),
        ("azgex3", AZGEX[2]),
        ("azgex4", AZGEX[3]),
        ("elgex1", ELGEX[0]),
        ("elgex2", ELGEX[1]),
        ("elgex3", ELGEX[2]),
        ("elgex4", ELGEX[3]),
        ("pspos", PSPOS),
        ("psvel", PSVEL),
        ("pstil", PSTIL),
        ("psacc", PSACC),
        ("psgyr", PSGYR),
        ("pscbi", PSCBI),
        ("pscfr", PSCFR),
        ("frapi", 0.0),
        ("frapa", 0.0),
        ("frapg", 0.0),
        ("frapc", 0.0),
        ("qspos", QSPOS),
        ("qsvel", QSVEL),
        ("qstil", QSTIL),
        ("qsacc", QSACC),
        ("qsgyr", QSGYR),
        ("qscbi", QSCBI),
        ("qscfr", QSCFR),
        ("fraq", 0.0),
        ("rspos", RSPOS),
        ("rsvel", RSVEL),
        ("frar", 0.0),
        ("tc", TC),
        ("brpat1", 0.0),
        ("brpat2", 0.0),
        ("brpat3", 0.0),
        ("brpat4", 0.0),
        ("rrrec1", 0.25),
        ("rrrec2", 0.25),
        ("rrrec3", 0.25),
        ("rrrec4", 0.25),
        ("rddyn1", 0.03),
        ("rddyn2", 0.03),
        ("rddyn3", 0.03),
        ("rddyn4", 0.03),
        ("cbias", 0.0),
        ("cfreq", 0.1),
        ("dtimgps", DTIMGPS),
        ("tanlat", 0.0),
    ):
        store.set(name, value)


def _plant_kinematics(store: StateStore, time: float = 0.0):
    eye = np.eye(3, dtype=float)
    for name, value, ftype, role, module in (
        ("time", time, "real", "exec", "environment"),
        ("SBEL", SBEL0, "vec", "state", "newton"),
        ("VBEL", VBEL0, "vec", "state", "newton"),
        ("VBELC", VBEL0.copy(), "vec", "out", "ins"),
        ("ESTTC", np.zeros(3), "vec", "out", "ins"),
        ("FSPCB", np.array([0.0, 0.0, -AGRAV], dtype=float), "vec", "out", "ins"),
        ("TBLC", eye, "mat", "out", "ins"),
        ("TVL", eye, "mat", "out", "newton"),
    ):
        if name not in store:
            store.define(Field(name, value, ftype, role, module))
        store.set(name, value)


def _ready(mgps: int, time: float = 0.0):
    vehicle = type("V", (), {"store": StateStore()})()
    gps = _gps_module()
    gps.define(vehicle)
    _plant_deck(vehicle.store)
    _plant_kinematics(vehicle.store, time=time)
    vehicle.store.set("mgps", mgps)
    gps.initialize(vehicle, _ctx(sim_time=time))
    return gps, vehicle


def test_cruise5_has_gps_module():
    # Break: gps not on Cruise5.modules / wrong Fortran S2 slot (after newton).
    vehicle = Cruise5("UAV", None, None)
    names = [module.name for module in vehicle.modules]
    assert "gps" in names
    assert names.index("gps") > names.index("newton")
    assert names.index("gps") < names.index("seeker")
    gps = next(m for m in vehicle.modules if m.name == "gps")
    assert type(gps).__name__ == "Cruise5Gps"


def test_s2i_satellite_positions_match_fortran():
    # Break: S2I SGE / PP diagonal wrong vs INGPS AZGEX/ELGEX formulas.
    gps, vehicle = _ready(0)
    store = vehicle.store
    for i in range(4):
        np.testing.assert_allclose(gps.sge[:, i], _expected_sge(i), rtol=RTOL, atol=ATOL)
    assert store.get("pspos") == PSPOS
    np.testing.assert_allclose(gps.pp[0, 0], PSPOS**2, rtol=RTOL, atol=0.0)
    np.testing.assert_allclose(gps.pp[15, 15], PSCBI**2, rtol=RTOL, atol=0.0)
    np.testing.assert_allclose(gps.qq[0, 0], QSPOS**2, rtol=RTOL, atol=0.0)
    np.testing.assert_allclose(gps.rr[0, 0], RSPOS**2, rtol=RTOL, atol=0.0)


def test_mgps_zero_skips_handshake_updates():
    # Break: MGPS=0 still writes GUSTTCL / sets mgps=2.
    gps, vehicle = _ready(0, time=5.0)
    store = vehicle.store
    gust0 = np.asarray(store.get("GUSTTCL"), dtype=float).copy()
    gps.execute(vehicle, _ctx(sim_time=5.0))
    assert store.get("mgps") == 0
    np.testing.assert_array_equal(store.get("GUSTTCL"), gust0)
    # Filter still extrapolates 1-sig diagnostics (Fortran S2 before MGPS gate).
    assert store.get("pspos") > 0.0


def test_mgps_one_enables_without_update_before_dtimgps():
    # Break: MGPS=1 does not enable meas path / immediately forces update.
    gps, vehicle = _ready(1, time=0.0)
    store = vehicle.store
    # TGPS=0 after S2I; time=0 < DTIMGPS → stay enabled, no update.
    gps.execute(vehicle, _ctx(sim_time=0.0))
    assert store.get("mgps") == 1
    np.testing.assert_array_equal(store.get("GUSTTCL"), np.zeros(3))


def test_mgps_two_update_writes_s4_handshake_fields():
    # Break: MGPS=2 path does not write GUSTTCL/… or leave mgps=2 for S4 reset.
    gps, vehicle = _ready(1, time=0.0)
    store = vehicle.store
    # Advance past DTIMGPS so S2 sets MGPS=2 and applies Kalman update.
    store.set("time", DTIMGPS)
    gps.execute(vehicle, _ctx(sim_time=DTIMGPS))
    assert store.get("mgps") == 2
    gust = np.asarray(store.get("GUSTTCL"), dtype=float)
    guv = np.asarray(store.get("GUVBEL"), dtype=float)
    gure = np.asarray(store.get("GURECEL"), dtype=float)
    guf = np.asarray(store.get("GUFSPB"), dtype=float)
    guw = np.asarray(store.get("GUWBEB"), dtype=float)
    assert np.all(np.isfinite(gust))
    assert np.all(np.isfinite(guv))
    assert np.all(np.isfinite(gure))
    assert np.all(np.isfinite(guf))
    assert np.all(np.isfinite(guw))
    # Fortran resets XH after writing handshake; leave MGPS=2 for S4→1.
    assert np.allclose(gps.xh, 0.0)


def test_mgps_unknown_raises():
    gps, vehicle = _ready(0)
    vehicle.store.set("mgps", 3)
    with pytest.raises(ValueError, match="mgps"):
        gps.execute(vehicle, _ctx())
