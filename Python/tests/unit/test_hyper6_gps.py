"""HYPER6 GPS module (mgps 0–3), source-faithful to Hyper::gps."""

import math

import numpy as np
import pytest

from cadac.constants import RAD, WEII3
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat3tr
from cadac.math.wgs84 import cad_in_geo84, cad_tdi84
from cadac.vehicles.round6.hyper6.vehicle import Hyper6

RTOL = 1e-12
ATOL = 1e-14

# Cape-ascent GPS filter deck (input_Cape_ascent_GPS.asc) and one-step EKF extrapolate of PP.
PPOS = 5.0
PVEL = 0.2
PCLOCKB = 3.0
PCLOCKF = 1.0
QPOS = 0.1
QVEL = 0.01
QCLOCKB = 0.5
QCLOCKF = 0.1
RPOS = 1.0
RVEL = 0.1
FACTP = 0.0
FACTQ = 0.0
FACTR = 0.0
UCTIME_COR = 100.0
GPS_ACQTIME = 10.0
GPS_STEP = 1.0
ALMANAC_TIME = 0.0
DEL_REARTH = 2317000.0
INT_STEP = 0.001
PP00_INIT = (PPOS * (1.0 + FACTP)) ** 2
# PHI*(PP+QQ*(dt/2))*PHI.T + QQ*(dt/2) with Cape P/Q and dt=0.001.
STD_POS_EXTRAP = 5.0000010039999045
STD_VEL_EXTRAP = 0.20000024999984378
STD_UCBIAS_EXTRAP = 3.000041833040833

LONX = -80.55
LATX = 28.45
ALT = 100.0
TIME = 0.0
SBIIC_OFF = np.array([50.0, -20.0, 10.0], dtype=float)
VBIIC_OFF = np.array([1.0, -0.5, 0.2], dtype=float)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(int_step=INT_STEP, sim_time=TIME):
    return SimContext(
        sim_time=sim_time,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _plant_geometry(store):
    time = TIME
    sbii = cad_in_geo84(LONX * RAD, LATX * RAD, ALT, time)
    tdi = cad_tdi84(LONX * RAD, LATX * RAD, ALT, time)
    tbd = mat3tr(0.0, 90.0 * RAD, 0.0)
    tbi = tbd @ tdi
    vbed = np.array([10.0, 0.0, 0.0], dtype=float)
    veic = np.array([-WEII3 * sbii[1], WEII3 * sbii[0], 0.0], dtype=float)
    vbii = tdi.T @ vbed + veic
    wbib = np.array([0.01, 0.02, 0.03], dtype=float)
    wbii = tbi.T @ wbib
    sbiic = sbii + SBIIC_OFF
    vbiic = vbii + VBIIC_OFF
    for name, value, ftype, role, module in (
        ("time", time, "real", "exec", "kinematics"),
        ("SBII", sbii, "vec", "state", "newton"),
        ("VBII", vbii, "vec", "state", "newton"),
        ("WBII", wbii, "vec", "out", "euler"),
        ("SBIIC", sbiic, "vec", "out", "ins"),
        ("VBIIC", vbiic, "vec", "out", "ins"),
        ("WBICI", wbii.copy(), "vec", "out", "ins"),
    ):
        if name not in store.names():
            store.define(Field(name, value, ftype, role, module))
        store.set(name, value)


def _plant_filter(store):
    for name, value in (
        ("almanac_time", ALMANAC_TIME),
        ("del_rearth", DEL_REARTH),
        ("gps_acqtime", GPS_ACQTIME),
        ("gps_step", GPS_STEP),
        ("uctime_cor", UCTIME_COR),
        ("ppos", PPOS),
        ("pvel", PVEL),
        ("pclockb", PCLOCKB),
        ("pclockf", PCLOCKF),
        ("qpos", QPOS),
        ("qvel", QVEL),
        ("qclockb", QCLOCKB),
        ("qclockf", QCLOCKF),
        ("rpos", RPOS),
        ("rvel", RVEL),
        ("factp", FACTP),
        ("factq", FACTQ),
        ("factr", FACTR),
    ):
        store.set(name, value)


def _gps_module():
    from cadac.vehicles.round6.hyper6.gps import Hyper6Gps

    return Hyper6Gps()


def _ready(mgps):
    vehicle = _Vehicle()
    gps = _gps_module()
    gps.define(vehicle)
    _plant_filter(vehicle.store)
    _plant_geometry(vehicle.store)
    vehicle.store.set("mgps", mgps)
    gps.initialize(vehicle, _ctx())
    return gps, vehicle


def test_hyper6_has_gps_module():
    # Break: gps not on Hyper6.modules / wrong C++ order (after propulsion, before ins).
    vehicle = Hyper6("Hypersonic", None, None)
    names = [module.name for module in vehicle.modules]
    assert "gps" in names
    assert names.index("gps") > names.index("propulsion")
    assert names.index("gps") < names.index("ins")
    gps = next(m for m in vehicle.modules if m.name == "gps")
    assert type(gps).__name__ == "Hyper6Gps"


def test_hyper6_mgps_2_one_step():
    # Break: mgps=1 does not fall through to extrapolate (std_pos==0 / PP still init).
    gps, vehicle = _ready(1)
    store = vehicle.store
    gps.execute(vehicle, _ctx())
    assert store.get("mgps") == 2
    assert store.get("gps_acq") == 1
    assert store.get("gps_epoch") == TIME
    assert store.get("std_pos") != 0.0
    assert gps.PP.shape == (8, 8)
    assert gps.PP[0, 0] != PP00_INIT
    assert store.get("std_pos") == pytest.approx(STD_POS_EXTRAP, rel=RTOL, abs=ATOL)
    assert store.get("std_vel") == pytest.approx(STD_VEL_EXTRAP, rel=RTOL, abs=ATOL)
    assert store.get("std_ucbias") == pytest.approx(
        STD_UCBIAS_EXTRAP, rel=RTOL, abs=ATOL
    )


def test_mgps_zero_returns_without_filter_outputs():
    gps, vehicle = _ready(0)
    gps.execute(vehicle, _ctx())
    assert vehicle.store.get("mgps") == 0
    assert vehicle.store.get("std_pos") == 0.0
    np.testing.assert_array_equal(vehicle.store.get("SXH"), np.zeros(3))


def test_mgps_four_raises():
    gps, vehicle = _ready(4)
    with pytest.raises(ValueError, match="mgps"):
        gps.execute(vehicle, _ctx())


def test_mgps_three_frozen_geometry_sxh_finite():
    gps, vehicle = _ready(1)
    gps.execute(vehicle, _ctx())
    vehicle.store.set("mgps", 3)
    gps.execute(vehicle, _ctx())
    sxh = np.asarray(vehicle.store.get("SXH"), dtype=float)
    assert np.all(np.isfinite(sxh))
    assert math.isfinite(vehicle.store.get("gdop"))
    assert vehicle.store.get("gdop") > 0.0
