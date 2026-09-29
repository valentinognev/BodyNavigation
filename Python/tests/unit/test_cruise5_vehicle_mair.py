"""CRUISE5 vehicle wires Cruise5Environment — MAIR=1 constant wind without weather deck."""

from math import cos, sin

import numpy as np
import pytest

from cadac.constants import RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.vehicles.round3.cruise5.environment import Cruise5Environment
from cadac.vehicles.round3.cruise5.satellite import Cruise5Satellite
from cadac.vehicles.round3.cruise5.target import Cruise5Target
from cadac.vehicles.round3.cruise5.vehicle import Cruise5

RTOL = 1e-12
ATOL = 1e-14
DT = 0.05
DVAEL = 10.0
PSIWLX = 90.0
DVAE3 = 0.0
TWIND = 1.0
VBEG = np.array([250.0, 0.0, 0.0], dtype=float)


def _smoothed_vael(dvw, psiwlx, dvae3, twind, dt):
    vael_raw = np.array(
        [
            -dvw * cos(psiwlx * RAD),
            -dvw * sin(psiwlx * RAD),
            dvae3,
        ],
        dtype=float,
    )
    vael = np.zeros(3)
    vaeld = np.zeros(3)
    vaeld_new = (vael_raw - vael) * (1.0 / twind)
    return integrate(vaeld_new, vaeld, vael, dt)


def _run_mair1(vehicle):
    assert isinstance(vehicle.modules[0], Cruise5Environment)
    vehicle.define()
    store = vehicle.store
    store.set("alt", 10000.0)
    store.set("dvbe", 250.0)
    store.set("vbeg", VBEG.copy())
    store.set("mair", 1)
    store.set("int_step_new", DT)
    store.set("dvael", DVAEL)
    store.set("psiwlx", PSIWLX)
    store.set("dvae3", DVAE3)
    ctx = SimContext(
        sim_time=0.0,
        int_step=DT,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )
    vehicle.modules[0].execute(vehicle, ctx)
    return store


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Cruise5("UAV", None, None),
        lambda: Cruise5Target("TANK"),
        lambda: Cruise5Satellite("SAT"),
    ],
    ids=["cruise5", "target", "satellite"],
)
def test_cruise5_vehicle_mair1_constant_wind_no_weather_deck(factory):
    """Live CRUISE5 G2: mair=1 uses dvael/VAEL; must not demand a weather Datadeck."""
    store = _run_mair1(factory())
    want = _smoothed_vael(DVAEL, PSIWLX, DVAE3, TWIND, DT)
    np.testing.assert_allclose(store.get("VAEL"), want, rtol=RTOL, atol=ATOL)
    assert "dvael" in store
    assert store.get("dvael") == DVAEL
