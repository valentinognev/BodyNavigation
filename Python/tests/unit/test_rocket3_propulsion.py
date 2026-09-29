"""ROCKET3 A2 propulsion — MPROP 0/1/2 burning fuel flow / mass (Fortran MODULE.FOR)."""

import pytest

from cadac.constants import AGRAV
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round3.rocket3.propulsion import Rocket3Propulsion

RTOL = 1e-12
ATOL = 1e-14
DT = 0.01

# Stage-1 INLAUNCH values
FUELI = 6200.0
FUELR = 94.0
SPI = 270.0
AEXIT = 0.5
THRTL = 1.0
VMASSI = 10800.0
PRESS_SL = 101325.0


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(dt=DT):
    return SimContext(0.0, dt, 0.0, 0.0, None, 0)


def _ready(*, mprop=2, press=PRESS_SL, **kw):
    vehicle = _Vehicle()
    prop = Rocket3Propulsion()
    prop.define(vehicle)
    store = vehicle.store
    store.define(Field("press", press, "real", "out", "environment"))
    data = {
        "mprop": mprop,
        "fueli": FUELI,
        "fuelr": FUELR,
        "spi": SPI,
        "aexit": AEXIT,
        "thrtl": THRTL,
        "vmassi": VMASSI,
    }
    data.update(kw)
    for name, value in data.items():
        store.set(name, value)
    prop.initialize(vehicle, _ctx())
    return vehicle, prop


def test_mprop_2_burning_fuel_flow_and_mass():
    """MPROP=2: FLR=FUELR*THRTL; thrust/mass/fuel match Fortran A2 (sea-level PRESS)."""
    vehicle, prop = _ready(mprop=2)
    store = vehicle.store
    ctx = _ctx()

    # First step: integrate with prior FMASSFD=0 → FMASSF stays 0; then set flow.
    prop.execute(vehicle, ctx)
    flr = FUELR * THRTL
    want_thrust = flr * SPI * AGRAV + (101325.0 - PRESS_SL) * AEXIT
    assert store.get("mprop") == 2
    assert store.get("fmassfd") == pytest.approx(flr, rel=RTOL, abs=ATOL)
    assert store.get("thrustx") == pytest.approx(want_thrust / 1000.0, rel=RTOL, abs=ATOL)
    assert store.get("fmassf") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert store.get("vmass") == pytest.approx(VMASSI, rel=RTOL, abs=ATOL)
    assert store.get("fuel") == pytest.approx(FUELI, rel=RTOL, abs=ATOL)

    # Second step: FMASSF += FMASSFD * DER (ICOOR=0 Euler, prior rate).
    prop.execute(vehicle, ctx)
    want_fmassf = flr * DT
    assert store.get("fmassfd") == pytest.approx(flr, rel=RTOL, abs=ATOL)
    assert store.get("fmassf") == pytest.approx(want_fmassf, rel=RTOL, abs=ATOL)
    assert store.get("vmass") == pytest.approx(VMASSI - want_fmassf, rel=RTOL, abs=ATOL)
    assert store.get("fuel") == pytest.approx(FUELI - want_fmassf, rel=RTOL, abs=ATOL)


def test_mprop_0_no_thrust_zero_flow():
    vehicle, prop = _ready(mprop=0)
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("mprop") == 0
    assert store.get("thrustx") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert store.get("fmassfd") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert store.get("vmass") == pytest.approx(VMASSI, rel=RTOL, abs=ATOL)


def test_mprop_2_burnout_sets_coast():
    """When FUEL <= 0 after burn, Fortran sets MPROP=1 and zeros thrust/flow."""
    vehicle, prop = _ready(mprop=2, fueli=0.0)
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("mprop") == 1
    assert store.get("thrustx") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert store.get("fmassfd") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
