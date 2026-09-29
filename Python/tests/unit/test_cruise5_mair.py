"""Task 79: CRUISE5 atmosphere/wind MAIR=|MATM|MWIND| packing."""

from math import cos, sin
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import R, RAD
from cadac.env.gravity import gravity
from cadac.env.iso62 import iso62
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck, Table
from cadac.vehicles.round3.cruise5.environment import Cruise5Environment

RTOL = 1e-12
ATOL = 1e-14

ALT = 10000.0
DVBE = 250.0
DT = 0.05
DVAEL = 10.0
PSIWLX = 90.0
DVAE3 = 0.0
VBEG = np.array([DVBE, 0.0, 0.0], dtype=float)
# Fortran G2 wind filter time constant is fixed at 1 s.
TWIND = 1.0


def _tabular_weather_deck():
    """Planted WEATHER deck: RHX/CTMP/WPRES vs altitude."""
    x1 = np.array([0.0, 20000.0])
    density = Table(
        name="density",
        dim=1,
        x1=x1,
        x2=None,
        x3=None,
        values=np.array([1.1, 0.08]),
    )
    pressure = Table(
        name="pressure",
        dim=1,
        x1=x1,
        x2=None,
        x3=None,
        values=np.array([102000.0, 5500.0]),
    )
    temperature = Table(
        name="temperature",
        dim=1,
        x1=x1,
        x2=None,
        x3=None,
        values=np.array([18.0, -58.0]),
    )
    return Datadeck.from_tables([density, pressure, temperature])


def _ctx(**kwargs):
    fields = dict(
        sim_time=0.0,
        int_step=DT,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )
    fields.update(kwargs)
    return SimContext(**fields)


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


def _vehicle(
    mair,
    *,
    weather_deck=None,
    alt=ALT,
    dvbe=DVBE,
    vbeg=None,
    dvael=DVAEL,
    psiwlx=PSIWLX,
    dvae3=DVAE3,
):
    store = StateStore()
    vehicle = SimpleNamespace(store=store, family="cruise5")
    env = Cruise5Environment(weather_deck=weather_deck)
    env.define(vehicle)
    store.define(Field("alt", 0.0, "real", "out", "newton"))
    store.define(Field("dvbe", 0.0, "real", "out", "newton"))
    store.define(Field("vbeg", (0.0, 0.0, 0.0), "vec", "state", "newton"))
    store.set("alt", alt)
    store.set("dvbe", dvbe)
    store.set("vbeg", np.array(VBEG if vbeg is None else vbeg, dtype=float))
    store.set("mair", mair)
    store.set("int_step_new", DT)
    store.set("dvael", dvael)
    store.set("psiwlx", psiwlx)
    store.set("dvae3", dvae3)
    return vehicle, env


def test_mair1_constant_wind_from_dvael_psiwlx():
    """MAIR=1 → MATM=0,MWIND=1: ISO atmos + constant wind smoothed into VAEL."""
    vehicle, env = _vehicle(mair=1)
    env.execute(vehicle, _ctx())
    store = vehicle.store

    want = _smoothed_vael(DVAEL, PSIWLX, DVAE3, TWIND, DT)
    np.testing.assert_allclose(store.get("VAEL"), want, rtol=RTOL, atol=ATOL)
    assert store.get("VAEL")[1] < 0.0  # psiwlx=90° → south/east local wind

    vbag = VBEG - store.get("VAEL")
    dvba = float(np.linalg.norm(vbag))
    np.testing.assert_allclose(store.get("dvba"), dvba, rtol=RTOL, atol=ATOL)

    atm = iso62(ALT, DVBE)
    np.testing.assert_allclose(store.get("rho"), atm["rho"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("press"), atm["press"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("mach"), abs(dvba / atm["vsound"]), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        store.get("pdynmc"), 0.5 * atm["rho"] * dvba * dvba, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(store.get("grav"), gravity(ALT), rtol=RTOL, atol=ATOL)


def test_mair10_tabular_atmosphere_from_weather_deck():
    """MAIR=10 → MATM=1,MWIND=0: tabular density/temp/press; no wind."""
    deck = _tabular_weather_deck()
    vehicle, env = _vehicle(mair=10, weather_deck=deck)
    env.execute(vehicle, _ctx())
    store = vehicle.store

    rho = deck.look_up("density", ALT)
    press = deck.look_up("pressure", ALT)
    tempc = deck.look_up("temperature", ALT)
    tempk = tempc + 273.16
    vsound = (1.4 * R * tempk) ** 0.5

    np.testing.assert_allclose(store.get("rho"), rho, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("press"), press, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("vsound"), vsound, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("mach"), abs(DVBE / vsound), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        store.get("pdynmc"), 0.5 * rho * DVBE * DVBE, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(store.get("VAEL"), np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("dvba"), DVBE, rtol=RTOL, atol=ATOL)


def test_mair10_without_weather_deck_raises():
    vehicle, env = _vehicle(mair=10, weather_deck=None)
    with pytest.raises(ValueError, match="weather"):
        env.execute(vehicle, _ctx())


def test_mair0_iso_unchanged():
    """MAIR=0 keeps ISO-62 and zero wind."""
    vehicle, env = _vehicle(mair=0, alt=3000.0, dvbe=250.0)
    env.execute(vehicle, _ctx())
    atm = iso62(3000.0, 250.0)
    store = vehicle.store
    np.testing.assert_allclose(store.get("rho"), atm["rho"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("press"), atm["press"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("mach"), atm["mach"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("pdynmc"), atm["pdynmc"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VAEL"), np.zeros(3), rtol=RTOL, atol=ATOL)
