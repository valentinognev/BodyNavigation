"""Task 40: ROCKET6 matmo=2 tabular atmosphere (mair=2xx)."""

from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import R, RAD
from cadac.eom.round6 import Round6Environment
from cadac.kernel.state import Field, StateStore
from cadac.math.wgs84 import cad_in_geo84
from cadac.tables.lookup import Datadeck, Table

RTOL = 1e-12
ATOL = 1e-14
ALT = 10000.0


def _tabular_atmosphere_deck():
    """Minimal WEATHER_DECK with density/pressure/temperature vs altitude."""
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


def _vehicle(mair, weather_deck=None, alt=ALT):
    s = StateStore()
    vehicle = SimpleNamespace(store=s, family="rocket6")
    env = Round6Environment(weather_deck=weather_deck)
    env.define(vehicle)
    s.define(Field("time", 0.0, "real", "exec", "kinematics"))
    s.define(Field("alt", 0.0, "real", "out", "newton"))
    s.define(Field("SBII", (0.0, 0.0, 0.0), "vec", "state", "newton"))
    s.define(Field("VBED", (0.0, 0.0, 0.0), "vec", "state", "newton"))
    time = 0.0
    lon = 10.0 * RAD
    lat = 10.0 * RAD
    s.set("time", time)
    s.set("alt", alt)
    s.set("SBII", cad_in_geo84(lon, lat, alt, time))
    s.set("VBED", np.array([1000.0, 0.0, 0.0], dtype=float))
    s.set("mair", mair)
    return vehicle, env


def test_round6_matmo_2_tabular():
    """mair=200 selects tabular atmosphere from WEATHER_DECK without raise."""
    deck = _tabular_atmosphere_deck()
    vehicle, env = _vehicle(mair=200, weather_deck=deck)
    env.execute(vehicle, SimpleNamespace(int_step=0.01))
    store = vehicle.store

    rho = deck.look_up("density", ALT)
    press = deck.look_up("pressure", ALT)
    tempc = deck.look_up("temperature", ALT)
    tempk = tempc + 273.16
    vsound = (1.4 * R * tempk) ** 0.5
    dvba = 1000.0

    np.testing.assert_allclose(store.get("rho"), rho, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("press"), press, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("tempc"), tempc, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("tempk"), tempk, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("vsound"), vsound, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("dvba"), dvba, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("vmach"), abs(dvba / vsound), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        store.get("pdynmc"), 0.5 * rho * dvba * dvba, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(store.get("VAED"), np.zeros(3), rtol=RTOL, atol=ATOL)


def test_round6_matmo_2_without_weather_deck_raises():
    vehicle, env = _vehicle(mair=200, weather_deck=None)
    with pytest.raises(ValueError, match="weather"):
        env.execute(vehicle, SimpleNamespace(int_step=0.01))
