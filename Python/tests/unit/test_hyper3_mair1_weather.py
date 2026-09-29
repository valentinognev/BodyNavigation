"""Task 82: HYPER3 tabular atmosphere (GHAME3 MAIR=1)."""

from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import R
from cadac.env.gravity import gravity
from cadac.env.iso62 import iso62
from cadac.eom.round3 import Round3Environment
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck, Table

RTOL = 1e-12
ATOL = 1e-14
ALT = 10000.0
DVBE = 250.0


def _tabular_weather_deck():
    """Planted WEATHER deck: RHX/CTMP/WPRES vs WALT (ROCKET6 table names)."""
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
        int_step=0.1,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )
    fields.update(kwargs)
    return SimContext(**fields)


def _vehicle(mair, weather_deck=None, alt=ALT, dvbe=DVBE):
    store = StateStore()
    vehicle = SimpleNamespace(store=store)
    env = Round3Environment(weather_deck=weather_deck)
    env.define(vehicle)
    store.define(Field("alt", 0.0, "real", "out", "newton"))
    store.define(Field("dvbe", 0.0, "real", "out", "newton"))
    store.set("alt", alt)
    store.set("dvbe", dvbe)
    store.set("mair", mair)
    return vehicle, env


def test_mair1_rho_press_temp_match_table_lookup():
    """MAIR=1 look_up RHX/CTMP/WPRES vs altitude matches TABLE interpolation."""
    deck = _tabular_weather_deck()
    vehicle, env = _vehicle(mair=1, weather_deck=deck)
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
    np.testing.assert_allclose(store.get("grav"), gravity(ALT), rtol=RTOL, atol=ATOL)


def test_mair1_without_weather_deck_raises():
    vehicle, env = _vehicle(mair=1, weather_deck=None)
    with pytest.raises(ValueError, match="weather"):
        env.execute(vehicle, _ctx())


def test_mair0_iso_unchanged_with_weather_deck_present():
    """MAIR=0 keeps ISO-62 even if a weather deck is attached."""
    deck = _tabular_weather_deck()
    vehicle, env = _vehicle(mair=0, weather_deck=deck, alt=3000.0, dvbe=250.0)
    env.execute(vehicle, _ctx())
    atm = iso62(3000.0, 250.0)
    store = vehicle.store
    np.testing.assert_allclose(store.get("rho"), atm["rho"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("press"), atm["press"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("mach"), atm["mach"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("pdynmc"), atm["pdynmc"], rtol=RTOL, atol=ATOL)
