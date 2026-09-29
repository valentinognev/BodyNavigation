"""Task 83: HYPER6 tabular atmosphere/wind (GHAME6 MATMO=3, MWIND=3)."""

from math import cos, sin
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import R, RAD
from cadac.eom.round6 import Round6Environment
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.wgs84 import cad_in_geo84
from cadac.tables.lookup import Datadeck, Table

RTOL = 1e-12
ATOL = 1e-14

DT = 0.01
TWIND = 0.1
VAED3 = 0.0
ALT = 10000.0
VBED = np.array([1000.0, 0.0, 0.0], dtype=float)

# GHAME6 Fortran packs MAIR=|MTURB|MWIND|MATMO| → mair=33 is tabular both.
# Round6 default (ROCKET6/C++ HYPER6) is |MATMO|MTURB|MWIND| → same switches = 303.
MAIR_GHAME6_TABULAR = 33
MAIR_ROCKET6_EQUIV = 303


def _weather_deck():
    """Minimal WEATHER_DECK: density/pressure/temperature + speed/direction."""
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
    speed = Table(
        name="speed",
        dim=1,
        x1=x1,
        x2=None,
        x3=None,
        values=np.array([0.0, 20.0]),
    )
    direction = Table(
        name="direction",
        dim=1,
        x1=x1,
        x2=None,
        x3=None,
        values=np.array([0.0, 180.0]),
    )
    return Datadeck.from_tables(
        [density, pressure, temperature, speed, direction]
    )


def _smoothed_vaed(dvw, psiwdx, vaed3, twind, dt):
    vaed_raw = np.array(
        [
            -dvw * cos(psiwdx * RAD),
            -dvw * sin(psiwdx * RAD),
            vaed3,
        ],
        dtype=float,
    )
    vaeds = np.zeros(3)
    vaedsd = np.zeros(3)
    vaedsd_new = (vaed_raw - vaeds) * (1.0 / twind)
    return integrate(vaedsd_new, vaedsd, vaeds, dt)


def _vehicle(mair, weather_deck=None, alt=ALT):
    s = StateStore()
    # HYPER6 vehicle type; Round6Environment still shared with ROCKET6.
    vehicle = SimpleNamespace(store=s, type="HYPER6")
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
    s.set("VBED", np.array(VBED, dtype=float))
    s.set("mair", mair)
    s.set("twind", TWIND)
    s.set("vaed3", VAED3)
    s.set("psiwdx", 0.0)
    s.set("dvae", 0.0)
    return vehicle, env


def test_ghame6_digit_packing_differs_from_rocket6():
    """GHAME6 Fortran: MTURB=INT(MAIR/100), MWIND=…/10, MATMO=ones.
    Round6 ROCKET6 path: matmo=hundreds, mturb=tens, mwind=ones.
    Fortran mair=33 (tabular atmos+wind) ↔ ROCKET6-packed mair=303."""
    mair_f = MAIR_GHAME6_TABULAR
    mturb = int(mair_f / 100)
    mwind = int((mair_f - mturb * 100) / 10)
    matmo = mair_f - mturb * 100 - mwind * 10
    assert (mturb, mwind, matmo) == (0, 3, 3)
    mair_rocket6 = matmo * 100 + mturb * 10 + mwind
    assert mair_rocket6 == MAIR_ROCKET6_EQUIV


def test_hyper6_matmo3_mwind3_density_and_wind_vs_table():
    """mair=33 (GHAME6) reads WEATHER_DECK density and tabular wind."""
    deck = _weather_deck()
    vehicle, env = _vehicle(mair=MAIR_GHAME6_TABULAR, weather_deck=deck)
    env.execute(vehicle, SimpleNamespace(int_step=DT))
    store = vehicle.store

    rho = deck.look_up("density", ALT)
    press = deck.look_up("pressure", ALT)
    tempc = deck.look_up("temperature", ALT)
    tempk = tempc + 273.16
    vsound = (1.4 * R * tempk) ** 0.5
    dvw = deck.look_up("speed", ALT)
    psiwdx = deck.look_up("direction", ALT)
    want_vaed = _smoothed_vaed(dvw, psiwdx, VAED3, TWIND, DT)
    vbad = VBED - want_vaed
    dvba = float(np.linalg.norm(vbad))

    np.testing.assert_allclose(store.get("rho"), rho, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("press"), press, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("tempc"), tempc, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("tempk"), tempk, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("vsound"), vsound, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VAED"), want_vaed, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("dvba"), dvba, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("vmach"), abs(dvba / vsound), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        store.get("pdynmc"), 0.5 * rho * dvba * dvba, rtol=RTOL, atol=ATOL
    )
    assert abs(dvw - 10.0) < 1e-12
    assert abs(psiwdx - 90.0) < 1e-12


def test_hyper6_matmo3_mwind3_without_weather_deck_raises():
    vehicle, env = _vehicle(mair=MAIR_GHAME6_TABULAR, weather_deck=None)
    with pytest.raises(ValueError, match="weather"):
        env.execute(vehicle, SimpleNamespace(int_step=DT))


def test_hyper6_rocket6_packed_303_also_tabular():
    """ROCKET6-packed 303 (matmo=3,mwind=3) is the same switches as GHAME6 33."""
    deck = _weather_deck()
    vehicle, env = _vehicle(mair=MAIR_ROCKET6_EQUIV, weather_deck=deck)
    env.execute(vehicle, SimpleNamespace(int_step=DT))
    store = vehicle.store
    np.testing.assert_allclose(
        store.get("rho"), deck.look_up("density", ALT), rtol=RTOL, atol=ATOL
    )
    dvw = deck.look_up("speed", ALT)
    psiwdx = deck.look_up("direction", ALT)
    want = _smoothed_vaed(dvw, psiwdx, VAED3, TWIND, DT)
    np.testing.assert_allclose(store.get("VAED"), want, rtol=RTOL, atol=ATOL)
