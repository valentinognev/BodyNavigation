"""Task 81: FALCON6 tabular weather (MWIND=3, MATMO=3) — Fortran G2 WEATHER deck."""

from math import cos, sin
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import R, RAD
from cadac.eom.flat6 import Flat6Environment
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck, Table

RTOL = 1e-12
ATOL = 1e-14

DT = 0.01
TWIND = 0.1
VAED3 = 0.0
HBE = 10000.0
VBEL = np.array([180.0, 0.0, 0.0], dtype=float)

# Fortran G2: MAIR=|MTURB|MWIND|MATMO| → 33 = tabular wind + tabular atmosphere
MAIR_TABULAR = 33


def _weather_deck():
    """Minimal WEATHER_DECK: RHX/CTMP/WPRES + WVEL/WDIR vs WALT (MODULE.FOR)."""
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


def _smoothed_vael(dvw, psiwdx, vaed3, twind, dt):
    vael_raw = np.array(
        [
            -dvw * cos(psiwdx * RAD),
            -dvw * sin(psiwdx * RAD),
            vaed3,
        ],
        dtype=float,
    )
    vaels = np.zeros(3)
    vaelsd = np.zeros(3)
    vaelsd_new = (vael_raw - vaels) * (1.0 / twind)
    return integrate(vaelsd_new, vaelsd, vaels, dt)


def _vehicle(mair, weather_deck=None, hbe=HBE):
    s = StateStore()
    vehicle = SimpleNamespace(store=s)
    env = Flat6Environment(weather_deck=weather_deck)
    env.define(vehicle)
    s.define(Field("hbe", hbe, "real", "out", "newton"))
    s.define(Field("VBEL", VBEL, "vec", "diag", "newton"))
    s.set("mair", mair)
    s.set("twind", TWIND)
    s.set("vaed3", VAED3)
    s.set("psiwdx", 0.0)
    s.set("dvae", 0.0)
    return vehicle, env


def test_falcon6_mair33_digit_packing():
    """Fortran G2: MTURB=INT(MAIR/100), MWIND=…/10, MATMO=ones → 33 = (0,3,3)."""
    mair = MAIR_TABULAR
    mturb = int(mair / 100)
    mwind = int((mair - mturb * 100) / 10)
    matmo = mair - mturb * 100 - mwind * 10
    assert (mturb, mwind, matmo) == (0, 3, 3)


def test_falcon6_matmo3_mwind3_density_and_wind_vs_table():
    """mair=33 reads WEATHER_DECK density/temp and tabular horizontal wind."""
    deck = _weather_deck()
    vehicle, env = _vehicle(mair=MAIR_TABULAR, weather_deck=deck)
    env.execute(vehicle, SimpleNamespace(int_step=DT))
    store = vehicle.store

    rho = deck.look_up("density", HBE)
    press = deck.look_up("pressure", HBE)
    tempc = deck.look_up("temperature", HBE)
    tempk = tempc + 273.16
    vsound = (1.4 * R * tempk) ** 0.5
    dvw = deck.look_up("speed", HBE)
    psiwdx = deck.look_up("direction", HBE)
    want_vael = _smoothed_vael(dvw, psiwdx, VAED3, TWIND, DT)
    vbal = VBEL - want_vael
    dvba = float(np.linalg.norm(vbal))

    np.testing.assert_allclose(store.get("rho"), rho, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("press"), press, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("tempk"), tempk, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("vsound"), vsound, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VAEL"), want_vael, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("dvba"), dvba, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("vmach"), abs(dvba / vsound), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        store.get("pdynmc"), 0.5 * rho * dvba * dvba, rtol=RTOL, atol=ATOL
    )
    assert abs(dvw - 10.0) < 1e-12
    assert abs(psiwdx - 90.0) < 1e-12


def test_falcon6_matmo3_mwind3_without_weather_deck_raises():
    vehicle, env = _vehicle(mair=MAIR_TABULAR, weather_deck=None)
    with pytest.raises(ValueError, match="weather"):
        env.execute(vehicle, SimpleNamespace(int_step=DT))
