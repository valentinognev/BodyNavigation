from math import cos, sin, sqrt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import PI, R, RAD
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.stoch import seed
from cadac.vehicles.flat6.agm6.environment import Agm6Environment

RTOL = 1e-12
ATOL = 1e-14

AGM6 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/AGM6_250217/AGM6"
WEATHER_ASC = AGM6 / "weather_deck.asc"

HBE = 7000.0
VBEL = np.array([293.0, 0.0, 0.0], dtype=float)
DVBE = 293.0
DT = 0.001
TWIND_DEFAULT = 0.1
TURB_LENGTH = 100.0
TURB_SIGMA = 0.5
DVAE = 5.0
PSIWDX = 0.0

# Hand-checked 1D interpolant of weather_deck.asc density at hbe=7000
# (5000 m → 0.70, 10000 m → 0.40): 0.70 + 0.4*(0.40-0.70) = 0.58
RHO_7000 = 0.58

# Harvested test_case_plot.csv t=0 VAEL3: Dryden tau with phipx=0, TBL=I,
# srand(12345), C++ def_ins+init_ins gauss, then markov_noise, then Dryden rand().
GOLDEN_T0_VAEL3 = 0.0173842
CSV_RTOL = 1e-5


def _weather_deck():
    _, tables = parse_asc_deck(WEATHER_ASC)
    return Datadeck.from_tables(tables)


def _ctx(int_step=DT):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _plant_newton(store, *, hbe=HBE, vbel=VBEL, dvbe=DVBE):
    store.define(Field("hbe", hbe, "real", "out", "newton"))
    store.define(Field("VBEL", tuple(vbel), "vec", "state", "newton"))
    store.define(Field("dvbe", dvbe, "real", "out", "newton"))


def _plant_kinematics(store):
    store.define(Field("TBD", np.eye(3), "mat", "out", "kinematics"))
    store.define(Field("alppx", 0.0, "real", "out", "kinematics"))
    store.define(Field("phipx", 0.0, "real", "out", "kinematics"))


def _ready(
    *,
    mair,
    weather_deck=None,
    kinematics=False,
    hbe=HBE,
    vbel=VBEL,
    dvbe=DVBE,
    dvae=0.0,
    psiwdx=0.0,
    twind=TWIND_DEFAULT,
    turb_length=TURB_LENGTH,
    turb_sigma=TURB_SIGMA,
    gauss_value=None,
    vaed3=0.0,
):
    vehicle = SimpleNamespace(store=StateStore())
    env = Agm6Environment(weather_deck=weather_deck)
    env.define(vehicle)
    _plant_newton(vehicle.store, hbe=hbe, vbel=vbel, dvbe=dvbe)
    if kinematics:
        _plant_kinematics(vehicle.store)
    store = vehicle.store
    store.set("mair", mair)
    store.set("dvae", dvae)
    store.set("psiwdx", psiwdx)
    store.set("twind", twind)
    store.set("turb_length", turb_length)
    store.set("turb_sigma", turb_sigma)
    store.set("vaed3", vaed3)
    if gauss_value is not None:
        store.set("gauss_value", gauss_value)
    env.initialize(vehicle, _ctx())
    return vehicle, env


def _smoothed_constant_wind(dvae, psiwdx, vaed3, twind, dt):
    vaed_raw = np.array(
        [
            -dvae * cos(psiwdx * RAD),
            -dvae * sin(psiwdx * RAD),
            vaed3,
        ],
        dtype=float,
    )
    vaels = np.zeros(3)
    vaelsd = np.zeros(3)
    vaedsd_new = (vaed_raw - vaels) * (1.0 / twind)
    return integrate(vaedsd_new, vaelsd, vaels, dt)


def test_parse_weather_deck_tables():
    deck = _weather_deck()
    assert deck.look_up("density", HBE) == pytest.approx(RHO_7000, rel=RTOL, abs=ATOL)
    for name in ("density", "pressure", "temperature", "speed", "direction"):
        assert np.isfinite(deck.look_up(name, HBE))


def test_mair_200_tabular_rho_matches_lookup():
    deck = _weather_deck()
    vehicle, env = _ready(mair=200, weather_deck=deck)
    env.execute(vehicle, _ctx())
    store = vehicle.store
    rho = deck.look_up("density", HBE)
    press = deck.look_up("pressure", HBE)
    tempc = deck.look_up("temperature", HBE)
    tempk = tempc + 273.16
    vsound = sqrt(1.4 * R * tempk)
    dvba = float(np.linalg.norm(VBEL))
    np.testing.assert_allclose(rho, RHO_7000, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("rho"), rho, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("press"), press, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("tempc"), tempc, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("tempk"), tempk, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("vsound"), vsound, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("grav"), gravity(HBE), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VAEL"), np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VBAL"), VBEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("dvba"), dvba, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("vmach"), abs(dvba / vsound), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        store.get("pdynmc"), 0.5 * rho * dvba * dvba, rtol=RTOL, atol=ATOL
    )


def test_mair_1_constant_wind_vael_north_negative():
    # Constant wind is mwind==1 → mair=1 (brief Step 1 "mair=2 constant wind"
    # names dvae=5, psiwdx=0 and C++ VAEL[0] = -dvw*cos).
    vehicle, env = _ready(mair=1, dvae=DVAE, psiwdx=PSIWDX)
    env.execute(vehicle, _ctx())
    store = vehicle.store
    want = _smoothed_constant_wind(DVAE, PSIWDX, 0.0, TWIND_DEFAULT, DT)
    np.testing.assert_allclose(store.get("VAEL"), want, rtol=RTOL, atol=ATOL)
    assert store.get("VAEL")[0] < 0.0
    np.testing.assert_allclose(
        store.get("VAEL")[0],
        -DVAE * cos(PSIWDX * RAD) * (DT / (2.0 * TWIND_DEFAULT)),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        store.get("VBAL"), VBEL - store.get("VAEL"), rtol=RTOL, atol=ATOL
    )
    rho, press, tempk = atmosphere76(HBE)
    np.testing.assert_allclose(store.get("rho"), rho, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("press"), press, rtol=RTOL, atol=ATOL)


def test_mair_2_tabular_wind_smoothes_vael():
    deck = _weather_deck()
    vehicle, env = _ready(mair=2, weather_deck=deck)
    env.execute(vehicle, _ctx())
    store = vehicle.store
    dvw = deck.look_up("speed", HBE)
    psiwdx = deck.look_up("direction", HBE)
    want = _smoothed_constant_wind(dvw, psiwdx, 0.0, TWIND_DEFAULT, DT)
    np.testing.assert_allclose(store.get("VAEL"), want, rtol=RTOL, atol=ATOL)
    assert store.get("VAEL")[0] < 0.0
    np.testing.assert_allclose(
        store.get("VBAL"), VBEL - store.get("VAEL"), rtol=RTOL, atol=ATOL
    )


def test_nmonte0_markov_values_zeroed_after_dryden_gauss():
    # C++ markov_noise always draws gauss then, when nmonte==0, stores 0.
    # JSONC keeps MARKOV sigmas (randt=0.0005, ...); they must not stay live.
    seed(12345)
    deck = _weather_deck()
    vehicle, env = _ready(
        mair=212,
        weather_deck=deck,
        kinematics=True,
        twind=1.0,
        turb_length=TURB_LENGTH,
        turb_sigma=TURB_SIGMA,
    )
    store = vehicle.store
    for name, sigma in (
        ("randal", 2.0),
        ("randt", 0.0005),
        ("randp", 0.001),
        ("randeh", 0.0002),
    ):
        store.define(Field(name, sigma, "real", "data", "sensor"))
    env.execute(vehicle, _ctx())
    for name in ("randal", "randt", "randp", "randeh"):
        assert store.get(name) == 0.0
    atol = max(1e-6, 5e-6 * abs(GOLDEN_T0_VAEL3))
    np.testing.assert_allclose(
        store.get("VAEL")[2], GOLDEN_T0_VAEL3, rtol=CSV_RTOL, atol=atol
    )


def test_mair_212_first_tau_matches_harvested_vael3():
    seed(12345)
    deck = _weather_deck()
    vehicle, env = _ready(
        mair=212,
        weather_deck=deck,
        kinematics=True,
        twind=1.0,
        turb_length=TURB_LENGTH,
        turb_sigma=TURB_SIGMA,
    )
    env.execute(vehicle, _ctx())
    store = vehicle.store
    atol = max(1e-6, 5e-6 * abs(GOLDEN_T0_VAEL3))
    np.testing.assert_allclose(
        store.get("tau"), GOLDEN_T0_VAEL3, rtol=CSV_RTOL, atol=atol
    )
    np.testing.assert_allclose(
        store.get("VAEL")[2], GOLDEN_T0_VAEL3, rtol=CSV_RTOL, atol=atol
    )


def test_mair_212_gauss_value_0_finite_vael():
    deck = _weather_deck()
    vehicle, env = _ready(
        mair=212,
        weather_deck=deck,
        kinematics=True,
        gauss_value=0.0,
    )
    env.execute(vehicle, _ctx())
    store = vehicle.store
    vael = store.get("VAEL")
    assert np.all(np.isfinite(vael))
    np.testing.assert_allclose(
        store.get("rho"), deck.look_up("density", HBE), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        store.get("VBAL"), VBEL - vael, rtol=RTOL, atol=ATOL
    )


def test_mturb_0_does_not_add_dryden():
    deck = _weather_deck()
    vehicle, env = _ready(mair=202, weather_deck=deck, gauss_value=1.0)
    env.execute(vehicle, _ctx())
    store = vehicle.store
    dvw = deck.look_up("speed", HBE)
    psiwdx = deck.look_up("direction", HBE)
    want = _smoothed_constant_wind(dvw, psiwdx, 0.0, TWIND_DEFAULT, DT)
    np.testing.assert_allclose(store.get("VAEL"), want, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("tau"), 0.0, rtol=RTOL, atol=ATOL)


def test_dryden_overwrites_stored_gauss_value_with_rand():
    seed(12345)
    deck = _weather_deck()
    vehicle, env = _ready(
        mair=210,
        weather_deck=deck,
        kinematics=True,
        gauss_value=1.0,
    )
    env.execute(vehicle, _ctx())
    assert np.any(np.abs(vehicle.store.get("VAEL")) > 0.0)
    assert vehicle.store.get("tau") != pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert vehicle.store.get("gauss_value") != pytest.approx(1.0, rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(
        vehicle.store.get("VBAL"),
        VBEL - vehicle.store.get("VAEL"),
        rtol=RTOL,
        atol=ATOL,
    )


def test_mwind_0_vael_zero_without_turbulence():
    vehicle, env = _ready(mair=0)
    env.execute(vehicle, _ctx())
    np.testing.assert_allclose(vehicle.store.get("VAEL"), np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("VBAL"), VBEL, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("mair", [100, 20, 3, 300, 111, 400])
def test_unknown_matmo_mturb_mwind_digits_raise(mair):
    deck = _weather_deck()
    vehicle, env = _ready(mair=mair, weather_deck=deck, kinematics=True)
    with pytest.raises(ValueError):
        env.execute(vehicle, _ctx())
