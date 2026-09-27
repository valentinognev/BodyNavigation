import math
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import R, RAD
from cadac.env.us76 import atmosphere76
from cadac.eom.round6 import Round6Environment
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.wgs84 import cad_grav84, cad_in_geo84
from cadac.stoch import dryden_white, gauss, seed
from cadac.tables.lookup import Datadeck, Table

# C++ srand(0); def_ins 18 gauss + init_ins 9 gauss; then each step
# markov_noise 15 gauss (nmonte==0 still draws then stores 0) before Dryden.
_ROCKET6_INS_GAUSS = 27
_ROCKET6_MARKOV_COUNT = 15

RTOL = 1e-12
ATOL = 1e-14


def _weather_speed_direction_deck():
    speed = Table(
        name="speed",
        dim=1,
        x1=np.array([0.0, 20000.0]),
        x2=None,
        x3=None,
        values=np.array([0.0, 20.0]),
    )
    direction = Table(
        name="direction",
        dim=1,
        x1=np.array([0.0, 20000.0]),
        x2=None,
        x3=None,
        values=np.array([0.0, 180.0]),
    )
    return Datadeck.from_tables([speed, direction])


def _vehicle_with_newton_state(
    alt=10000.0,
    time=0.0,
    vbed=(1000.0, 0.0, 0.0),
    mair=0,
    lon=10.0 * RAD,
    lat=10.0 * RAD,
    weather_deck=None,
):
    s = StateStore()
    vehicle = SimpleNamespace(store=s)
    if weather_deck is None:
        env = Round6Environment()
    else:
        env = Round6Environment(weather_deck=weather_deck)
    env.define(vehicle)
    s.define(Field("time", 0.0, "real", "exec", "kinematics"))
    s.define(Field("alt", 0.0, "real", "out", "newton"))
    s.define(Field("SBII", (0.0, 0.0, 0.0), "vec", "state", "newton"))
    s.define(Field("VBED", (0.0, 0.0, 0.0), "vec", "state", "newton"))
    sbii = cad_in_geo84(lon, lat, alt, time)
    s.set("time", time)
    s.set("alt", alt)
    s.set("SBII", sbii)
    s.set("VBED", np.array(vbed, dtype=float))
    s.set("mair", mair)
    return vehicle, env


def test_name_is_environment():
    assert Round6Environment().name == "environment"


def test_environment_does_not_define_newton_kinematics_or_hyper_fields():
    s = StateStore()
    env = Round6Environment()
    env.define(SimpleNamespace(store=s))
    for name in ("alt", "SBII", "VBED", "time", "mach", "mfreeze", "trcode", "mguid"):
        assert name not in s.names()
    assert s.get("mair") == 0
    for name in ("press", "rho", "vsound", "vmach", "pdynmc", "tempk", "grav", "dvba"):
        assert s.get(name) == 0.0
    np.testing.assert_allclose(s.get("GRAVG"), np.zeros(3))
    np.testing.assert_allclose(s.get("VAED"), np.zeros(3))


def test_mair0_us76_grav84_vmach_finite():
    vehicle, env = _vehicle_with_newton_state()
    env.execute(vehicle, None)
    s = vehicle.store
    rho, press, tempk = atmosphere76(10000.0)
    vsound = (1.4 * R * tempk) ** 0.5
    dvba = 1000.0
    gravg = cad_grav84(s.get("SBII"), s.get("time"))
    np.testing.assert_allclose(s.get("rho"), rho, rtol=1e-12)
    np.testing.assert_allclose(s.get("press"), press, rtol=1e-12)
    np.testing.assert_allclose(s.get("tempk"), tempk, rtol=1e-12)
    np.testing.assert_allclose(s.get("vsound"), vsound, rtol=1e-12)
    np.testing.assert_allclose(s.get("dvba"), dvba, rtol=1e-12)
    np.testing.assert_allclose(s.get("vmach"), abs(dvba / vsound), rtol=1e-12)
    assert np.isfinite(s.get("vmach"))
    np.testing.assert_allclose(s.get("pdynmc"), 0.5 * rho * dvba**2, rtol=1e-12)
    np.testing.assert_allclose(s.get("GRAVG"), gravg, rtol=1e-12)
    np.testing.assert_allclose(s.get("grav"), np.linalg.norm(gravg), rtol=1e-12)
    np.testing.assert_allclose(s.get("VAED"), np.zeros(3), rtol=1e-12)


def test_mair_100_nasa_at_10_km():
    """mair=100 at 10 km is in range for us76_nasa2002 (not unknown mair)."""
    vehicle, env = _vehicle_with_newton_state(mair=100, alt=10000.0)
    env.execute(vehicle, SimpleNamespace(int_step=0.01))
    store = vehicle.store
    np.testing.assert_allclose(store.get("rho"), 0.41351069327108436, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(store.get("press"), 26499.86774251078, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(store.get("tempk"), 223.25169383345863, rtol=1e-12, atol=0.0)


def test_other_nonzero_mair_raises():
    for mair in (1, 10, 11, 101):
        vehicle, env = _vehicle_with_newton_state(mair=mair)
        with pytest.raises(ValueError):
            env.execute(vehicle, None)


def test_execute_skips_trcode_mfreeze_when_absent():
    vehicle, env = _vehicle_with_newton_state()
    env.execute(vehicle, None)
    s = vehicle.store
    assert "trcode" not in s.names()
    assert "mfreeze" not in s.names()
    assert np.isfinite(s.get("vmach"))


def test_initialize_sets_dvba_from_dvbe():
    s = StateStore()
    vehicle = SimpleNamespace(store=s)
    env = Round6Environment()
    env.define(vehicle)
    s.define(Field("dvbe", 0.0, "real", "out", "newton"))
    s.set("dvbe", 1234.5)
    env.initialize(vehicle, None)
    np.testing.assert_allclose(s.get("dvba"), 1234.5, rtol=RTOL, atol=ATOL)


def test_mair_12_without_weather_deck_raises():
    vehicle, env = _vehicle_with_newton_state(mair=12)
    ctx = SimpleNamespace(int_step=0.01)
    with pytest.raises(ValueError, match="weather"):
        env.execute(vehicle, ctx)


def test_mair_12_tabular_wind_zero_dryden():
    alt = 10000.0
    twind = 1.0
    dt = 0.01
    deck = _weather_speed_direction_deck()
    vehicle, env = _vehicle_with_newton_state(mair=12, weather_deck=deck)
    s = vehicle.store
    s.define(Field("TBD", np.eye(3), "mat", "out", "kinematics"))
    s.define(Field("alppx", 0.0, "real", "out", "kinematics"))
    s.define(Field("phipx", 0.0, "real", "out", "kinematics"))
    s.set("twind", twind)
    s.set("turb_sigma", 0.0)
    s.set("turb_length", 100.0)
    s.set("alppx", 0.0)
    s.set("phipx", 0.0)
    s.set("TBD", np.eye(3))
    s.set("dvba", float(np.linalg.norm(s.get("VBED"))))
    ctx = SimpleNamespace(int_step=dt)
    env.execute(vehicle, ctx)

    dvw = 10.0
    psiwdx = 90.0
    vaed3 = 0.0
    vaeds = np.zeros(3)
    vaedsd = np.zeros(3)
    vaed_raw = np.array(
        [
            -dvw * math.cos(psiwdx * RAD),
            -dvw * math.sin(psiwdx * RAD),
            vaed3,
        ],
        dtype=float,
    )
    vaedsd_new = (vaed_raw - vaeds) * (1.0 / twind)
    vaeds = integrate(vaedsd_new, vaedsd, vaeds, dt)
    np.testing.assert_allclose(s.get("VAED"), vaeds, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(s.get("tau"), 0.0, rtol=RTOL, atol=ATOL)
    vbed = s.get("VBED")
    dvba = float(np.linalg.norm(vbed - vaeds))
    np.testing.assert_allclose(s.get("dvba"), dvba, rtol=RTOL, atol=ATOL)


def test_mair_12_dryden_gauss_value_matches_cpp_srand0_stream():
    alt = 10000.0
    dt = 0.001
    deck = _weather_speed_direction_deck()
    vehicle, env = _vehicle_with_newton_state(alt=alt, mair=12, weather_deck=deck)
    s = vehicle.store
    s.define(Field("TBD", np.eye(3), "mat", "out", "kinematics"))
    s.define(Field("alppx", 0.0, "real", "out", "kinematics"))
    s.define(Field("phipx", 0.0, "real", "out", "kinematics"))
    s.set("twind", 1.0)
    s.set("turb_sigma", 0.5)
    s.set("turb_length", 100.0)
    s.set("alppx", 0.0)
    s.set("phipx", 0.0)
    s.set("TBD", np.eye(3))
    s.set("dvba", float(np.linalg.norm(s.get("VBED"))))
    seed(0)
    for _ in range(_ROCKET6_INS_GAUSS + _ROCKET6_MARKOV_COUNT):
        gauss(0.0, 1.0)
    want = dryden_white(dt)
    seed(0)
    env.execute(vehicle, SimpleNamespace(int_step=dt))
    np.testing.assert_allclose(s.get("gauss_value"), want, rtol=RTOL, atol=ATOL)
    assert abs(s.get("gauss_value")) > 1e-12
