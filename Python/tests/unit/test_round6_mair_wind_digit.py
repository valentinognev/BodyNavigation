"""Task 37: HYPER6 mair wind digit 1 (constant) and digit 2 (shear)."""

from math import cos, sin
from types import SimpleNamespace

import numpy as np

from cadac.constants import RAD
from cadac.env.us76 import atmosphere76
from cadac.eom.round6 import Round6Environment
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.wgs84 import cad_in_geo84

RTOL = 1e-12
ATOL = 1e-14

DT = 0.01
TWIND = 0.1
DVAE = 5.0
PSIWDX = 0.0
VAED3 = 0.0
ALT = 10000.0
VBED = np.array([1000.0, 0.0, 0.0], dtype=float)

# Shear profile: mid-band → linear interpolate
DVAEL = 5.0
WALTL = 0.0
DVAEH = 15.0
WALTH = 20000.0


def _ctx(int_step=DT):
    return SimpleNamespace(int_step=int_step)


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


def _vehicle(
    *,
    mair,
    dvae=DVAE,
    psiwdx=PSIWDX,
    vaed3=VAED3,
    twind=TWIND,
    alt=ALT,
    vbed=VBED,
    dvael=DVAEL,
    waltl=WALTL,
    dvaeh=DVAEH,
    walth=WALTH,
):
    s = StateStore()
    # HYPER6 uses family None; wind digit 2 is shear (not ROCKET6 tabular).
    vehicle = SimpleNamespace(store=s)
    env = Round6Environment()
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
    s.set("VBED", np.array(vbed, dtype=float))
    s.set("mair", mair)
    s.set("dvae", dvae)
    s.set("psiwdx", psiwdx)
    s.set("vaed3", vaed3)
    s.set("twind", twind)
    s.set("dvael", dvael)
    s.set("waltl", waltl)
    s.set("dvaeh", dvaeh)
    s.set("walth", walth)
    return vehicle, env


def test_round6_mair_wind_digit_1():
    """mair=101 (matmo=1, mturb=0, mwind=1): NASA atmos + constant wind from dvae."""
    vehicle, env = _vehicle(mair=101)
    env.execute(vehicle, _ctx())
    store = vehicle.store

    want = _smoothed_vaed(DVAE, PSIWDX, VAED3, TWIND, DT)
    np.testing.assert_allclose(store.get("VAED"), want, rtol=RTOL, atol=ATOL)
    assert store.get("VAED")[0] < 0.0

    vbad = VBED - store.get("VAED")
    dvba = float(np.linalg.norm(vbad))
    np.testing.assert_allclose(store.get("dvba"), dvba, rtol=RTOL, atol=ATOL)

    # matmo=1 → us76_nasa2002 at 10 km (same golden as test_mair_100_nasa_at_10_km)
    np.testing.assert_allclose(store.get("rho"), 0.41351069327108436, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("press"), 26499.86774251078, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("tempk"), 223.25169383345863, rtol=RTOL, atol=ATOL)


def test_round6_mair_shear_digit_2():
    """mair=2 (matmo=0, mturb=0, mwind=2): HYPER6 shear wind, no weather deck."""
    vehicle, env = _vehicle(mair=2)
    env.execute(vehicle, _ctx())
    store = vehicle.store

    dvw = DVAEL + (DVAEH - DVAEL) * (ALT - WALTL) / (WALTH - WALTL)
    want = _smoothed_vaed(dvw, PSIWDX, VAED3, TWIND, DT)
    np.testing.assert_allclose(store.get("VAED"), want, rtol=RTOL, atol=ATOL)
    assert abs(dvw - 10.0) < 1e-12
    assert store.get("VAED")[0] < 0.0

    vbad = VBED - store.get("VAED")
    dvba = float(np.linalg.norm(vbad))
    np.testing.assert_allclose(store.get("dvba"), dvba, rtol=RTOL, atol=ATOL)

    rho, press, tempk = atmosphere76(ALT)
    np.testing.assert_allclose(store.get("rho"), rho, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("press"), press, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("tempk"), tempk, rtol=RTOL, atol=ATOL)
