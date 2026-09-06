from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import R, RAD
from cadac.env.us76 import atmosphere76
from cadac.eom.round6 import Round6Environment
from cadac.kernel.state import Field, StateStore
from cadac.math.wgs84 import cad_grav84, cad_in_geo84


def _vehicle_with_newton_state(
    alt=10000.0,
    time=0.0,
    vbed=(1000.0, 0.0, 0.0),
    mair=0,
    lon=10.0 * RAD,
    lat=10.0 * RAD,
):
    s = StateStore()
    vehicle = SimpleNamespace(store=s)
    env = Round6Environment()
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


def test_mair_100_raises():
    vehicle, env = _vehicle_with_newton_state(mair=100)
    with pytest.raises(ValueError):
        env.execute(vehicle, None)


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
