from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import R
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.eom.rotor import RotorEnvironment
from cadac.kernel.state import Field, StateStore


def _vehicle(hbe=1000.0, vbel=(16.6, 0.0, 0.0), mwind=0):
    s = StateStore()
    vehicle = SimpleNamespace(store=s)
    env = RotorEnvironment()
    env.define(vehicle)
    s.define(Field("hbe", hbe, "real", "out", "trajectory"))
    s.define(Field("VBEL", vbel, "vec", "diag", "trajectory"))
    s.set("mwind", mwind)
    return vehicle, env


def test_name_is_environment():
    assert RotorEnvironment().name == "environment"


def test_environment_does_not_define_trajectory_fields():
    s = StateStore()
    RotorEnvironment().define(SimpleNamespace(store=s))
    for name in ("hbe", "VBEL", "dvbe", "alt", "SBII"):
        assert name not in s.names()
    assert s.get("mwind") == 0


def test_mwind0_us76_at_hbe_1000():
    vehicle, env = _vehicle()
    env.execute(vehicle, None)
    s = vehicle.store
    rho, press, tempk = atmosphere76(1000.0)
    vsound = (1.4 * R * tempk) ** 0.5
    dvba = 16.6
    np.testing.assert_allclose(s.get("rho"), rho, rtol=1e-12)
    np.testing.assert_allclose(s.get("press"), press, rtol=1e-12)
    np.testing.assert_allclose(s.get("tempk"), tempk, rtol=1e-12)
    np.testing.assert_allclose(s.get("vsound"), vsound, rtol=1e-12)
    np.testing.assert_allclose(s.get("dvba"), dvba, rtol=1e-12)
    np.testing.assert_allclose(s.get("vmach"), abs(dvba / vsound), rtol=1e-12)
    np.testing.assert_allclose(s.get("pdynmc"), 0.5 * rho * dvba**2, rtol=1e-12)
    np.testing.assert_allclose(s.get("grav"), gravity(1000.0), rtol=1e-12)
    np.testing.assert_allclose(s.get("VAEL"), np.zeros(3), rtol=1e-12)
    np.testing.assert_allclose(s.get("VBAL"), np.array([16.6, 0.0, 0.0]), rtol=1e-12)


def test_mwind_1_raises():
    vehicle, env = _vehicle(mwind=1)
    with pytest.raises(ValueError, match="mwind"):
        env.execute(vehicle, None)
