from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import R
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.eom.flat6 import Flat6Environment
from cadac.kernel.state import Field, StateStore


def _vehicle_with_newton_state(hbe=1000.0, vbel=(180.0, 0.0, 0.0), mwind=0):
    s = StateStore()
    vehicle = SimpleNamespace(store=s)
    env = Flat6Environment()
    env.define(vehicle)
    s.define(Field("hbe", 0.0, "real", "out", "newton"))
    s.define(Field("VBEL", (0.0, 0.0, 0.0), "vec", "state", "newton"))
    s.set("hbe", hbe)
    s.set("VBEL", np.array(vbel, dtype=float))
    s.set("mwind", mwind)
    return vehicle, env


def test_name_is_environment():
    assert Flat6Environment().name == "environment"


def test_env_from_hbe_and_vbel():
    vehicle, env = _vehicle_with_newton_state(hbe=1000.0, vbel=(180.0, 0.0, 0.0))
    env.execute(vehicle, None)
    s = vehicle.store
    rho, press, tempk = atmosphere76(1000.0)
    vsound = (1.4 * R * tempk) ** 0.5
    dvba = 180.0
    np.testing.assert_allclose(s.get("rho"), rho)
    np.testing.assert_allclose(s.get("press"), press)
    np.testing.assert_allclose(s.get("grav"), gravity(1000.0))
    np.testing.assert_allclose(s.get("tempk"), tempk)
    np.testing.assert_allclose(s.get("vsound"), vsound)
    np.testing.assert_allclose(s.get("dvba"), dvba)
    np.testing.assert_allclose(s.get("vmach"), abs(dvba / vsound))
    np.testing.assert_allclose(s.get("pdynmc"), 0.5 * rho * dvba**2)
    np.testing.assert_allclose(s.get("VAEL"), np.zeros(3))
    np.testing.assert_allclose(s.get("VBAL"), s.get("VBEL"))


def test_environment_does_not_define_newton_or_plane_fields():
    s = StateStore()
    env = Flat6Environment()
    env.define(SimpleNamespace(store=s))
    assert "hbe" not in s.names()
    assert "VBEL" not in s.names()
    assert "mach" not in s.names()
    for name in ("mfreeze", "mguid", "trcode"):
        assert name not in s.names()
    for name in ("grav", "rho", "pdynmc", "vmach", "vsound", "press", "tempk", "dvba"):
        assert s.get(name) == 0.0
    assert s.get("mwind") == 0
    np.testing.assert_allclose(s.get("VAEL"), np.zeros(3))
    np.testing.assert_allclose(s.get("VBAL"), np.zeros(3))


def test_unknown_mwind_raises():
    """mwind 1/2 are supported; other nonzero values still raise."""
    vehicle, env = _vehicle_with_newton_state(mwind=3)
    with pytest.raises(ValueError, match="mwind"):
        env.execute(vehicle, None)
