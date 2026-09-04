from types import SimpleNamespace

import numpy as np

from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.eom.flat3 import Flat3Environment
from cadac.kernel.state import Field, StateStore


def test_name_is_environment():
    assert Flat3Environment().name == "environment"


def test_env_from_sbel():
    s = StateStore()
    vehicle = SimpleNamespace(store=s)
    env = Flat3Environment()
    env.define(vehicle)
    s.define(Field("SBEL", (0.0, 0.0, 0.0), "vec", "state", "newton"))
    s.define(Field("dvbe", 0.0, "real", "init/out", "newton"))
    s.set("SBEL", np.array([0.0, 0.0, -3500.0]))
    s.set("dvbe", 200.0)
    env.execute(vehicle, None)
    rho, press, tempk = atmosphere76(3500.0)
    vsound = (1.4 * 287.053 * tempk) ** 0.5
    alt_ok = "alt" in s.names() and abs(s.get("alt") - 3500.0) < 1e-12
    assert alt_ok or abs((-s.get("SBEL")[2]) - 3500) < 1e-12
    assert abs(s.get("grav") - gravity(3500.0)) < 1e-12
    assert abs(s.get("rho") - rho) < 1e-12
    assert abs(s.get("mach") - abs(200.0 / vsound)) < 1e-12
    assert abs(s.get("vsound") - vsound) < 1e-12
    assert abs(s.get("press") - press) < 1e-12
    assert abs(s.get("pdynmc") - 0.5 * rho * 200.0**2) < 1e-12


def test_environment_does_not_define_alt():
    s = StateStore()
    env = Flat3Environment()
    env.define(SimpleNamespace(store=s))
    assert "alt" not in s.names()
    assert "SBEL" not in s.names()
    for name in ("grav", "rho", "pdynmc", "mach", "vsound", "press"):
        assert s.get(name) == 0.0
