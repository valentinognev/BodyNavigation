import math
from types import SimpleNamespace

import numpy as np

from cadac.constants import R
from cadac.env.us76 import atmosphere76
from cadac.eom.flat3 import Flat3AircraftEnvironment, Flat3AircraftNewton
from cadac.kernel.state import Field, StateStore

RTOL = 1e-12
ATOL = 1e-14


def _target_after_init():
    store = StateStore()
    vehicle = SimpleNamespace(store=store)
    newton = Flat3AircraftNewton()
    env = Flat3AircraftEnvironment()
    newton.define(vehicle)
    env.define(vehicle)
    store.define(Field("FSPA", (0.0, 0.0, 0.0), "vec", "out", "forces"))
    store.define(Field("phiavout", 0.0, "real", "out", "control"))
    store.set("sael1", 10000.0)
    store.set("sael2", 500.0)
    store.set("sael3", -2000.0)
    store.set("psialx", 180.0)
    store.set("thtalx", 0.0)
    store.set("dvae", 250.0)
    newton.initialize(vehicle, None)
    return vehicle, newton, env


def test_init_1v1_target_sael_dvae_alt():
    vehicle, _, _ = _target_after_init()
    store = vehicle.store
    np.testing.assert_allclose(
        store.get("SAEL"), [10000.0, 500.0, -2000.0], rtol=RTOL, atol=ATOL
    )
    assert store.get("dvae") == 250.0
    assert store.get("alt") == 2000.0


def test_one_step_sael_changes_mach_from_dvae():
    vehicle, newton, env = _target_after_init()
    store = vehicle.store
    sael_before = store.get("SAEL").copy()
    ctx = SimpleNamespace(int_step=0.001)
    newton.execute(vehicle, ctx)
    env.execute(vehicle, ctx)
    assert not np.allclose(store.get("SAEL"), sael_before, rtol=RTOL, atol=ATOL)
    rho, press, tempk = atmosphere76(2000.0)
    vsound = math.sqrt(1.4 * R * tempk)
    np.testing.assert_allclose(
        store.get("mach"), abs(store.get("dvae") / vsound), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(store.get("rho"), rho, rtol=RTOL, atol=ATOL)
    assert "SBEL" not in store.names()
    assert "dvbe" not in store.names()
    assert "FSPV" not in store.names()
