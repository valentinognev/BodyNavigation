import numpy as np

from cadac.constants import DEG
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.sraam6.euler import Sraam6Euler

RTOL = 1e-12
ATOL = 1e-14


def test_zero_moment_holds_roll_rate():
    vehicle = type("V", (), {"store": StateStore()})()
    euler = Sraam6Euler()
    euler.define(vehicle)
    store = vehicle.store
    store.define(Field("FMB", (0.0, 0.0, 0.0), "vec", "out", "forces"))
    store.define(Field("ai11", 0.308, "real", "out", "propulsion"))
    store.define(Field("ai33", 59.80, "real", "out", "propulsion"))
    store.set("pp", 0.1)
    euler.execute(vehicle, SimContext(0.0, 0.001, 0.0, 0.0, None, 0))
    np.testing.assert_allclose(store.get("pp"), 0.1, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ppx"), 0.1 * DEG, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("WBEB"), [0.1, 0.0, 0.0], rtol=RTOL, atol=ATOL)


def test_roll_moment_integrates_pp():
    vehicle = type("V", (), {"store": StateStore()})()
    euler = Sraam6Euler()
    euler.define(vehicle)
    store = vehicle.store
    store.define(Field("FMB", (0.308, 0.0, 0.0), "vec", "out", "forces"))
    store.define(Field("ai11", 0.308, "real", "out", "propulsion"))
    store.define(Field("ai33", 59.80, "real", "out", "propulsion"))
    ppd_new = 0.308 / 0.308
    expected = integrate(ppd_new, 0.0, 0.0, 0.001)
    euler.execute(vehicle, SimContext(0.0, 0.001, 0.0, 0.0, None, 0))
    np.testing.assert_allclose(store.get("pp"), expected, rtol=RTOL, atol=ATOL)
