import math

import numpy as np

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.sraam6.kinematics import Sraam6Kinematics

RTOL = 1e-12
ATOL = 1e-14
SMALL = 1e-7


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(sim_time=0.0, int_step=0.001):
    return SimContext(sim_time, int_step, 0.0, 0.0, None, 0)


def _ready(psiblx=0.0, thtblx=0.0, phiblx=0.0):
    vehicle = _Vehicle()
    kin = Sraam6Kinematics()
    kin.define(vehicle)
    store = vehicle.store
    store.set("psiblx", psiblx)
    store.set("thtblx", thtblx)
    store.set("phiblx", phiblx)
    return vehicle, kin


def _plant(store, *, vbeb, trcvel=10e-4, tralp=1.0, trcond=0):
    for name, value, ftype in (
        ("VBEB", tuple(vbeb), "vec"),
        ("trcvel", trcvel, "real"),
        ("tralp", tralp, "real"),
        ("trcond", trcond, "int"),
    ):
        store.define(Field(name, value, ftype, "data", "ext"))


def test_init_1v1_identity_tbl_and_finite_q0():
    vehicle, kin = _ready(psiblx=0.0, thtblx=0.0, phiblx=0.0)
    assert kin.name == "kinematics"
    kin.initialize(vehicle, _ctx())
    store = vehicle.store
    np.testing.assert_allclose(store.get("TBL"), np.eye(3), rtol=RTOL, atol=ATOL)
    assert np.isfinite(store.get("q0"))


def test_execute_zero_rates_orthonormal_zero_incidence():
    vehicle, kin = _ready()
    store = vehicle.store
    _plant(store, vbeb=(250.0, 0.0, 0.0))
    store.set("pp", 0.0)
    store.set("qq", 0.0)
    store.set("rr", 0.0)
    kin.initialize(vehicle, _ctx())
    kin.execute(vehicle, _ctx())
    np.testing.assert_allclose(store.get("etbl"), 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("alphax"), 0.0, rtol=RTOL, atol=ATOL)
    assert store.get("phip") == 0.0


def test_phip_uses_small_when_vbeb2_below_small():
    vehicle, kin = _ready()
    store = vehicle.store
    _plant(store, vbeb=(250.0, 1e-8, 1.0))
    store.set("pp", 0.0)
    store.set("qq", 0.0)
    store.set("rr", 0.0)
    kin.initialize(vehicle, _ctx())
    kin.execute(vehicle, _ctx())
    assert store.get("phip") == math.atan2(SMALL, 1.0)


def test_broken_quaternion_sets_trcond_1():
    vehicle, kin = _ready()
    store = vehicle.store
    _plant(store, vbeb=(250.0, 0.0, 0.0), trcvel=10e-4, trcond=0)
    store.set("pp", 0.0)
    store.set("qq", 0.0)
    store.set("rr", 0.0)
    kin.initialize(vehicle, _ctx())
    store.set("q0", 2.0)
    store.set("q1", 0.0)
    store.set("q2", 0.0)
    store.set("q3", 0.0)
    kin.execute(vehicle, _ctx())
    assert store.get("trcond") == 1
