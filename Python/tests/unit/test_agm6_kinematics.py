import math
from types import SimpleNamespace

import numpy as np

from cadac.constants import RAD
from cadac.vehicles.agm6.kinematics import Agm6Kinematics
from cadac.kernel.state import Field, StateStore

RTOL = 1e-12
ATOL = 1e-14
SMALL = 1e-7
DVBA = 293.0
ALPHA0X = 3.0


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(int_step=0.001):
    return SimpleNamespace(int_step=int_step)


def _ready():
    vehicle = _Vehicle()
    kin = Agm6Kinematics()
    kin.define(vehicle)
    store = vehicle.store
    zeros3 = (0.0, 0.0, 0.0)
    store.define(Field("VBAL", zeros3, "vec", "out", "environment"))
    store.define(Field("VAEL", zeros3, "vec", "out", "environment"))
    store.define(Field("VBEB", zeros3, "vec", "state", "newton"))
    store.define(Field("dvba", 0.0, "real", "out", "environment"))
    store.define(Field("WBEB", zeros3, "vec", "state", "euler"))
    store.define(Field("tralp", 1.0, "real", "data", "aerodynamics"))
    store.define(Field("trcond", 0, "int", "init", "aerodynamics"))
    return vehicle, kin


def test_name_is_kinematics():
    assert Agm6Kinematics().name == "kinematics"


def test_incidence_uses_vbeb_minus_tbl_vael_not_vbal():
    vehicle, kin = _ready()
    store = vehicle.store
    kin.initialize(vehicle, None)
    salp = math.sin(ALPHA0X * RAD)
    calp = math.cos(ALPHA0X * RAD)
    vbeb = np.array([calp * DVBA, 0.0, salp * DVBA], dtype=float)
    store.set("VBEB", vbeb)
    store.set("VAEL", np.zeros(3))
    store.set("VBAL", np.array([DVBA, 0.0, 0.0], dtype=float))
    store.set("dvba", DVBA)
    store.set("WBEB", np.zeros(3))
    kin.execute(vehicle, _ctx())
    np.testing.assert_allclose(store.get("alppx"), ALPHA0X, rtol=RTOL, atol=1e-12)
    np.testing.assert_allclose(store.get("alphax"), ALPHA0X, rtol=RTOL, atol=1e-12)
    assert store.get("phip") == math.atan2(SMALL, salp * DVBA)


def test_incidence_uses_vaels_not_dryden_vael():
    vehicle, kin = _ready()
    store = vehicle.store
    store.define(Field("VAELS", (0.0, 0.0, 0.0), "vec", "state", "environment"))
    kin.initialize(vehicle, None)
    salp = math.sin(ALPHA0X * RAD)
    calp = math.cos(ALPHA0X * RAD)
    vbeb = np.array([calp * DVBA, 0.0, salp * DVBA], dtype=float)
    store.set("VBEB", vbeb)
    store.set("VAEL", np.array([10.0, 20.0, 30.0], dtype=float))
    store.set("VAELS", np.zeros(3))
    store.set("VBAL", np.array([DVBA, 0.0, 0.0], dtype=float))
    store.set("dvba", DVBA)
    store.set("WBEB", np.zeros(3))
    kin.execute(vehicle, _ctx())
    np.testing.assert_allclose(store.get("alppx"), ALPHA0X, rtol=RTOL, atol=1e-12)
    np.testing.assert_allclose(store.get("alphax"), ALPHA0X, rtol=RTOL, atol=1e-12)


def test_phip_uses_small_when_vbab2_below_small():
    vehicle, kin = _ready()
    store = vehicle.store
    kin.initialize(vehicle, None)
    store.set("VBEB", np.array([DVBA, 1e-8, 1.0], dtype=float))
    store.set("VAEL", np.zeros(3))
    store.set("VBAL", np.array([DVBA, 0.0, 0.0], dtype=float))
    store.set("dvba", DVBA)
    store.set("WBEB", np.zeros(3))
    kin.execute(vehicle, _ctx())
    assert store.get("phip") == math.atan2(SMALL, 1.0)
