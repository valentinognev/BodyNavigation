import numpy as np

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.sraam6.forces import Sraam6Forces

RTOL = 1e-12
ATOL = 1e-14

PDYNMC = 1e4
REFA = 0.01824
REFL = 0.1524
CA = 0.5
THRUST = 100.0
FPB = (1.0, 2.0, 3.0)
FMPB = (4.0, 5.0, 6.0)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(0.0, 0.001, 0.0, 0.0, None, 0)


def _plant_aero_prop(store, *, ca=CA, cy=0.0, cn=0.0, cll=0.0, clm=0.0, cln=0.0, thrust=THRUST):
    store.define(Field("pdynmc", PDYNMC, "real", "out", "environment"))
    store.define(Field("thrust", thrust, "real", "out", "propulsion"))
    store.define(Field("refl", REFL, "real", "init", "aerodynamics"))
    store.define(Field("refa", REFA, "real", "init", "aerodynamics"))
    store.define(Field("ca", ca, "real", "out", "aerodynamics"))
    store.define(Field("cy", cy, "real", "out", "aerodynamics"))
    store.define(Field("cn", cn, "real", "out", "aerodynamics"))
    store.define(Field("cll", cll, "real", "out", "aerodynamics"))
    store.define(Field("clm", clm, "real", "out", "aerodynamics"))
    store.define(Field("cln", cln, "real", "out", "aerodynamics"))


def _ready(*, mtvc=None, fpb=None, fmpb=None):
    vehicle = _Vehicle()
    forces = Sraam6Forces()
    forces.define(vehicle)
    _plant_aero_prop(vehicle.store)
    if mtvc is not None:
        vehicle.store.define(Field("mtvc", mtvc, "int", "data", "tvc"))
    if fpb is not None:
        vehicle.store.define(Field("FPB", fpb, "vec", "out", "tvc"))
    if fmpb is not None:
        vehicle.store.define(Field("FMPB", fmpb, "vec", "out", "tvc"))
    return vehicle, forces


def test_absent_mtvc_adds_thrust_to_axial_force():
    vehicle, forces = _ready()
    forces.execute(vehicle, _ctx())
    want0 = -PDYNMC * REFA * CA + THRUST
    np.testing.assert_allclose(vehicle.store.get("FAPB")[0], want0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("FAPB")[1:], 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("FMB"), np.zeros(3), rtol=RTOL, atol=ATOL)


def test_mtvc2_adds_fpb_not_raw_thrust():
    vehicle, forces = _ready(mtvc=2, fpb=FPB, fmpb=FMPB)
    forces.execute(vehicle, _ctx())
    aero = np.array([-PDYNMC * REFA * CA, 0.0, 0.0])
    fapb = vehicle.store.get("FAPB")
    np.testing.assert_allclose(fapb, aero + np.array(FPB), rtol=RTOL, atol=ATOL)
    assert not np.isclose(fapb[0], aero[0] + THRUST, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("FMB"), np.array(FMPB), rtol=RTOL, atol=ATOL)


def test_mtvc0_adds_thrust_not_fpb():
    vehicle, forces = _ready(mtvc=0, fpb=FPB, fmpb=FMPB)
    forces.execute(vehicle, _ctx())
    want0 = -PDYNMC * REFA * CA + THRUST
    np.testing.assert_allclose(vehicle.store.get("FAPB")[0], want0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("FAPB")[1:], 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("FMB"), np.zeros(3), rtol=RTOL, atol=ATOL)


def test_does_not_write_fspb():
    vehicle, forces = _ready()
    sentinel = np.array([9.0, 8.0, 7.0])
    vehicle.store.define(Field("FSPB", sentinel, "vec", "out", "newton"))
    forces.execute(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("FSPB"), sentinel)
    assert Sraam6Forces().name == "forces"
    defined = StateStore()
    dummy = _Vehicle()
    dummy.store = defined
    Sraam6Forces().define(dummy)
    assert list(defined.names()) == ["FAPB", "FMB"]
