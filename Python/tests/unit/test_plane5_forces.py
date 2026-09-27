import numpy as np

from cadac.constants import RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat3.falcon5.forces import Plane5Forces


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.05,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _expected_fspv(pdynmc, area, cd, cl, thrust, mass, alphax, phimvx):
    alpha = alphax * RAD
    phimv = phimvx * RAD
    f1 = (-pdynmc * area * cd + thrust * np.cos(alpha)) / mass
    f2 = np.sin(phimv) * (pdynmc * area * cl + thrust * np.sin(alpha)) / mass
    f3 = -np.cos(phimv) * (pdynmc * area * cl + thrust * np.sin(alpha)) / mass
    return np.array([f1, f2, f3])


def _ready_forces(pdynmc, area, cd, cl, thrust, mass, alphax, phimvx):
    vehicle = _Vehicle()
    forces = Plane5Forces()
    forces.define(vehicle)
    store = vehicle.store
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("cl", cl, "real", "out", "aerodynamics"))
    store.define(Field("cd", cd, "real", "out", "aerodynamics"))
    store.define(Field("area", area, "real", "data", "aerodynamics"))
    store.define(Field("thrust", thrust, "real", "out", "propulsion"))
    store.define(Field("mass", mass, "real", "out", "propulsion"))
    store.define(Field("alphax", alphax, "real", "out", "control"))
    store.define(Field("phimvx", phimvx, "real", "out", "control"))
    return vehicle, forces


# FALCON5 turning-to-IP scale (area, mass); pdynmc from propulsion unit tests
_PDYNMC = 17000.0
_AREA = 27.87
_CD = 0.05
_CL = 0.2
_THRUST = 20000.0
_MASS = 12701.0
_ALPHAX = 5.0


def test_name_is_forces():
    assert Plane5Forces().name == "forces"


def test_define_registers_fspv_only():
    vehicle = _Vehicle()
    Plane5Forces().define(vehicle)
    store = vehicle.store
    fspv = store.field("FSPV")
    np.testing.assert_array_equal(fspv.value, np.zeros(3))
    assert fspv.type == "vec"
    assert fspv.role == "out"
    assert fspv.module == "forces"
    assert fspv.outputs == ("plot",)
    assert "phimvx" not in store.names()
    assert "alphax" not in store.names()


def test_fspv_level():
    vehicle, forces = _ready_forces(
        _PDYNMC, _AREA, _CD, _CL, _THRUST, _MASS, _ALPHAX, 0.0
    )
    expected = _expected_fspv(
        _PDYNMC, _AREA, _CD, _CL, _THRUST, _MASS, _ALPHAX, 0.0
    )

    forces.execute(vehicle, _ctx())

    np.testing.assert_array_equal(vehicle.store.get("FSPV"), expected)
    assert vehicle.store.get("FSPV")[1] == 0.0


def test_fspv_banked():
    vehicle, forces = _ready_forces(
        _PDYNMC, _AREA, _CD, _CL, _THRUST, _MASS, _ALPHAX, 90.0
    )
    expected = _expected_fspv(
        _PDYNMC, _AREA, _CD, _CL, _THRUST, _MASS, _ALPHAX, 90.0
    )

    forces.execute(vehicle, _ctx())

    np.testing.assert_array_equal(vehicle.store.get("FSPV"), expected)
    assert vehicle.store.get("FSPV")[1] != 0.0


def test_fspv_nonzero_phimvx():
    phimvx = 30.0
    vehicle, forces = _ready_forces(
        _PDYNMC, _AREA, _CD, _CL, _THRUST, _MASS, _ALPHAX, phimvx
    )
    expected = _expected_fspv(
        _PDYNMC, _AREA, _CD, _CL, _THRUST, _MASS, _ALPHAX, phimvx
    )

    forces.execute(vehicle, _ctx())

    np.testing.assert_array_equal(vehicle.store.get("FSPV"), expected)
    fspv = vehicle.store.get("FSPV")
    assert fspv[1] != 0.0
    assert fspv[2] != 0.0
