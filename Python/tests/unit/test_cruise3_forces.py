import numpy as np

from cadac.constants import RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.cruise3.forces import Cruise3Forces


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.01,
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
    forces = Cruise3Forces()
    forces.define(vehicle)
    store = vehicle.store
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("area", area, "real", "data", "aerodynamics"))
    store.define(Field("cd", cd, "real", "out", "aerodynamics"))
    store.define(Field("cl", cl, "real", "out", "aerodynamics"))
    store.define(Field("thrust", thrust, "real", "out", "propulsion"))
    store.define(Field("mass", mass, "real", "out", "propulsion"))
    store.define(Field("alphax", alphax, "real", "data", "aerodynamics"))
    store.set("phimvx", phimvx)
    return vehicle, forces


def test_name_is_forces():
    assert Cruise3Forces().name == "forces"


def test_define_registers_fspv_and_phimvx():
    vehicle = _Vehicle()
    Cruise3Forces().define(vehicle)
    store = vehicle.store
    np.testing.assert_array_equal(store.get("FSPV"), np.zeros(3))
    assert store.get("phimvx") == 0.0


def test_define_skips_existing_fspv():
    vehicle = _Vehicle()
    store = vehicle.store
    store.define(Field("FSPV", (1.0, 2.0, 3.0), "vec", "out", "forces"))
    store.define(Field("phimvx", 15.0, "real", "data", "aerodynamics"))
    Cruise3Forces().define(vehicle)
    np.testing.assert_array_equal(store.get("FSPV"), np.array([1.0, 2.0, 3.0]))
    assert store.get("phimvx") == 15.0


def test_fspv_level():
    pdynmc, area, cd, cl, thrust, mass, alphax, phimvx = (
        28410.0,
        557.42,
        0.05,
        0.2,
        239241.0,
        136077.0,
        7.0,
        0.0,
    )
    vehicle, forces = _ready_forces(
        pdynmc, area, cd, cl, thrust, mass, alphax, phimvx
    )
    expected = _expected_fspv(pdynmc, area, cd, cl, thrust, mass, alphax, phimvx)

    forces.execute(vehicle, _ctx())

    np.testing.assert_array_equal(vehicle.store.get("FSPV"), expected)


def test_fspv_banked():
    pdynmc, area, cd, cl, thrust, mass, alphax, phimvx = (
        28410.0,
        557.42,
        0.05,
        0.2,
        239241.0,
        136077.0,
        7.0,
        90.0,
    )
    vehicle, forces = _ready_forces(
        pdynmc, area, cd, cl, thrust, mass, alphax, phimvx
    )
    expected = _expected_fspv(pdynmc, area, cd, cl, thrust, mass, alphax, phimvx)

    forces.execute(vehicle, _ctx())

    np.testing.assert_array_equal(vehicle.store.get("FSPV"), expected)
    assert vehicle.store.get("FSPV")[1] != 0.0
