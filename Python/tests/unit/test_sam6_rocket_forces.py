from pathlib import Path

import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.sam6.rocket import Sam6RocketForces

RTOL = 1e-12
ATOL = 1e-14
ZEROS3 = (0.0, 0.0, 0.0)
THRUST = 128600.0
CATGT = 0.0
PDYNMC = 0.0
AREA = 0.636
MASS = 6000.0
GRAV = 9.81
CYTGT = 0.0
CNTGT = 0.0

DEFINED = ("FSPA", "aax", "alx", "anx")
ROLES = {
    "FSPA": "out",
    "aax": "diag",
    "alx": "diag",
    "anx": "diag",
}
OUTPUTS = {
    "FSPA": (),
    "aax": (),
    "alx": ("com",),
    "anx": ("com",),
}
TYPES = {
    "FSPA": "vec",
    "aax": "real",
    "alx": "real",
    "anx": "real",
}
NOT_DEFINED = (
    "acc_longx",
    "thrust",
    "catgt",
    "pdynmc",
    "area",
    "mass",
    "grav",
    "cytgt",
    "cntgt",
    "time",
    "SAEL",
    "VAEL",
    "TAL",
    "alphax",
    "betax",
    "ancomx",
    "alcomx",
    "mprop",
    "FAPB",
    "FMB",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.001,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _fspa(thrust, catgt, pdynmc, area, mass, cytgt, cntgt):
    return (
        (thrust - catgt * pdynmc * area) / mass,
        (cytgt * pdynmc * area) / mass,
        (-cntgt * pdynmc * area) / mass,
    )


def _plant(
    store,
    *,
    thrust=THRUST,
    catgt=CATGT,
    pdynmc=PDYNMC,
    area=AREA,
    mass=MASS,
    grav=GRAV,
    cytgt=CYTGT,
    cntgt=CNTGT,
):
    store.define(Field("thrust", thrust, "real", "out", "propulsion", ("com",)))
    store.define(Field("catgt", catgt, "real", "out", "aerodynamics"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("area", area, "real", "data", "aerodynamics"))
    store.define(Field("mass", mass, "real", "out", "propulsion"))
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("cytgt", cytgt, "real", "out", "aerodynamics"))
    store.define(Field("cntgt", cntgt, "real", "out", "aerodynamics"))


def _ready(
    *,
    thrust=THRUST,
    catgt=CATGT,
    pdynmc=PDYNMC,
    area=AREA,
    mass=MASS,
    grav=GRAV,
    cytgt=CYTGT,
    cntgt=CNTGT,
):
    vehicle = _Vehicle()
    forces = Sam6RocketForces()
    forces.define(vehicle)
    forces.initialize(vehicle, _ctx())
    _plant(
        vehicle.store,
        thrust=thrust,
        catgt=catgt,
        pdynmc=pdynmc,
        area=area,
        mass=mass,
        grav=grav,
        cytgt=cytgt,
        cntgt=cntgt,
    )
    return vehicle, forces


def test_name_is_forces():
    assert Sam6RocketForces().name == "forces"


def test_define_registers_cpp_fields():
    vehicle = _Vehicle()
    Sam6RocketForces().define(vehicle)
    store = vehicle.store
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        assert field.module == "forces", name
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        assert field.type == TYPES[name], name
    np.testing.assert_allclose(store.get("FSPA"), ZEROS3, rtol=RTOL, atol=ATOL)
    assert store.get("FSPA").shape == (3,)
    assert _approx(store.get("aax"), 0.0)
    assert _approx(store.get("alx"), 0.0)
    assert _approx(store.get("anx"), 0.0)
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_initialize_is_pass():
    vehicle = _Vehicle()
    forces = Sam6RocketForces()
    forces.define(vehicle)
    vehicle.store.set("FSPA", (1.0, 2.0, 3.0))
    vehicle.store.set("aax", 4.0)
    vehicle.store.set("alx", 5.0)
    vehicle.store.set("anx", 6.0)
    before = {name: np.array(vehicle.store.get(name), copy=True) for name in DEFINED}
    assert forces.initialize(vehicle, _ctx()) is None
    np.testing.assert_allclose(
        vehicle.store.get("FSPA"), before["FSPA"], rtol=RTOL, atol=ATOL
    )
    for name in ("aax", "alx", "anx"):
        assert _approx(vehicle.store.get(name), float(before[name]))


def test_vacuum_boost_fspa_x_is_thrust_over_mass():
    vehicle, forces = _ready(
        thrust=THRUST, catgt=0.0, pdynmc=0.0, area=AREA, mass=MASS
    )
    forces.execute(vehicle, _ctx())
    want = THRUST / MASS
    assert vehicle.store.get("FSPA")[0] == pytest.approx(want, rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(
        vehicle.store.get("FSPA"), (want, 0.0, 0.0), rtol=RTOL, atol=ATOL
    )


def test_axial_aero_subtracts_from_thrust():
    catgt = 0.3
    pdynmc = 50000.0
    cytgt = 0.1
    cntgt = 0.2
    vehicle, forces = _ready(
        thrust=THRUST,
        catgt=catgt,
        pdynmc=pdynmc,
        area=AREA,
        mass=MASS,
        cytgt=cytgt,
        cntgt=cntgt,
    )
    forces.execute(vehicle, _ctx())
    want = _fspa(THRUST, catgt, pdynmc, AREA, MASS, cytgt, cntgt)
    np.testing.assert_allclose(
        vehicle.store.get("FSPA"), want, rtol=RTOL, atol=ATOL
    )


def test_diagnostics_are_fspa_over_grav():
    catgt = 0.3
    pdynmc = 50000.0
    cytgt = 0.1
    cntgt = 0.2
    vehicle, forces = _ready(
        thrust=THRUST,
        catgt=catgt,
        pdynmc=pdynmc,
        area=AREA,
        mass=MASS,
        grav=GRAV,
        cytgt=cytgt,
        cntgt=cntgt,
    )
    forces.execute(vehicle, _ctx())
    fspa = vehicle.store.get("FSPA")
    assert _approx(vehicle.store.get("aax"), fspa[0] / GRAV)
    assert _approx(vehicle.store.get("alx"), fspa[1] / GRAV)
    assert _approx(vehicle.store.get("anx"), -fspa[2] / GRAV)


def test_missing_acc_longx_is_ignored():
    vehicle, forces = _ready()
    assert "acc_longx" not in vehicle.store.names()
    forces.execute(vehicle, _ctx())
    want = THRUST / MASS
    assert vehicle.store.get("FSPA")[0] == pytest.approx(want, rel=RTOL, abs=ATOL)


def test_present_acc_longx_does_not_change_fspa():
    vehicle, forces = _ready()
    vehicle.store.define(Field("acc_longx", 9.0, "real", "data", "forces"))
    forces.execute(vehicle, _ctx())
    want = THRUST / MASS
    assert vehicle.store.get("FSPA")[0] == pytest.approx(want, rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(
        vehicle.store.get("FSPA"), (want, 0.0, 0.0), rtol=RTOL, atol=ATOL
    )
    assert _approx(vehicle.store.get("acc_longx"), 9.0)


def test_execute_does_not_write_plant_names():
    vehicle, forces = _ready(
        thrust=THRUST, catgt=0.4, pdynmc=1000.0, area=AREA, mass=MASS, cytgt=0.2, cntgt=0.3
    )
    forces.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("thrust"), THRUST)
    assert _approx(store.get("catgt"), 0.4)
    assert _approx(store.get("pdynmc"), 1000.0)
    assert _approx(store.get("area"), AREA)
    assert _approx(store.get("mass"), MASS)
    assert _approx(store.get("grav"), GRAV)
    assert _approx(store.get("cytgt"), 0.2)
    assert _approx(store.get("cntgt"), 0.3)


def test_fspa_is_a_fresh_vector():
    vehicle, forces = _ready()
    first = vehicle.store.get("FSPA")
    forces.execute(vehicle, _ctx())
    written = vehicle.store.get("FSPA")
    assert written is not first
    written[0] = 99.0
    forces.execute(vehicle, _ctx())
    want = THRUST / MASS
    np.testing.assert_allclose(
        vehicle.store.get("FSPA"), (want, 0.0, 0.0), rtol=RTOL, atol=ATOL
    )


def test_terminate_is_pass():
    vehicle, forces = _ready()
    assert forces.terminate(vehicle, _ctx()) is None
    np.testing.assert_allclose(
        vehicle.store.get("FSPA"), ZEROS3, rtol=RTOL, atol=ATOL
    )


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.sam6.rocket as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "cadac.eom.flat3" not in src
    assert "Flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "Plane5" not in src
    assert "Plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src
    assert "class Sam6Rocket:" not in src
    assert "class Sam6RocketForces:" in src
