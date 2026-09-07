from pathlib import Path

import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.sam6.rocket import Sam6RocketPropulsion

RTOL = 1e-12
ATOL = 1e-14
PSL = 101325.0
THRUST_SL = 128600.0
MASS_LAUNCH = 6000.0
MASS_FUEL = 4000.0
ISP = 230.0
AEXIT = 0.282
G0 = 9.81

DEFINED = (
    "mprop",
    "pres_sl",
    "aexit",
    "mass_launch",
    "mass_fuel",
    "isp",
    "thrust_sl",
    "thrust",
    "mass",
)
ROLES = {
    "mprop": "data/out",
    "pres_sl": "data",
    "aexit": "data",
    "mass_launch": "data",
    "mass_fuel": "data",
    "isp": "data",
    "thrust_sl": "data",
    "thrust": "out",
    "mass": "out",
}
OUTPUTS = {
    "mprop": (),
    "pres_sl": (),
    "aexit": (),
    "mass_launch": (),
    "mass_fuel": (),
    "isp": (),
    "thrust_sl": (),
    "thrust": ("com",),
    "mass": (),
}
DEFAULTS = {
    "mprop": 0,
    "pres_sl": 101325.0,
    "aexit": 0.282,
    "mass_launch": 6000.0,
    "mass_fuel": 4000.0,
    "isp": 230.0,
    "thrust_sl": 128600.0,
    "thrust": 0.0,
    "mass": 0.0,
}
NOT_DEFINED = ("launch_time", "press", "time")


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


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _plant(store, *, launch_time, press, mprop=None):
    store.define(Field("launch_time", launch_time, "real", "diag", "kinematics"))
    store.define(Field("press", press, "real", "out", "environment"))
    if mprop is not None:
        store.set("mprop", mprop)


def _ready(*, launch_time=0.0, press=PSL, mprop=1):
    vehicle = _Vehicle()
    prop = Sam6RocketPropulsion()
    prop.define(vehicle)
    prop.initialize(vehicle, _ctx())
    _plant(vehicle.store, launch_time=launch_time, press=press, mprop=mprop)
    return vehicle, prop


def _mass_flow(thrust_sl=THRUST_SL, isp=ISP):
    return thrust_sl / (isp * G0)


def test_name_is_propulsion():
    assert Sam6RocketPropulsion().name == "propulsion"


def test_define_cpp_fields():
    vehicle = _Vehicle()
    Sam6RocketPropulsion().define(vehicle)
    store = vehicle.store
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        assert field.module == "propulsion", name
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        if name == "mprop":
            assert field.type == "int"
            assert store.get(name) == DEFAULTS[name]
        else:
            assert field.type == "real"
            assert _approx(store.get(name), DEFAULTS[name]), name
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_initialize_is_pass():
    vehicle = _Vehicle()
    prop = Sam6RocketPropulsion()
    prop.define(vehicle)
    before = {name: vehicle.store.get(name) for name in DEFINED}
    assert prop.initialize(vehicle, _ctx()) is None
    assert vehicle.store.get("mprop") == before["mprop"]
    for name in DEFINED:
        if name == "mprop":
            continue
        assert _approx(vehicle.store.get(name), before[name]), name


def test_sea_level_launch_thrust_and_mass():
    vehicle, prop = _ready(launch_time=0.0, mprop=1, press=PSL)
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("thrust"), THRUST_SL)
    assert _approx(store.get("mass"), MASS_LAUNCH)
    assert store.get("mprop") == 1


def test_unknown_mprop_raises():
    vehicle, prop = _ready(mprop=2)
    with pytest.raises(ValueError, match="mprop"):
        prop.execute(vehicle, _ctx())


def test_mprop_zero_leaves_mass_and_zeros_thrust():
    vehicle, prop = _ready(launch_time=10.0, mprop=1, press=PSL)
    prop.execute(vehicle, _ctx())
    mass_on = vehicle.store.get("mass")
    vehicle.store.set("mprop", 0)
    vehicle.store.set("thrust", 99.0)
    prop.execute(vehicle, _ctx())
    assert vehicle.store.get("mprop") == 0
    assert _approx(vehicle.store.get("thrust"), 0.0)
    assert _approx(vehicle.store.get("mass"), mass_on)


def test_altitude_corrects_thrust():
    press = 80000.0
    vehicle, prop = _ready(launch_time=0.0, mprop=1, press=press)
    prop.execute(vehicle, _ctx())
    want = THRUST_SL + (PSL - press) * AEXIT
    assert _approx(vehicle.store.get("thrust"), want)
    assert _approx(vehicle.store.get("mass"), MASS_LAUNCH)


def test_mass_decreases_with_launch_time():
    launch_time = 10.0
    vehicle, prop = _ready(launch_time=launch_time, mprop=1, press=PSL)
    prop.execute(vehicle, _ctx())
    want_mass = MASS_LAUNCH - _mass_flow() * launch_time
    assert _approx(vehicle.store.get("mass"), want_mass)
    assert vehicle.store.get("mprop") == 1
    assert _approx(vehicle.store.get("thrust"), THRUST_SL)


def test_burnout_sets_mprop_off():
    mass_flow = _mass_flow()
    launch_time = (MASS_FUEL / mass_flow) + 1.0
    vehicle, prop = _ready(launch_time=launch_time, mprop=1, press=PSL)
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("mprop") == 0
    assert _approx(store.get("thrust"), 0.0)
    want_mass = MASS_LAUNCH - mass_flow * launch_time
    assert _approx(store.get("mass"), want_mass)
    assert store.get("mass") <= MASS_LAUNCH - MASS_FUEL


def test_exact_burnout_threshold():
    mass_flow = _mass_flow()
    launch_time = MASS_FUEL / mass_flow
    vehicle, prop = _ready(launch_time=launch_time, mprop=1, press=PSL)
    prop.execute(vehicle, _ctx())
    assert vehicle.store.get("mprop") == 0
    assert _approx(vehicle.store.get("thrust"), 0.0)
    assert _approx(vehicle.store.get("mass"), MASS_LAUNCH - MASS_FUEL)


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.sam6.rocket as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src


def test_terminate_is_pass():
    vehicle, prop = _ready()
    assert prop.terminate(vehicle, _ctx()) is None
    assert _approx(vehicle.store.get("thrust"), 0.0)
