import inspect

import numpy as np
import pytest

from cadac.constants import AGRAV
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.agm6.propulsion import Agm6Propulsion

RTOL = 1e-12
ATOL = 1e-14

MPROP = 1
THRSL = 10000.0
THROTL = 1.0
SPI = 210.0
AEXIT = 0.02
PRESS = 101325.0
PSL = 101325.0
VMASS0 = 1360.0
FMASS0 = 250.0
AI11 = 42.5
AI33 = 2632.0
DT = 0.001
G_CPP = 9.81

DEFINED = (
    "mprop",
    "aexit",
    "vmass",
    "thrust",
    "vmass0",
    "ai11",
    "ai33",
    "mfreeze_prop",
    "thrustf",
    "vmassf",
    "spi",
    "throtl",
    "thrsl",
    "fmass0",
    "fmasse",
    "fmassed",
    "IBBB",
    "eng_ang_mom",
)
ROLES = {
    "mprop": "data",
    "aexit": "data",
    "vmass": "out",
    "thrust": "out",
    "vmass0": "data",
    "ai11": "out",
    "ai33": "out",
    "mfreeze_prop": "save",
    "thrustf": "save",
    "vmassf": "save",
    "spi": "data",
    "throtl": "data",
    "thrsl": "data",
    "fmass0": "data",
    "fmasse": "state",
    "fmassed": "state",
    "IBBB": "out",
    "eng_ang_mom": "out",
}
OUTPUTS = {
    "mprop": (),
    "aexit": (),
    "vmass": ("scrn", "plot"),
    "thrust": ("scrn", "plot"),
    "vmass0": (),
    "ai11": (),
    "ai33": (),
    "mfreeze_prop": (),
    "thrustf": (),
    "vmassf": (),
    "spi": (),
    "throtl": (),
    "thrsl": (),
    "fmass0": (),
    "fmasse": ("plot",),
    "fmassed": (),
    "IBBB": (),
    "eng_ang_mom": (),
}
INT_FIELDS = ("mprop", "mfreeze_prop")
MAT_FIELDS = ("IBBB",)
NOT_DEFINED = ("press", "mfreeze")

FMASSED_NEW = THRSL * THROTL / (SPI * G_CPP)
FMASSE_STEP1 = integrate(FMASSED_NEW, 0.0, 0.0, DT)
THRUST_SL = THRSL * THROTL + (PSL - PRESS) * AEXIT
IBBB_DIAG = np.diag([AI11, AI33, AI33])


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(dt=DT):
    return SimContext(
        sim_time=0.0,
        int_step=dt,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _plant_press(store, press=PRESS):
    store.define(Field("press", press, "real", "out", "environment"))


def _ready(*, mprop=MPROP, press=PRESS, plant_press=True, **overrides):
    vehicle = _Vehicle()
    prop = Agm6Propulsion()
    prop.define(vehicle)
    store = vehicle.store
    if plant_press:
        _plant_press(store, press)
    vals = {
        "mprop": mprop,
        "aexit": AEXIT,
        "vmass0": VMASS0,
        "ai11": AI11,
        "ai33": AI33,
        "spi": SPI,
        "throtl": THROTL,
        "thrsl": THRSL,
        "fmass0": FMASS0,
        "fmasse": 0.0,
        "fmassed": 0.0,
    }
    vals.update(overrides)
    for name, value in vals.items():
        store.set(name, value)
    prop.initialize(vehicle, _ctx())
    return vehicle, prop


def test_name_is_propulsion():
    assert Agm6Propulsion.name == "propulsion"
    assert Agm6Propulsion().name == "propulsion"


def test_constructor_takes_no_deck():
    sig = inspect.signature(Agm6Propulsion.__init__)
    assert list(sig.parameters) == ["self"]
    Agm6Propulsion()


def test_define_registers_cpp_fields_plus_ibbb():
    vehicle = _Vehicle()
    Agm6Propulsion().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    zeros33 = np.zeros((3, 3))
    for name in DEFINED:
        field = store.field(name)
        assert field.role == ROLES[name], name
        assert field.module == "propulsion"
        assert field.outputs == OUTPUTS[name], name
        if name in INT_FIELDS:
            assert field.type == "int"
            assert store.get(name) == 0
        elif name in MAT_FIELDS:
            assert field.type == "mat"
            np.testing.assert_allclose(store.get(name), zeros33)
        else:
            assert field.type == "real"
            assert store.get(name) == 0.0
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_initialize_sets_vmass_from_vmass0_not_inertia():
    vehicle = _Vehicle()
    prop = Agm6Propulsion()
    prop.define(vehicle)
    store = vehicle.store
    store.set("vmass0", VMASS0)
    store.set("ai11", AI11)
    store.set("ai33", AI33)
    prop.initialize(vehicle, _ctx())
    assert store.get("vmass") == VMASS0
    assert store.get("ai11") == AI11
    assert store.get("ai33") == AI33
    np.testing.assert_allclose(store.get("IBBB"), np.zeros((3, 3)))
    assert store.get("eng_ang_mom") == 0.0


def test_mprop_1_sea_level_thrust_fuel_and_ibbb():
    vehicle, prop = _ready(mprop=1)
    store = vehicle.store
    assert store.get("ai11") == AI11
    prop.execute(vehicle, _ctx())
    assert _approx(store.get("thrust"), THRUST_SL)
    assert store.get("thrust") == pytest.approx(10000.0, rel=RTOL, abs=ATOL)
    assert store.get("fmasse") > 0.0
    assert _approx(store.get("fmasse"), FMASSE_STEP1)
    assert _approx(store.get("fmassed"), FMASSED_NEW)
    assert _approx(store.get("vmass"), VMASS0 - FMASSE_STEP1)
    assert store.get("IBBB")[0, 0] == AI11
    np.testing.assert_allclose(store.get("IBBB"), IBBB_DIAG, rtol=RTOL, atol=ATOL)
    assert store.get("eng_ang_mom") == 0.0
    assert store.get("mprop") == 1
    assert not _approx(FMASSED_NEW, THRSL * THROTL / (SPI * AGRAV))


def test_mprop_0_zero_thrust_does_not_integrate_fuel():
    vehicle, prop = _ready(mprop=0, fmassed=1.0, plant_press=False)
    store = vehicle.store
    prop.execute(vehicle, _ctx())
    assert store.get("thrust") == 0.0
    assert store.get("fmasse") == 0.0
    assert store.get("fmassed") == 1.0
    assert store.get("vmass") == VMASS0
    assert store.get("mprop") == 0
    np.testing.assert_allclose(store.get("IBBB"), IBBB_DIAG, rtol=RTOL, atol=ATOL)
    assert store.get("eng_ang_mom") == 0.0
    assert "mfreeze" not in store.names()


def test_mprop_2_raises():
    vehicle, prop = _ready(mprop=2, plant_press=False)
    with pytest.raises(ValueError):
        prop.execute(vehicle, _ctx())


@pytest.mark.parametrize("mprop", [-1, 3, 99])
def test_unknown_mprop_raises(mprop):
    vehicle, prop = _ready(mprop=mprop, plant_press=False)
    with pytest.raises(ValueError):
        prop.execute(vehicle, _ctx())


def test_fuel_expended_sets_mprop_0():
    vehicle, prop = _ready(mprop=1, fmasse=FMASS0)
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("mprop") == 0
    assert store.get("fmasse") >= FMASS0


def test_altitude_thrust_uses_nozzle_exit():
    press = 90000.0
    want = THRSL * THROTL + (PSL - press) * AEXIT
    vehicle, prop = _ready(mprop=1, press=press)
    prop.execute(vehicle, _ctx())
    assert _approx(vehicle.store.get("thrust"), want)
    assert want == pytest.approx(10226.5, rel=RTOL, abs=ATOL)


def test_stored_slope_second_step():
    vehicle, prop = _ready(mprop=1)
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    fmasse_1 = store.get("fmasse")
    fmassed_1 = store.get("fmassed")
    fmasse_2 = integrate(FMASSED_NEW, fmassed_1, fmasse_1, DT)
    prop.execute(vehicle, _ctx())
    assert _approx(store.get("fmasse"), fmasse_2)
    assert store.get("fmasse") > fmasse_1
    assert _approx(store.get("fmassed"), FMASSED_NEW)


def test_execute_skips_mfreeze_when_absent():
    vehicle, prop = _ready(mprop=1)
    prop.execute(vehicle, _ctx())
    assert "mfreeze" not in vehicle.store.names()
    assert np.isfinite(vehicle.store.get("thrust"))
    assert vehicle.store.get("thrust") == pytest.approx(10000.0, rel=RTOL, abs=ATOL)


def test_terminate_exists_and_is_pass():
    vehicle, prop = _ready(mprop=0, plant_press=False)
    assert prop.terminate(vehicle, _ctx()) is None
