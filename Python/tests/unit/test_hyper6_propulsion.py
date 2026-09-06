from math import cos
from pathlib import Path

import numpy as np
import pytest

from cadac.constants import AGRAV, R, RAD
from cadac.env.us76 import atmosphere76
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.hyper6.propulsion import Hyper6Propulsion

HYPER6 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/HYPER6_250125/HYPER6"
PROP = HYPER6 / "ghame6_prop_deck.asc"
AERO = HYPER6 / "ghame6_aero_deck.asc"

RTOL = 1e-12
ATOL = 1e-14

ALPHAX = 2.5
DVBA = 1000.0
ALT = 10000.0
VMASS0 = 136077.0
FMASS0 = 81646.0
QHOLD = 200000.0
TQ = 1.0
ACOWL = 27.87
THRTL_IDLE = 0.05
THRTL_MAX = 2.0
THROTTLE0 = 0.05
REFA = 557.42
INT_STEP = 0.01

IBBB0 = np.array(
    [
        [1.573e6, 0.0, 0.38e6],
        [0.0, 31.6e6, 0.0],
        [0.38e6, 0.0, 32.54e6],
    ],
    dtype=float,
)
IBBB1 = np.array(
    [
        [1.18e6, 0.0, 0.24e6],
        [0.0, 19.25e6, 0.0],
        [0.24e6, 0.0, 20.2e6],
    ],
    dtype=float,
)

DEFINED = (
    "mprop",
    "acowl",
    "throttle",
    "thrtl_max",
    "qhold",
    "vmass",
    "vmass0",
    "IBBB",
    "IBBB0",
    "IBBB1",
    "fmass0",
    "fmasse",
    "fmassd",
    "ca",
    "spi",
    "thrust",
    "mass_flow",
    "fmassr",
    "thrustx",
    "tq",
    "thrtl_idle",
    "fuel_flow_rate",
    "vmass0_st",
    "fmass0_st",
    "moi_roll_exo_0",
    "moi_roll_exo_1",
    "moi_trans_exo_0",
    "moi_trans_exo_1",
    "mfreeze_prop",
    "thrustf",
    "vmassf",
    "IBBBF",
)
ROLES = {
    "mprop": "data",
    "acowl": "data",
    "throttle": "data/diag",
    "thrtl_max": "data",
    "qhold": "data",
    "vmass": "out",
    "vmass0": "data",
    "IBBB": "out",
    "IBBB0": "init",
    "IBBB1": "init",
    "fmass0": "data",
    "fmasse": "state",
    "fmassd": "state",
    "ca": "diag",
    "spi": "diag",
    "thrust": "out",
    "mass_flow": "diag",
    "fmassr": "diag",
    "thrustx": "diag",
    "tq": "data",
    "thrtl_idle": "data",
    "fuel_flow_rate": "data",
    "vmass0_st": "data",
    "fmass0_st": "data",
    "moi_roll_exo_0": "data",
    "moi_roll_exo_1": "data",
    "moi_trans_exo_0": "data",
    "moi_trans_exo_1": "data",
    "mfreeze_prop": "save",
    "thrustf": "save",
    "vmassf": "save",
    "IBBBF": "save",
}
OUTPUTS = {
    "mprop": (),
    "acowl": (),
    "throttle": ("scrn", "plot"),
    "thrtl_max": (),
    "qhold": (),
    "vmass": ("scrn", "plot"),
    "vmass0": (),
    "IBBB": (),
    "IBBB0": (),
    "IBBB1": (),
    "fmass0": (),
    "fmasse": (),
    "fmassd": (),
    "ca": (),
    "spi": (),
    "thrust": (),
    "mass_flow": (),
    "fmassr": ("scrn", "plot"),
    "thrustx": ("scrn", "plot"),
    "tq": (),
    "thrtl_idle": (),
    "fuel_flow_rate": (),
    "vmass0_st": (),
    "fmass0_st": (),
    "moi_roll_exo_0": (),
    "moi_roll_exo_1": (),
    "moi_trans_exo_0": (),
    "moi_trans_exo_1": (),
    "mfreeze_prop": (),
    "thrustf": (),
    "vmassf": (),
    "IBBBF": (),
}
INT_FIELDS = ("mprop", "mfreeze_prop")
MAT_FIELDS = ("IBBB", "IBBB0", "IBBB1", "IBBBF")
NOT_DEFINED = (
    "vmach",
    "pdynmc",
    "cd",
    "cx",
    "area",
    "refa",
    "alphax",
    "time",
    "rho",
    "dvba",
    "mfreeze",
    "isp_fuel",
    "burntime",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


class _BoomDeck:
    def look_up(self, *args, **kwargs):
        raise AssertionError("look_up must not be called")


def _ctx(dt=INT_STEP):
    return SimContext(
        sim_time=0.0,
        int_step=dt,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _deck():
    _, tables = parse_asc_deck(PROP)
    return Datadeck.from_tables(tables)


def _aero_deck():
    _, tables = parse_asc_deck(AERO)
    return Datadeck.from_tables(tables)


def _us76_air(alt=ALT, dvba=DVBA):
    rho, _press, tempk = atmosphere76(alt)
    vsound = (1.4 * R * tempk) ** 0.5
    vmach = abs(dvba / vsound)
    pdynmc = 0.5 * rho * dvba * dvba
    return rho, vmach, pdynmc


def _climb_cd(vmach, alphax=ALPHAX):
    deck = _aero_deck()
    cd0 = deck.look_up("cd0_vs_alpha_mach", alphax, vmach)
    cda = deck.look_up("cda_vs_alpha_mach", alphax, vmach)
    return cd0 + cda * alphax


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _plant(store, **overrides):
    rho, vmach, pdynmc = _us76_air()
    vals = {
        "rho": rho,
        "vmach": vmach,
        "pdynmc": pdynmc,
        "dvba": DVBA,
        "cd": _climb_cd(vmach),
        "cx": 0.0,
        "refa": REFA,
        "alphax": ALPHAX,
        "time": 0.0,
    }
    vals.update(overrides)
    store.define(Field("rho", vals["rho"], "real", "out", "environment"))
    store.define(Field("vmach", vals["vmach"], "real", "out", "environment"))
    store.define(Field("pdynmc", vals["pdynmc"], "real", "out", "environment"))
    store.define(Field("dvba", vals["dvba"], "real", "out", "environment"))
    store.define(Field("cd", vals["cd"], "real", "dia", "aerodynamics"))
    store.define(Field("cx", vals["cx"], "real", "out", "aerodynamics"))
    store.define(Field("refa", vals["refa"], "real", "init", "aerodynamics"))
    store.define(Field("alphax", vals["alphax"], "real", "init/diag", "kinematics"))
    store.define(Field("time", vals["time"], "real", "exec", "kinematics"))


def _prop_data(store, *, mprop, **overrides):
    vals = {
        "acowl": ACOWL,
        "throttle": THROTTLE0,
        "thrtl_max": THRTL_MAX,
        "qhold": QHOLD,
        "vmass0": VMASS0,
        "fmass0": FMASS0,
        "tq": TQ,
        "thrtl_idle": THRTL_IDLE,
        "fmasse": 0.0,
        "fmassd": 0.0,
    }
    vals.update(overrides)
    store.set("mprop", mprop)
    for name, value in vals.items():
        store.set(name, value)


def _ready(deck, *, mprop, plant=None, **prop_kw):
    vehicle = _Vehicle()
    prop = Hyper6Propulsion(deck)
    prop.define(vehicle)
    _plant(vehicle.store, **(plant or {}))
    _prop_data(vehicle.store, mprop=mprop, **prop_kw)
    prop.initialize(vehicle, _ctx())
    return vehicle, prop


def _expected(deck, store, dt=INT_STEP):
    mprop = store.get("mprop")
    throttle = store.get("throttle")
    qhold = store.get("qhold")
    vmass0 = store.get("vmass0")
    fmass0 = store.get("fmass0")
    tq = store.get("tq")
    thrtl_idle = store.get("thrtl_idle")
    thrtl_max = store.get("thrtl_max")
    acowl = store.get("acowl")
    ibbb0 = np.array(store.get("IBBB0"), dtype=float)
    ibbb1 = np.array(store.get("IBBB1"), dtype=float)
    vmass = store.get("vmass")
    fmasse = store.get("fmasse")
    fmassd = store.get("fmassd")
    rho = store.get("rho")
    vmach = store.get("vmach")
    pdynmc = store.get("pdynmc")
    dvba = store.get("dvba")
    refa = store.get("refa")
    cd = store.get("cd")
    alphax = store.get("alphax")
    ibbb = np.array(store.get("IBBB"), dtype=float)
    fmassr = store.get("fmassr")

    spi = 0.0
    ca = 0.0
    thrust = 0.0
    mass_flow = 0.0

    if mprop > 0:
        if mprop == 1 or mprop == 2:
            spi = deck.look_up("spi_vs_throttle_mach", throttle, vmach)
            ca = deck.look_up("ca_vs_alpha_mach", alphax, vmach)
        if mprop == 1:
            thrust = spi * 0.029 * throttle * AGRAV * rho * dvba * ca * acowl
        if mprop == 2:
            denom = 0.029 * spi * AGRAV * rho * dvba * ca * acowl
            if denom != 0:
                thrst_req = refa * cd * qhold / cos(alphax * RAD)
                throtl_req = thrst_req / denom
                gainq = 2 * vmass / (rho * dvba * denom * tq)
                ethrotl = gainq * (qhold - pdynmc)
                throttle = ethrotl + throtl_req
            if throttle < 0:
                throttle = thrtl_idle
            if throttle > thrtl_max:
                throttle = thrtl_max
            spi = deck.look_up("spi_vs_throttle_mach", throttle, vmach)
            thrust = spi * 0.029 * throttle * AGRAV * rho * dvba * ca * acowl
        if spi != 0:
            fmassd_next = thrust / (spi * AGRAV)
            fmasse = integrate(fmassd_next, fmassd, fmasse, dt)
            fmassd = fmassd_next
        vmass = vmass0 - fmasse
        fmassr = fmass0 - fmasse
        mass_ratio = fmasse / fmass0
        ibbb = ibbb0 + (ibbb1 - ibbb0) * mass_ratio
        mass_flow = thrust / (AGRAV * spi)
        if fmassr <= 0:
            mprop = 0
    if mprop == 0:
        fmassd = 0.0
        thrust = 0.0
    thrustx = thrust / 1000.0
    return {
        "mprop": mprop,
        "throttle": throttle,
        "spi": spi,
        "ca": ca,
        "thrust": thrust,
        "thrustx": thrustx,
        "mass_flow": mass_flow,
        "fmassd": fmassd,
        "fmasse": fmasse,
        "vmass": vmass,
        "fmassr": fmassr,
        "IBBB": ibbb,
    }


def test_name_is_propulsion():
    assert Hyper6Propulsion(None).name == "propulsion"


def test_define_registers_cpp_fields_not_env_aero():
    vehicle = _Vehicle()
    Hyper6Propulsion(None).define(vehicle)
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
            if name == "throttle":
                assert store.get(name) == THROTTLE0
            else:
                assert store.get(name) == 0.0
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_initialize_sets_mass_and_ibbb():
    vehicle = _Vehicle()
    prop = Hyper6Propulsion(None)
    prop.define(vehicle)
    vehicle.store.set("vmass0", VMASS0)
    prop.initialize(vehicle, _ctx())
    store = vehicle.store
    assert store.get("vmass") == VMASS0
    np.testing.assert_allclose(store.get("IBBB0"), IBBB0)
    np.testing.assert_allclose(store.get("IBBB1"), IBBB1)
    np.testing.assert_allclose(store.get("IBBB"), IBBB0)
    assert store.get("vmass0_st") == 0.0
    assert store.get("fmass0_st") == 0.0


def test_mprop_0_zero_thrust_without_lookup():
    vehicle, prop = _ready(_BoomDeck(), mprop=0, fmassd=1.0)
    store = vehicle.store
    store.set("throttle", 0.8)
    prop.execute(vehicle, _ctx())
    assert store.get("thrust") == 0.0
    assert store.get("thrustx") == 0.0
    assert store.get("fmassd") == 0.0
    assert store.get("fmasse") == 0.0
    assert store.get("vmass") == VMASS0
    assert store.get("mprop") == 0
    assert store.get("ca") == 0.0
    assert store.get("spi") == 0.0
    assert store.get("mass_flow") == 0.0
    np.testing.assert_allclose(store.get("IBBB"), IBBB0)
    assert "mfreeze" not in store.names()


def test_mprop_2_climb_autothrottle_throttle_in_range():
    deck = _deck()
    vehicle, prop = _ready(deck, mprop=2)
    store = vehicle.store
    rho, vmach, pdynmc = _us76_air()
    assert _approx(store.get("vmach"), vmach)
    assert vmach == pytest.approx(3.3, rel=0.02)
    assert store.get("qhold") == QHOLD
    want = _expected(deck, store)
    assert 0.0 < want["throttle"] <= THRTL_MAX

    prop.execute(vehicle, _ctx())
    got = store.get("throttle")
    assert 0.0 < got <= THRTL_MAX
    assert _approx(got, want["throttle"])
    assert _approx(store.get("thrust"), want["thrust"])
    assert store.get("thrust") > 0.0
    assert store.get("mprop") == 2


def test_mprop_2_matches_cadac_formulas():
    deck = _deck()
    vehicle, prop = _ready(deck, mprop=2)
    store = vehicle.store
    want = _expected(deck, store)
    prop.execute(vehicle, _ctx())
    for name, value in want.items():
        if name == "IBBB":
            np.testing.assert_allclose(store.get(name), value, rtol=RTOL, atol=ATOL)
        else:
            assert _approx(store.get(name), value), name


def test_mprop_4_raises():
    vehicle = _Vehicle()
    prop = Hyper6Propulsion(None)
    prop.define(vehicle)
    vehicle.store.set("mprop", 4)
    with pytest.raises(ValueError):
        prop.execute(vehicle, _ctx())


@pytest.mark.parametrize("mprop", [-1, 3, 99])
def test_other_mprop_raises(mprop):
    vehicle = _Vehicle()
    prop = Hyper6Propulsion(None)
    prop.define(vehicle)
    vehicle.store.set("mprop", mprop)
    with pytest.raises(ValueError):
        prop.execute(vehicle, _ctx())


def test_mprop_1_matches_cpp():
    deck = _deck()
    vehicle, prop = _ready(deck, mprop=1, throttle=1.0)
    store = vehicle.store
    want = _expected(deck, store)
    assert want["throttle"] == 1.0
    prop.execute(vehicle, _ctx())
    assert store.get("throttle") == 1.0
    assert _approx(store.get("thrust"), want["thrust"])
    assert _approx(store.get("spi"), want["spi"])
    assert _approx(store.get("ca"), want["ca"])
    assert _approx(store.get("fmasse"), want["fmasse"])
    assert _approx(store.get("vmass"), want["vmass"])
    assert _approx(store.get("fmassr"), want["fmassr"])
    assert store.get("mprop") == 1


def test_mprop_1_shuts_down_when_fuel_expended():
    deck = _deck()
    vehicle, prop = _ready(deck, mprop=1, fmasse=FMASS0)
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("fmassr") <= 0.0
    assert store.get("mprop") == 0
    assert store.get("thrust") == 0.0
    assert store.get("fmassd") == 0.0


def test_mprop_2_stored_slope_second_step():
    deck = _deck()
    vehicle, prop = _ready(deck, mprop=2)
    first = _expected(deck, vehicle.store)
    prop.execute(vehicle, _ctx())
    second = _expected(deck, vehicle.store)
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("throttle"), second["throttle"])
    assert _approx(store.get("thrust"), second["thrust"])
    assert _approx(store.get("fmasse"), second["fmasse"])
    assert _approx(store.get("fmassd"), second["fmassd"])
    assert first["fmasse"] != second["fmasse"]


def test_execute_skips_mfreeze_when_absent():
    deck = _deck()
    vehicle, prop = _ready(deck, mprop=2)
    prop.execute(vehicle, _ctx())
    assert "mfreeze" not in vehicle.store.names()
    assert np.isfinite(vehicle.store.get("thrust"))


def test_terminate_exists_and_is_pass():
    vehicle, prop = _ready(None, mprop=0)
    assert prop.terminate(vehicle, _ctx()) is None
