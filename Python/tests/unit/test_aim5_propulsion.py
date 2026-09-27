from pathlib import Path

import numpy as np
import pytest

from cadac.env.us76 import atmosphere76
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.flat3.aim5.propulsion import Aim5Propulsion

AIM5 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/AIM5_250114/AIM5"
PROP = AIM5 / "aim5_prop_deck.asc"
RTOL = 1e-12
ATOL = 1e-14
# input_hori Missile
MPROP = 1
MASS = 63.8
AEXIT = 0.00948
ALT = 10000.0
PRESS = atmosphere76(ALT)[1]
PRES_SL = 101325.0


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.002,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _deck():
    _, tables = parse_asc_deck(PROP)
    return Datadeck.from_tables(tables)


def _expected_motor_on(deck, time, pres_sl, press, aexit):
    thrust_sl = deck.look_up("thrust_vs_time", time)
    thrust = thrust_sl + (pres_sl - press) * aexit
    mass = deck.look_up("mass_vs_time", time)
    return thrust_sl, thrust, mass


def _ready(*, time=0.0, mprop=MPROP, mass=MASS, aexit=AEXIT, press=PRESS):
    vehicle = _Vehicle()
    prop = Aim5Propulsion(_deck())
    prop.define(vehicle)
    store = vehicle.store
    store.define(Field("time", time, "real", "out", "kinematics"))
    store.define(Field("press", press, "real", "out", "environment"))
    store.set("mprop", mprop)
    store.set("mass", mass)
    store.set("aexit", aexit)
    return vehicle, prop, _deck()


def test_name_is_propulsion():
    assert Aim5Propulsion(_deck()).name == "propulsion"


def test_pres_sl_default_101325_after_define():
    vehicle = _Vehicle()
    Aim5Propulsion(_deck()).define(vehicle)
    np.testing.assert_allclose(vehicle.store.get("pres_sl"), PRES_SL, rtol=RTOL, atol=ATOL)
    field = vehicle.store.field("pres_sl")
    assert field.role == "data"
    assert field.module == "propulsion"


def test_time0_mprop1_thrust_mass_match_lookup_and_cpp():
    vehicle, prop, deck = _ready(time=0.0, mprop=1)
    prop.execute(vehicle, _ctx())
    _, want_thrust, want_mass = _expected_motor_on(deck, 0.0, PRES_SL, PRESS, AEXIT)
    np.testing.assert_allclose(vehicle.store.get("thrust"), want_thrust, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("mass"), want_mass, rtol=RTOL, atol=ATOL)


def test_time4_table_thrust_sl_zero_sets_mprop_and_thrust_zero():
    vehicle, prop, deck = _ready(time=4.0, mprop=1)
    thrust_sl, _, _ = _expected_motor_on(deck, 4.0, PRES_SL, PRESS, AEXIT)
    np.testing.assert_allclose(thrust_sl, 0.0, rtol=RTOL, atol=ATOL)
    prop.execute(vehicle, _ctx())
    assert vehicle.store.get("mprop") == 0
    np.testing.assert_allclose(vehicle.store.get("thrust"), 0.0, rtol=RTOL, atol=ATOL)


def test_mprop0_at_time0_thrust_zero_without_motor_on_tables():
    planted_mass = 99.0
    vehicle, prop, deck = _ready(time=0.0, mprop=0, mass=planted_mass)
    motor_thrust_sl, motor_thrust, motor_mass = _expected_motor_on(
        deck, 0.0, PRES_SL, PRESS, AEXIT
    )
    assert motor_thrust_sl != 0.0
    prop.execute(vehicle, _ctx())
    np.testing.assert_allclose(vehicle.store.get("thrust"), 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("mass"), planted_mass, rtol=RTOL, atol=ATOL)
    assert vehicle.store.get("thrust") != motor_thrust
    assert vehicle.store.get("mass") != motor_mass
    assert vehicle.store.get("mprop") == 0


def test_mprop2_raises_valueerror():
    vehicle, prop, _ = _ready(time=0.0, mprop=2)
    with pytest.raises(ValueError):
        prop.execute(vehicle, _ctx())


def test_define_does_not_register_time_or_press():
    vehicle = _Vehicle()
    Aim5Propulsion(_deck()).define(vehicle)
    names = vehicle.store.names()
    assert "mprop" in names and "pres_sl" in names and "aexit" in names
    assert "thrust" in names and "mass" in names
    assert "time" not in names and "press" not in names
    mprop = vehicle.store.field("mprop")
    assert mprop.type == "int" and mprop.role == "diag"
    thrust = vehicle.store.field("thrust")
    assert thrust.role == "out" and thrust.outputs == ("scrn", "plot")
    mass = vehicle.store.field("mass")
    assert mass.role == "out" and mass.outputs == ("scrn", "plot")
