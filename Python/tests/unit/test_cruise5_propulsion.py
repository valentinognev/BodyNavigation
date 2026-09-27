from math import cos
from pathlib import Path

import pytest

from cadac.constants import RAD
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.round3.cruise5.propulsion import Cruise5Propulsion

CRUISE5 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/CRUISE5_250115/CRUISE5"
RTOL = 1e-12
ATOL = 1e-14
INT_STEP = 0.05
MASS_INIT = 1000.0
FUEL_INIT = 150.0
ALT = 7000.0
MACH = 0.7
ALPHAX = 0.0
AREA = 0.929
MACH_COM = 0.7
GFTHM = 893620.0
TFTH = 1.0


def _deck():
    _, tables = parse_asc_deck(CRUISE5 / "cruise3_prop_deck.asc")
    return Datadeck.from_tables(tables)


def _ctx():
    return SimContext(
        sim_time=0.0, int_step=INT_STEP, event_time=0.0,
        out_fact=0.0, combus=None, vehicle_slot=0,
    )


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ready(mprop, cd=0.05, pdynmc=5000.0, thrust_com=1000.0):
    vehicle = _Vehicle()
    prop = Cruise5Propulsion(_deck())
    prop.define(vehicle)
    store = vehicle.store
    for name, value, ftype, role, module in (
        ("pdynmc", pdynmc, "real", "out", "environment"),
        ("mach", MACH, "real", "out", "environment"),
        ("alt", ALT, "real", "init/out", "newton"),
        ("cd", cd, "real", "out", "aerodynamics"),
        ("area", AREA, "real", "data", "aerodynamics"),
        ("alphax", ALPHAX, "real", "out", "control"),
    ):
        if name not in store.names():
            store.define(Field(name, value, ftype, role, module))
        else:
            store.set(name, value)
    store.set("mprop", mprop)
    store.set("mass_init", MASS_INIT)
    store.set("fuel_init", FUEL_INIT)
    store.set("mach_com", MACH_COM)
    store.set("gfthm", GFTHM)
    store.set("tfth", TFTH)
    store.set("thrust_com", thrust_com)
    prop.initialize(vehicle, _ctx())
    return vehicle, prop


def test_name_is_propulsion():
    assert Cruise5Propulsion(_deck()).name == "propulsion"


def test_mprop_0_zero_thrust_no_fuel_burn():
    vehicle, prop = _ready(0)
    store = vehicle.store
    prop.execute(vehicle, _ctx())
    assert store.get("thrust") == 0.0
    assert store.get("mass") == pytest.approx(MASS_INIT, rel=RTOL, abs=ATOL)
    assert store.get("fmasse") == 0.0


def test_mprop_1_commanded_thrust_and_fuel_table():
    vehicle, prop = _ready(1, thrust_com=1000.0)
    deck = _deck()
    ff = deck.look_up("ff_vs_thrust_alt_mach", 1000.0, ALT, MACH)
    fmasse = integrate(ff, 0.0, 0.0, INT_STEP)
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("thrust") == pytest.approx(1000.0, rel=RTOL, abs=ATOL)
    assert store.get("fmasse") == pytest.approx(fmasse, rel=RTOL, abs=ATOL)
    assert store.get("mass") == pytest.approx(MASS_INIT - fmasse, rel=RTOL, abs=ATOL)


def test_mprop_2_idle_iff_vs_alt():
    vehicle, prop = _ready(2)
    deck = _deck()
    fidle = deck.look_up("fidle_vs_alt_mach", ALT, MACH)
    ff = deck.look_up("iff_vs_alt", ALT)
    prop.execute(vehicle, _ctx())
    assert vehicle.store.get("thrust") == pytest.approx(fidle, rel=RTOL, abs=ATOL)
    assert vehicle.store.get("fidle") == pytest.approx(fidle, rel=RTOL, abs=ATOL)
    assert vehicle.store.get("fmasse") == pytest.approx(
        integrate(ff, 0.0, 0.0, INT_STEP), rel=RTOL, abs=ATOL
    )


def test_mprop_3_max_tav():
    vehicle, prop = _ready(3)
    deck = _deck()
    tav = deck.look_up("tav_vs_alt_mach", ALT, MACH)
    prop.execute(vehicle, _ctx())
    assert vehicle.store.get("thrust") == pytest.approx(tav, rel=RTOL, abs=ATOL)
    assert vehicle.store.get("tav") == pytest.approx(tav, rel=RTOL, abs=ATOL)


def test_mprop_4_mach_hold_one_step():
    cd = 0.05
    pdynmc = 5000.0
    vehicle, prop = _ready(4, cd=cd, pdynmc=pdynmc)
    deck = _deck()
    fidle = deck.look_up("fidle_vs_alt_mach", ALT, MACH)
    tav = deck.look_up("tav_vs_alt_mach", ALT, MACH)
    treqs = cd * pdynmc * AREA
    tcom = (MACH_COM - MACH) * GFTHM + treqs
    treqd_new = (tcom - 2.0 * 0.0) / TFTH
    treq = integrate(treqd_new, 0.0, 0.0, INT_STEP)
    treqb = treq / cos(ALPHAX * RAD)
    mprop = 4
    if treqb < fidle:
        mprop = 5
        treqb = fidle
    if treqb > tav:
        mprop = 6
        treqb = tav
    ff = deck.look_up("ff_vs_thrust_alt_mach", treqb, ALT, MACH)
    fmasse = integrate(ff, 0.0, 0.0, INT_STEP)
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("thrust") == pytest.approx(treqb, rel=RTOL, abs=ATOL)
    assert store.get("mprop") == mprop
    assert store.get("fmasse") == pytest.approx(fmasse, rel=RTOL, abs=ATOL)


def test_mprop_negative_raises():
    vehicle, prop = _ready(-1)
    with pytest.raises(ValueError, match="mprop"):
        prop.execute(vehicle, _ctx())
