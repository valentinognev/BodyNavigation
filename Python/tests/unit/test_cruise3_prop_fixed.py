from pathlib import Path

from cadac.constants import AGRAV
from cadac.env.iso62 import iso62
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.round3.hyper3.propulsion import Cruise3Propulsion

HYPER3 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/HYPER3_250114/HYPER3"
PROP = HYPER3 / "ghame3_prop_deck.asc"

MASS0 = 136077.0
FMASS0 = 81646.0
THROTTLE = 0.2
ACOWL = 27.87
ALPHAX = 7.0
DVBE = 250.0
INT_STEP = 0.01


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=INT_STEP,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _deck():
    _, tables = parse_asc_deck(PROP)
    return Datadeck.from_tables(tables)


def _climb_externals(store):
    atm = iso62(3000.0, DVBE)
    store.define(Field("rho", atm["rho"], "real", "out", "environment"))
    store.define(Field("mach", atm["mach"], "real", "out", "environment"))
    store.define(Field("dvbe", DVBE, "real", "init/out", "newton"))
    store.define(Field("alphax", ALPHAX, "real", "data", "aerodynamics"))
    return atm


def _climb_prop_data(store, *, mprop, fmasse=0.0, fmassd=0.0):
    store.set("mprop", mprop)
    store.set("throttle", THROTTLE)
    store.set("mass0", MASS0)
    store.set("fmass0", FMASS0)
    store.set("acowl", ACOWL)
    store.set("fmasse", fmasse)
    store.set("fmassd", fmassd)


def test_name_is_propulsion():
    assert Cruise3Propulsion(_deck()).name == "propulsion"


def test_define_registers_propulsion_fields():
    vehicle = _Vehicle()
    Cruise3Propulsion(_deck()).define(vehicle)
    store = vehicle.store
    assert store.get("mprop") == 0
    for name in (
        "acowl",
        "throttle",
        "thrtl_max",
        "qhold",
        "mass",
        "mass0",
        "tq",
        "thrtl_idle",
        "fmass0",
        "fmasse",
        "fmassd",
        "ca",
        "spi",
        "thrust",
        "mass_flow",
        "fmassr",
    ):
        assert store.get(name) == 0.0


def test_initialize_sets_mass_from_mass0():
    vehicle = _Vehicle()
    prop = Cruise3Propulsion(_deck())
    prop.define(vehicle)
    vehicle.store.set("mass0", MASS0)
    prop.initialize(vehicle, _ctx())
    assert vehicle.store.get("mass") == MASS0


def test_mprop_0_zero_thrust_and_fmassd():
    deck = _deck()
    vehicle = _Vehicle()
    prop = Cruise3Propulsion(deck)
    prop.define(vehicle)
    _climb_externals(vehicle.store)
    _climb_prop_data(vehicle.store, mprop=0, fmasse=0.0, fmassd=1.0)
    prop.initialize(vehicle, _ctx())
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("thrust") == 0.0
    assert store.get("fmassd") == 0.0
    assert store.get("fmasse") == 0.0
    assert store.get("mass") == MASS0
    assert store.get("mprop") == 0


def test_mprop_1_fixed_throttle_climb_ic():
    deck = _deck()
    vehicle = _Vehicle()
    prop = Cruise3Propulsion(deck)
    prop.define(vehicle)
    atm = _climb_externals(vehicle.store)
    _climb_prop_data(vehicle.store, mprop=1, fmasse=0.0, fmassd=0.0)
    prop.initialize(vehicle, _ctx())

    rho = atm["rho"]
    mach = atm["mach"]
    spi = deck.look_up("spi_vs_throttle_mach", THROTTLE, mach)
    ca = deck.look_up("ca_vs_alpha_mach", ALPHAX, mach)
    thrust = spi * 0.029 * THROTTLE * AGRAV * rho * DVBE * ca * ACOWL
    fmassd_next = thrust / (spi * AGRAV)
    fmasse = integrate(fmassd_next, 0.0, 0.0, INT_STEP)
    mass = MASS0 - fmasse
    fmassr = FMASS0 - fmasse

    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("spi") == spi
    assert store.get("ca") == ca
    assert store.get("thrust") == thrust
    assert store.get("fmassd") == fmassd_next
    assert store.get("fmasse") == fmasse
    assert store.get("mass") == mass
    assert store.get("fmassr") == fmassr
    assert store.get("mprop") == 1
    assert store.get("throttle") == THROTTLE


def test_mprop_1_shuts_down_when_fuel_expended():
    deck = _deck()
    vehicle = _Vehicle()
    prop = Cruise3Propulsion(deck)
    prop.define(vehicle)
    _climb_externals(vehicle.store)
    _climb_prop_data(vehicle.store, mprop=1, fmasse=FMASS0, fmassd=0.0)
    prop.initialize(vehicle, _ctx())
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("fmassr") <= 0.0
    assert store.get("mprop") == 0
    assert store.get("thrust") == 0.0
    assert store.get("fmassd") == 0.0
