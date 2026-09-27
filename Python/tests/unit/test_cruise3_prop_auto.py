from math import cos
from pathlib import Path

from cadac.constants import AGRAV, RAD
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
THROTTLE0 = 0.2
ACOWL = 27.87
ALPHAX = 2.5
DVBE = 250.0
INT_STEP = 0.01
QHOLD = 50000.0
TQ = 1.0
THRTL_IDLE = 0.05
THRTL_MAX = 2.0
AREA = 557.42
CD = 0.05
ALT = 3000.0


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


def _atm():
    return iso62(ALT, DVBE)


def _auto_externals(store, *, pdynmc):
    atm = _atm()
    store.define(Field("rho", atm["rho"], "real", "out", "environment"))
    store.define(Field("mach", atm["mach"], "real", "out", "environment"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("dvbe", DVBE, "real", "init/out", "newton"))
    store.define(Field("alphax", ALPHAX, "real", "data", "aerodynamics"))
    store.define(Field("cd", CD, "real", "out", "aerodynamics"))
    store.define(Field("area", AREA, "real", "data", "aerodynamics"))
    return atm


def _auto_prop_data(store):
    store.set("mprop", 2)
    store.set("throttle", THROTTLE0)
    store.set("mass0", MASS0)
    store.set("fmass0", FMASS0)
    store.set("acowl", ACOWL)
    store.set("qhold", QHOLD)
    store.set("tq", TQ)
    store.set("thrtl_idle", THRTL_IDLE)
    store.set("thrtl_max", THRTL_MAX)
    store.set("fmasse", 0.0)
    store.set("fmassd", 0.0)


def _expected_auto(deck, atm, pdynmc):
    rho = atm["rho"]
    mach = atm["mach"]
    spi = deck.look_up("spi_vs_throttle_mach", THROTTLE0, mach)
    ca = deck.look_up("ca_vs_alpha_mach", ALPHAX, mach)
    denom = 0.029 * spi * AGRAV * rho * DVBE * ca * ACOWL
    throttle = THROTTLE0
    if denom != 0:
        thrst_req = AREA * CD * QHOLD / cos(ALPHAX * RAD)
        throtl_req = thrst_req / denom
        gainq = 2 * MASS0 / (rho * DVBE * denom * TQ)
        ethrotl = gainq * (QHOLD - pdynmc)
        throttle = ethrotl + throtl_req
    if throttle < 0:
        throttle = THRTL_IDLE
    if throttle > THRTL_MAX:
        throttle = THRTL_MAX
    spi = deck.look_up("spi_vs_throttle_mach", throttle, mach)
    thrust = spi * 0.029 * throttle * AGRAV * rho * DVBE * ca * ACOWL
    fmassd_next = thrust / (spi * AGRAV)
    fmasse = integrate(fmassd_next, 0.0, 0.0, INT_STEP)
    return throttle, spi, ca, thrust, fmassd_next, fmasse


def test_mprop_2_clips_max_throttle_and_thrust():
    deck = _deck()
    vehicle = _Vehicle()
    prop = Cruise3Propulsion(deck)
    prop.define(vehicle)
    atm = _auto_externals(vehicle.store, pdynmc=_atm()["pdynmc"])
    _auto_prop_data(vehicle.store)
    prop.initialize(vehicle, _ctx())

    throttle, spi, ca, thrust, fmassd_next, fmasse = _expected_auto(
        deck, atm, atm["pdynmc"]
    )
    assert throttle == THRTL_MAX

    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("throttle") == throttle
    assert store.get("spi") == spi
    assert store.get("ca") == ca
    assert store.get("thrust") == thrust
    assert store.get("fmassd") == fmassd_next
    assert store.get("fmasse") == fmasse
    assert store.get("mass") == MASS0 - fmasse
    assert store.get("fmassr") == FMASS0 - fmasse
    assert store.get("mprop") == 2


def test_mprop_2_clips_idle_throttle_and_thrust():
    deck = _deck()
    vehicle = _Vehicle()
    prop = Cruise3Propulsion(deck)
    prop.define(vehicle)
    pdynmc = 80000.0
    atm = _auto_externals(vehicle.store, pdynmc=pdynmc)
    _auto_prop_data(vehicle.store)
    prop.initialize(vehicle, _ctx())

    throttle, spi, ca, thrust, fmassd_next, fmasse = _expected_auto(
        deck, atm, pdynmc
    )
    assert throttle == THRTL_IDLE

    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("throttle") == throttle
    assert store.get("spi") == spi
    assert store.get("ca") == ca
    assert store.get("thrust") == thrust
    assert store.get("fmassd") == fmassd_next
    assert store.get("fmasse") == fmasse
    assert store.get("mprop") == 2
