from math import cos, isfinite
from pathlib import Path

from cadac.constants import RAD
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.flat3.falcon5.propulsion import Plane5Propulsion

FALCON5 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/FALCON5_250116/FALCON5"
PROP = FALCON5 / "Falcon5_prop_deck.asc"

MASS_INIT = 12701.0
FUEL_INIT = 4461.0
GFTHM = 893620.0
TFTH = 1.0
MACH_COM = 0.6
ALT = 3500.0
MACH = 0.6
AREA = 27.87
CD = 0.05
ALPHAX = 5.0
PDYNMC = 17000.0
INT_STEP = 0.05
THRUST_COM = 10000.0

DEFINED = (
    "mprop",
    "fidle",
    "thrust_com",
    "thrust",
    "treqd",
    "treq",
    "fmassed",
    "fmasse",
    "fuelmass",
    "mach_com",
    "gfthm",
    "tfth",
    "mass",
    "tav",
    "mass_init",
    "fuel_init",
    "ff",
)
EXTERNALS = ("pdynmc", "mach", "alt", "cd", "area", "alphax")


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


def _externals(store):
    store.define(Field("pdynmc", PDYNMC, "real", "out", "environment"))
    store.define(Field("mach", MACH, "real", "out", "environment"))
    store.define(Field("alt", ALT, "real", "out", "newton"))
    store.define(Field("cd", CD, "real", "out", "aerodynamics"))
    store.define(Field("area", AREA, "real", "data", "aerodynamics"))
    store.define(Field("alphax", ALPHAX, "real", "out", "control"))


def _turning_prop_data(store, *, mprop):
    store.set("mprop", mprop)
    store.set("mach_com", MACH_COM)
    store.set("mass_init", MASS_INIT)
    store.set("fuel_init", FUEL_INIT)
    store.set("gfthm", GFTHM)
    store.set("tfth", TFTH)
    store.set("thrust_com", THRUST_COM)


def _ready(mprop):
    deck = _deck()
    vehicle = _Vehicle()
    prop = Plane5Propulsion(deck)
    prop.define(vehicle)
    _externals(vehicle.store)
    _turning_prop_data(vehicle.store, mprop=mprop)
    prop.initialize(vehicle, _ctx())
    return deck, vehicle, prop


def _mach_hold_expected(deck, store, dt):
    fidle = deck.look_up("fidle_vs_alt_mach", store.get("alt"), store.get("mach"))
    tav = deck.look_up("tav_vs_alt_mach", store.get("alt"), store.get("mach"))
    mprop = 4
    treqs = store.get("cd") * store.get("pdynmc") * store.get("area")
    epsmch = store.get("mach_com") - store.get("mach")
    tcom = epsmch * store.get("gfthm") + treqs
    treq = store.get("treq")
    treqd = store.get("treqd")
    treqd_new = (tcom - 2.0 * treq) / store.get("tfth")
    treq = integrate(treqd_new, treqd, treq, dt)
    treqb = treq / cos(store.get("alphax") * RAD)
    if treqb < fidle:
        mprop = 5
        treqb = fidle
    if treqb > tav:
        mprop = 6
        treqb = tav
    thrust = treqb
    ff = deck.look_up("ff_vs_thrust_alt_mach", thrust, store.get("alt"), store.get("mach"))
    fmasse = integrate(ff, store.get("fmassed"), store.get("fmasse"), dt)
    mass = store.get("mass_init") - fmasse
    fuelmass = store.get("fuel_init") - fmasse
    if fuelmass <= 0:
        thrust = 0.0
    return {
        "thrust": thrust,
        "mass": mass,
        "mprop": mprop,
        "treq": treq,
        "treqd": treqd_new,
        "fmasse": fmasse,
        "fmassed": ff,
        "fuelmass": fuelmass,
        "ff": ff,
        "fidle": fidle,
        "tav": tav,
    }


def test_name_is_propulsion():
    assert Plane5Propulsion(_deck()).name == "propulsion"


def test_define_registers_cpp_fields_not_externals():
    vehicle = _Vehicle()
    Plane5Propulsion(_deck()).define(vehicle)
    store = vehicle.store
    assert store.get("mprop") == 0
    assert store.field("mprop").type == "int"
    for name in DEFINED:
        if name == "mprop":
            continue
        assert store.get(name) == 0.0
    for name in EXTERNALS:
        assert name not in store.names()
    assert "cg" not in store.names()


def test_initialize_sets_mass_from_mass_init():
    vehicle = _Vehicle()
    prop = Plane5Propulsion(_deck())
    prop.define(vehicle)
    vehicle.store.set("mass_init", MASS_INIT)
    prop.initialize(vehicle, _ctx())
    assert vehicle.store.get("mass") == MASS_INIT


def test_mprop_0_thrust_stays_zero():
    _, vehicle, prop = _ready(mprop=0)
    prop.execute(vehicle, _ctx())
    assert vehicle.store.get("thrust") == 0.0


def test_mprop_0_returns_without_writing():
    _, vehicle, prop = _ready(mprop=0)
    store = vehicle.store
    store.set("thrust", 999.0)
    store.set("ff", 1.0)
    store.set("mass", 100.0)
    store.set("fmasse", 50.0)
    store.set("fuelmass", 10.0)
    store.set("fidle", 3.0)
    store.set("tav", 4.0)
    prop.execute(vehicle, _ctx())
    assert store.get("thrust") == 999.0
    assert store.get("ff") == 1.0
    assert store.get("mass") == 100.0
    assert store.get("fmasse") == 50.0
    assert store.get("fuelmass") == 10.0
    assert store.get("fidle") == 3.0
    assert store.get("tav") == 4.0
    assert store.get("mprop") == 0


def test_mprop_4_one_step_thrust_and_mass_finite():
    _, vehicle, prop = _ready(mprop=4)
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert isfinite(store.get("thrust"))
    assert isfinite(store.get("mass"))


def test_mprop_4_mach_hold_matches_cpp():
    deck, vehicle, prop = _ready(mprop=4)
    want = _mach_hold_expected(deck, vehicle.store, INT_STEP)
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("thrust") == want["thrust"]
    assert store.get("mass") == want["mass"]
    assert store.get("mprop") == want["mprop"]
    assert store.get("treq") == want["treq"]
    assert store.get("treqd") == want["treqd"]
    assert store.get("fmasse") == want["fmasse"]
    assert store.get("fmassed") == want["fmassed"]
    assert store.get("fuelmass") == want["fuelmass"]
    assert store.get("ff") == want["ff"]
    assert store.get("fidle") == want["fidle"]
    assert store.get("tav") == want["tav"]


def test_mprop_gt_3_forces_mach_hold():
    deck, vehicle, prop = _ready(mprop=7)
    want = _mach_hold_expected(deck, vehicle.store, INT_STEP)
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("mprop") == want["mprop"]
    assert store.get("thrust") == want["thrust"]
    assert store.get("mass") == want["mass"]


def test_mprop_1_commanded_thrust():
    deck, vehicle, prop = _ready(mprop=1)
    store = vehicle.store
    ff = deck.look_up("ff_vs_thrust_alt_mach", THRUST_COM, ALT, MACH)
    fmasse = integrate(ff, 0.0, 0.0, INT_STEP)
    prop.execute(vehicle, _ctx())
    assert store.get("thrust") == THRUST_COM
    assert store.get("treq") == THRUST_COM
    assert store.get("ff") == ff
    assert store.get("fmasse") == fmasse
    assert store.get("mass") == MASS_INIT - fmasse
    assert store.get("fuelmass") == FUEL_INIT - fmasse
    assert store.get("mprop") == 1


def test_mprop_2_idle():
    deck, vehicle, prop = _ready(mprop=2)
    store = vehicle.store
    fidle = deck.look_up("fidle_vs_alt_mach", ALT, MACH)
    ff = deck.look_up("iff_vs_alt", ALT)
    fmasse = integrate(ff, 0.0, 0.0, INT_STEP)
    prop.execute(vehicle, _ctx())
    assert store.get("thrust") == fidle
    assert store.get("fidle") == fidle
    assert store.get("ff") == ff
    assert store.get("fmasse") == fmasse
    assert store.get("mass") == MASS_INIT - fmasse
    assert store.get("mprop") == 2


def test_mprop_3_max():
    deck, vehicle, prop = _ready(mprop=3)
    store = vehicle.store
    tav = deck.look_up("tav_vs_alt_mach", ALT, MACH)
    ff = deck.look_up("ff_vs_thrust_alt_mach", tav, ALT, MACH)
    fmasse = integrate(ff, 0.0, 0.0, INT_STEP)
    prop.execute(vehicle, _ctx())
    assert store.get("thrust") == tav
    assert store.get("tav") == tav
    assert store.get("ff") == ff
    assert store.get("fmasse") == fmasse
    assert store.get("mass") == MASS_INIT - fmasse
    assert store.get("mprop") == 3


def test_mach_hold_clips_idle_sets_mprop_5():
    deck, vehicle, prop = _ready(mprop=4)
    store = vehicle.store
    store.set("mach", 1.0)
    want = _mach_hold_expected(deck, store, INT_STEP)
    assert want["mprop"] == 5
    prop.execute(vehicle, _ctx())
    assert store.get("mprop") == 5
    assert store.get("thrust") == want["thrust"]
    assert store.get("thrust") == store.get("fidle")


def test_mach_hold_clips_max_sets_mprop_6():
    deck, vehicle, prop = _ready(mprop=4)
    store = vehicle.store
    store.set("treq", 1.0e6)
    want = _mach_hold_expected(deck, store, INT_STEP)
    assert want["mprop"] == 6
    prop.execute(vehicle, _ctx())
    assert store.get("mprop") == 6
    assert store.get("thrust") == want["thrust"]
    assert store.get("thrust") == store.get("tav")


def test_fuel_expended_zeros_thrust():
    _, vehicle, prop = _ready(mprop=3)
    store = vehicle.store
    store.set("fmasse", FUEL_INIT)
    prop.execute(vehicle, _ctx())
    assert store.get("fuelmass") <= 0.0
    assert store.get("thrust") == 0.0
    assert store.get("mprop") == 3
