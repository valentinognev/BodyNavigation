from math import acos, atan2, cos, fabs, sin, tan
from pathlib import Path

import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.sam6.rocket import Sam6RocketAero

SAM6 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/SAM6_250217/SAM6"
AERO = SAM6 / "SRBM_aero_deck.asc"

RTOL = 1e-12
ATOL = 1e-14
SMALL = 1e-7

ALPHAX = 5.0
MACH = 0.5
MPROP = 1
BETAX = 0.0
MASS = 6000.0
GRAV = 9.81
PDYNMC = 50000.0
ALPMAX = 20.0

DEFINED = (
    "area",
    "alpha_t0x",
    "beta_t0x",
    "alpmax",
    "alppx",
    "phipx",
    "cnptgt",
    "cltgt",
    "cdtgt",
    "catgt",
    "cytgt",
    "cntgt",
    "cnalp",
    "cybet",
    "gmax",
)
ROLES = {
    "area": "data",
    "alpha_t0x": "data",
    "beta_t0x": "data",
    "alpmax": "data",
    "alppx": "diag",
    "phipx": "diag",
    "cnptgt": "diag",
    "cltgt": "out",
    "cdtgt": "out",
    "catgt": "out",
    "cytgt": "out",
    "cntgt": "out",
    "cnalp": "out",
    "cybet": "out",
    "gmax": "out",
}
DEFAULTS = {
    "area": 0.636,
    "alpha_t0x": 0.0,
    "beta_t0x": 0.0,
    "alpmax": 0.0,
    "alppx": 0.0,
    "phipx": 0.0,
    "cnptgt": 0.0,
    "cltgt": 0.0,
    "cdtgt": 0.0,
    "catgt": 0.0,
    "cytgt": 0.0,
    "cntgt": 0.0,
    "cnalp": 0.0,
    "cybet": 0.0,
    "gmax": 0.0,
}
NOT_DEFINED = (
    "mach",
    "alphax",
    "betax",
    "mprop",
    "mass",
    "grav",
    "pdynmc",
    "vmach",
    "ca",
    "cn",
)


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


def _deck():
    _, tables = parse_asc_deck(AERO)
    return Datadeck.from_tables(tables)


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _plant(
    store,
    *,
    alphax=ALPHAX,
    betax=BETAX,
    mach=MACH,
    mprop=MPROP,
    mass=MASS,
    grav=GRAV,
    pdynmc=PDYNMC,
    alpmax=ALPMAX,
):
    store.define(Field("alphax", alphax, "real", "out", "control"))
    store.define(Field("betax", betax, "real", "out", "control"))
    store.define(Field("mach", mach, "real", "out", "environment"))
    store.define(Field("mprop", mprop, "int", "out", "propulsion"))
    store.define(Field("mass", mass, "real", "out", "propulsion"))
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.set("alpmax", alpmax)


def _ready(
    *,
    alphax=ALPHAX,
    betax=BETAX,
    mach=MACH,
    mprop=MPROP,
    mass=MASS,
    grav=GRAV,
    pdynmc=PDYNMC,
    alpmax=ALPMAX,
):
    deck = _deck()
    vehicle = _Vehicle()
    aero = Sam6RocketAero(deck)
    aero.define(vehicle)
    aero.initialize(vehicle, _ctx())
    _plant(
        vehicle.store,
        alphax=alphax,
        betax=betax,
        mach=mach,
        mprop=mprop,
        mass=mass,
        grav=grav,
        pdynmc=pdynmc,
        alpmax=alpmax,
    )
    return deck, vehicle, aero


def _expected(deck, store):
    area = store.get("area")
    alpmax = store.get("alpmax")
    grav = store.get("grav")
    pdynmc = store.get("pdynmc")
    mach = store.get("mach")
    mprop = store.get("mprop")
    mass = store.get("mass")
    alphax = store.get("alphax")
    betax = store.get("betax")
    alpha = alphax * RAD
    beta = betax * RAD
    alpp = acos(cos(alpha) * cos(beta))
    phip = 0.0
    dum1 = tan(beta)
    dum2 = sin(alpha)
    if dum1 * dum1 > SMALL and dum2 * dum2 > SMALL:
        phip = atan2(dum1, dum2)
    alppx = alpp * DEG
    phipx = phip * DEG
    cltgt = deck.look_up("cltgt_vs_alpha_mach", alppx, mach)
    cdtgt = deck.look_up("cdtgt_vs_alpha_mach", alppx, mach)
    cos_alpha = cos(alpha)
    sin_alpha = sin(alpha)
    catgt = cdtgt * cos_alpha - cltgt * sin_alpha
    if mprop == 0:
        catgt = catgt * 1.1
    cnptgt = cdtgt * sin_alpha + cltgt * cos_alpha
    cntgt = fabs(cnptgt) * cos(phip)
    cytgt = -fabs(cnptgt) * sin(phip)
    cltgt_max = deck.look_up("cltgt_vs_alpha_mach", alpmax, mach)
    cdtgt_max = deck.look_up("cdtgt_vs_alpha_mach", alpmax, mach)
    cnp_max = cdtgt_max * sin(alpmax * RAD) + cltgt_max * cos(alpmax * RAD)
    gmax = cnp_max * pdynmc * area / (mass * grav)
    return {
        "alppx": alppx,
        "phipx": phipx,
        "cltgt": cltgt,
        "cdtgt": cdtgt,
        "catgt": catgt,
        "cnptgt": cnptgt,
        "cntgt": cntgt,
        "cytgt": cytgt,
        "gmax": gmax,
    }


def test_name_is_aerodynamics():
    assert Sam6RocketAero(_deck()).name == "aerodynamics"


def test_define_cpp_fields():
    vehicle = _Vehicle()
    Sam6RocketAero(_deck()).define(vehicle)
    store = vehicle.store
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        assert field.module == "aerodynamics", name
        assert field.role == ROLES[name], name
        assert field.outputs == ()
        assert field.type == "real"
        assert _approx(store.get(name), DEFAULTS[name]), name
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_initialize_sets_cnalp():
    vehicle = _Vehicle()
    aero = Sam6RocketAero(_deck())
    aero.define(vehicle)
    assert aero.initialize(vehicle, _ctx()) is None
    assert _approx(vehicle.store.get("cnalp"), 7.468)
    assert _approx(vehicle.store.get("cybet"), -7.468)


def test_cltgt_matches_lookup_at_alpha5_mach05():
    deck, vehicle, aero = _ready(alphax=5.0, mach=0.5, mprop=1)
    aero.execute(vehicle, _ctx())
    want = _expected(deck, vehicle.store)
    assert _approx(vehicle.store.get("cltgt"), want["cltgt"])
    assert _approx(
        vehicle.store.get("cltgt"),
        deck.look_up("cltgt_vs_alpha_mach", want["alppx"], 0.5),
    )


def test_cdtgt_matches_lookup():
    deck, vehicle, aero = _ready(alphax=5.0, mach=0.5, mprop=1)
    aero.execute(vehicle, _ctx())
    want = _expected(deck, vehicle.store)
    assert _approx(vehicle.store.get("cdtgt"), want["cdtgt"])
    assert _approx(
        vehicle.store.get("cdtgt"),
        deck.look_up("cdtgt_vs_alpha_mach", want["alppx"], 0.5),
    )


def test_mprop_zero_scales_catgt():
    deck, burning, aero_on = _ready(mprop=1)
    aero_on.execute(burning, _ctx())
    cat_on = burning.store.get("catgt")
    _, coast, aero_off = _ready(mprop=0)
    aero_off.execute(coast, _ctx())
    assert _approx(coast.store.get("catgt"), cat_on * 1.1)
    want = _expected(deck, coast.store)
    assert _approx(coast.store.get("catgt"), want["catgt"])


def test_execute_matches_cpp_body_and_gmax():
    deck, vehicle, aero = _ready()
    aero.execute(vehicle, _ctx())
    want = _expected(deck, vehicle.store)
    for name in want:
        assert _approx(vehicle.store.get(name), want[name]), name


def test_sideslip_sets_phip_and_cytgt():
    deck, vehicle, aero = _ready(alphax=8.0, betax=4.0, mach=1.05)
    aero.execute(vehicle, _ctx())
    want = _expected(deck, vehicle.store)
    assert abs(want["phipx"]) > 1.0
    assert _approx(vehicle.store.get("phipx"), want["phipx"])
    assert _approx(vehicle.store.get("cytgt"), want["cytgt"])
    assert _approx(vehicle.store.get("cntgt"), want["cntgt"])


def test_parses_srbm_aero_deck():
    _, tables = parse_asc_deck(AERO)
    names = {table.name for table in tables}
    assert "cltgt_vs_alpha_mach" in names
    assert "cdtgt_vs_alpha_mach" in names
    deck = Datadeck.from_tables(tables)
    cltgt = deck.look_up("cltgt_vs_alpha_mach", 5.0, 0.5)
    cdtgt = deck.look_up("cdtgt_vs_alpha_mach", 5.0, 0.5)
    assert np.isfinite(cltgt)
    assert np.isfinite(cdtgt)


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.sam6.rocket as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "Plane5" not in src
    assert "Plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src


def test_terminate_is_pass():
    _, vehicle, aero = _ready()
    assert aero.terminate(vehicle, _ctx()) is None
    assert _approx(vehicle.store.get("cltgt"), 0.0)
