from pathlib import Path

import numpy as np
import pytest

from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.sam6.propulsion import Sam6Propulsion

SAM6 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/SAM6_250217/SAM6"
PROP = SAM6 / "SAM_prop_deck.asc"

RTOL = 1e-12
ATOL = 1e-14
PSL = 101325.0
AEXIT = 0.0314

DEFINED = (
    "mprop",
    "aexit",
    "mass",
    "thrust",
    "xcgref",
    "xcg",
    "ai11",
    "ai33",
    "mfreeze_prop",
    "thrustf",
    "massf",
    "xcgf",
    "ai11f",
    "ai33f",
)
ROLES = {
    "mprop": "out",
    "aexit": "data",
    "mass": "out",
    "thrust": "out",
    "xcgref": "data",
    "xcg": "diag",
    "ai11": "out",
    "ai33": "out",
    "mfreeze_prop": "save",
    "thrustf": "save",
    "massf": "save",
    "xcgf": "save",
    "ai11f": "save",
    "ai33f": "save",
}
OUTPUTS = {
    "mprop": (),
    "aexit": (),
    "mass": ("plot", "scrn"),
    "thrust": ("scrn", "plot"),
    "xcgref": (),
    "xcg": ("scrn", "plot"),
    "ai11": ("plot",),
    "ai33": ("plot",),
    "mfreeze_prop": (),
    "thrustf": (),
    "massf": (),
    "xcgf": (),
    "ai11f": (),
    "ai33f": (),
}
INT_FIELDS = ("mprop", "mfreeze_prop")
DEFAULTS = {
    "mprop": 0,
    "aexit": 0.0314,
    "mass": 300.0,
    "thrust": 0.0,
    "xcgref": 0.0,
    "xcg": 2.9,
    "ai11": 2.9,
    "ai33": 440.0,
    "mfreeze_prop": 0,
    "thrustf": 0.0,
    "massf": 0.0,
    "xcgf": 0.0,
    "ai11f": 0.0,
    "ai33f": 0.0,
}
NOT_DEFINED = ("msl_time", "press", "mfreeze")


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
    _, tables = parse_asc_deck(PROP)
    return Datadeck.from_tables(tables)


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _plant(store, *, msl_time, press, mfreeze=None):
    store.define(Field("msl_time", msl_time, "real", "out", "kinematics"))
    store.define(Field("press", press, "real", "out", "environment"))
    if mfreeze is not None:
        store.define(Field("mfreeze", mfreeze, "int", "data", "control"))


def _ready(*, msl_time=0.0, press=PSL, mfreeze=None, aexit=None):
    deck = _deck()
    vehicle = _Vehicle()
    prop = Sam6Propulsion(deck)
    prop.define(vehicle)
    prop.initialize(vehicle, _ctx())
    _plant(vehicle.store, msl_time=msl_time, press=press, mfreeze=mfreeze)
    if aexit is not None:
        vehicle.store.set("aexit", aexit)
    return deck, vehicle, prop


def _expected(deck, msl_time, press, aexit=AEXIT):
    tsl = deck.look_up("thrust_vs_time", msl_time)
    return {
        "thrust": tsl + (PSL - press) * aexit,
        "tsl": tsl,
        "mass": deck.look_up("mass_vs_time", msl_time),
        "xcg": deck.look_up("cg_vs_time", msl_time),
        "ai33": deck.look_up("moipitch_vs_time", msl_time),
        "ai11": deck.look_up("moiroll_vs_time", msl_time),
        "mprop": 1 if msl_time <= 60 else 0,
    }


def test_name_is_propulsion():
    assert Sam6Propulsion(_deck()).name == "propulsion"


def test_define_cpp_fields():
    vehicle = _Vehicle()
    Sam6Propulsion(_deck()).define(vehicle)
    store = vehicle.store
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        assert field.module == "propulsion"
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        if name in INT_FIELDS:
            assert field.type == "int"
            assert store.get(name) == DEFAULTS[name]
        else:
            assert field.type == "real"
            assert _approx(store.get(name), DEFAULTS[name]), name
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_initialize_is_pass():
    vehicle = _Vehicle()
    prop = Sam6Propulsion(_deck())
    prop.define(vehicle)
    before = {name: vehicle.store.get(name) for name in DEFINED}
    assert prop.initialize(vehicle, _ctx()) is None
    for name in DEFINED:
        if name in INT_FIELDS:
            assert vehicle.store.get(name) == before[name]
        else:
            assert _approx(vehicle.store.get(name), before[name]), name


def test_sea_level_t0_mass_mprop_thrust():
    deck, vehicle, prop = _ready(msl_time=0.0, press=PSL)
    tsl = deck.look_up("thrust_vs_time", 0.0)
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("mass"), 300.0)
    assert store.get("mprop") == 1
    assert _approx(store.get("thrust"), tsl)
    assert _approx(tsl, 0.0)
    assert _approx(store.get("thrust"), tsl + (PSL - PSL) * AEXIT)


def test_msl_time_61_mprop_off():
    _, vehicle, prop = _ready(msl_time=61.0, press=PSL)
    prop.execute(vehicle, _ctx())
    assert vehicle.store.get("mprop") == 0


def test_msl_time_60_mprop_on():
    _, vehicle, prop = _ready(msl_time=60.0, press=PSL)
    prop.execute(vehicle, _ctx())
    assert vehicle.store.get("mprop") == 1


def test_tables_and_back_pressure_match_cpp():
    press = 26500.0
    msl_time = 2.0
    deck, vehicle, prop = _ready(msl_time=msl_time, press=press)
    want = _expected(deck, msl_time, press)
    assert want["tsl"] != 0.0
    assert want["thrust"] != want["tsl"]
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("mprop") == 1
    assert _approx(store.get("thrust"), want["thrust"])
    assert _approx(store.get("mass"), want["mass"])
    assert _approx(store.get("xcg"), want["xcg"])
    assert _approx(store.get("ai33"), want["ai33"])
    assert _approx(store.get("ai11"), want["ai11"])
    assert _approx(store.get("mass"), 280.0)
    assert _approx(store.get("xcg"), 2.80)
    assert _approx(store.get("ai33"), 380.0)
    assert _approx(store.get("ai11"), 2.70)


def test_skip_mfreeze_when_absent():
    deck, vehicle, prop = _ready(msl_time=2.0, press=PSL)
    want = _expected(deck, 2.0, PSL)
    prop.execute(vehicle, _ctx())
    assert "mfreeze" not in vehicle.store.names()
    assert _approx(vehicle.store.get("thrust"), want["thrust"])
    assert np.isfinite(vehicle.store.get("thrust"))


def test_mfreeze_latches_when_present():
    deck, vehicle, prop = _ready(msl_time=2.0, press=PSL, mfreeze=1)
    want1 = _expected(deck, 2.0, PSL)
    prop.execute(vehicle, _ctx())
    live = vehicle.store.get("thrust")
    assert vehicle.store.get("mfreeze_prop") == 1
    assert _approx(vehicle.store.get("thrustf"), live)
    assert _approx(live, want1["thrust"])
    vehicle.store.set("msl_time", 10.0)
    want_live = _expected(deck, 10.0, PSL)
    assert want_live["thrust"] != pytest.approx(live, rel=RTOL, abs=ATOL)
    prop.execute(vehicle, _ctx())
    assert _approx(vehicle.store.get("thrust"), live)
    assert _approx(vehicle.store.get("mass"), want1["mass"])
    assert _approx(vehicle.store.get("xcg"), want1["xcg"])
    assert _approx(vehicle.store.get("ai11"), want1["ai11"])
    assert _approx(vehicle.store.get("ai33"), want1["ai33"])


def test_mfreeze_zero_clears_latch():
    deck, vehicle, prop = _ready(msl_time=2.0, press=PSL, mfreeze=1)
    prop.execute(vehicle, _ctx())
    held = vehicle.store.get("thrust")
    vehicle.store.set("mfreeze", 0)
    vehicle.store.set("msl_time", 10.0)
    want = _expected(deck, 10.0, PSL)
    prop.execute(vehicle, _ctx())
    assert vehicle.store.get("mfreeze_prop") == 0
    assert vehicle.store.get("thrust") != pytest.approx(held, rel=RTOL, abs=ATOL)
    assert _approx(vehicle.store.get("thrust"), want["thrust"])


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.sam6.propulsion as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src


def test_terminate_exists_and_is_pass():
    _, vehicle, prop = _ready()
    assert prop.terminate(vehicle, _ctx()) is None
