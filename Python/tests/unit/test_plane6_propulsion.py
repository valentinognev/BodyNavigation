from math import cos
from pathlib import Path

import cadac.constants as cadac_constants
import pytest

from cadac.constants import RAD
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.plane6.propulsion import Plane6Propulsion

FALCON6 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/FALCON6_250201/FALCON6"
PROP = FALCON6 / "f16_prop_deck.asc"

RTOL = 1e-12
ATOL = 1e-14

# C++ global_constants.hpp — duplicated here, not imported from production
FOOT = 3.280834
NT = 4.448

VMACHCOM = 0.6
GMACH = 30.0
VMACH = 0.58
HBE = 1000.0
ALPHAX = 1.0
PDYNMC = 17000.0
REFA = 27.87
CDRAG = 0.05
DT = 0.001

DEFINED = (
    "mprop",
    "vmachcom",
    "throttle",
    "gmach",
    "thrustf",
    "thrust",
    "thrust_req",
    "mfreeze_prop",
    "powerd",
    "power",
    "power_com",
    "tpower",
    "idle",
    "mil",
    "max",
)
INT_FIELDS = ("mprop", "mfreeze_prop")
EXTERNALS = ("vmach", "pdynmc", "alphax", "refa", "cdrag", "hbe", "time", "mfreeze")
FUEL_FIELDS = ("ff", "fmasse", "fmassed", "fuelmass", "mass")


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


def _deck():
    _, tables = parse_asc_deck(PROP)
    return Datadeck.from_tables(tables)


def _externals(
    store,
    *,
    vmach=VMACH,
    pdynmc=PDYNMC,
    alphax=ALPHAX,
    refa=REFA,
    cdrag=CDRAG,
    hbe=HBE,
    mfreeze=None,
):
    store.define(Field("time", 0.0, "real", "exec", "newton"))
    store.define(Field("vmach", vmach, "real", "out", "environment"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("alphax", alphax, "real", "diag", "kinematics"))
    store.define(Field("refa", refa, "real", "init", "aerodynamics"))
    store.define(Field("cdrag", cdrag, "real", "out", "aerodynamics"))
    store.define(Field("hbe", hbe, "real", "out", "newton"))
    if mfreeze is not None:
        store.define(Field("mfreeze", mfreeze, "int", "data", "control"))


def _ready(*, mprop=2, throttle=0.0, vmach=VMACH, mfreeze=None, **kw):
    deck = _deck()
    vehicle = _Vehicle()
    prop = Plane6Propulsion(deck)
    prop.define(vehicle)
    _externals(vehicle.store, vmach=vmach, mfreeze=mfreeze, **kw)
    store = vehicle.store
    store.set("mprop", mprop)
    store.set("vmachcom", VMACHCOM)
    store.set("gmach", GMACH)
    store.set("throttle", throttle)
    prop.initialize(vehicle, _ctx())
    return deck, vehicle, prop


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _propulsion_thrust(deck, store, throttle, dt):
    power = store.get("power")
    powerd = store.get("powerd")
    if throttle <= 0.77:
        power_com = 64.94 * throttle
    else:
        power_com = 217.38 * throttle - 117.38
    tpower = 1.0 if power_com <= 50 else 0.2
    powerd_new = (power_com - power) / tpower
    power = integrate(powerd_new, powerd, power, dt)
    hbe_ft = store.get("hbe") * FOOT
    vmach = store.get("vmach")
    idle = deck.look_up("idle_vs_mach_alt", vmach, hbe_ft) * NT
    mil = deck.look_up("mil_vs_mach_alt", vmach, hbe_ft) * NT
    max_thr = deck.look_up("max_vs_mach_alt", vmach, hbe_ft) * NT
    if power < 50:
        thrust = idle + power * 0.02 * (mil - idle)
    else:
        thrust = mil + (power - 50) * 0.02 * (max_thr - mil)
    return {
        "thrust": thrust,
        "power": power,
        "powerd": powerd_new,
        "power_com": power_com,
        "tpower": tpower,
        "idle": idle,
        "mil": mil,
        "max": max_thr,
    }


def _expected(deck, store, dt, mfreeze=None):
    mprop = store.get("mprop")
    throttle = store.get("throttle")
    thrustf = store.get("thrustf")
    mfreeze_prop = store.get("mfreeze_prop")
    thrust_req = 0.0
    pt = None
    if mprop == 1:
        pt = _propulsion_thrust(deck, store, throttle, dt)
        thrust = pt["thrust"]
    elif mprop == 2:
        throttle = store.get("gmach") * (store.get("vmachcom") - store.get("vmach"))
        if throttle < 0:
            throttle = 0.0
        if throttle > 0.77:
            throttle = 0.77
        pt = _propulsion_thrust(deck, store, throttle, dt)
        thrust = pt["thrust"]
        thrust_req = (
            store.get("cdrag")
            * store.get("pdynmc")
            * store.get("refa")
            / cos(store.get("alphax") * RAD)
        )
    else:
        thrust = 0.0
    if mfreeze is not None:
        if mfreeze == 0:
            mfreeze_prop = 0
        else:
            if mfreeze != mfreeze_prop:
                mfreeze_prop = mfreeze
                thrustf = thrust
            thrust = thrustf
    out = {
        "thrust": thrust,
        "thrust_req": thrust_req,
        "throttle": throttle,
        "thrustf": thrustf,
        "mfreeze_prop": mfreeze_prop,
    }
    if pt is not None:
        out.update(pt)
        out["thrust"] = thrust
        out["thrustf"] = thrustf
        out["mfreeze_prop"] = mfreeze_prop
    return out


def _assert_step(store, want, *, power_integrated):
    assert _approx(store.get("thrust"), want["thrust"])
    assert _approx(store.get("thrust_req"), want["thrust_req"])
    assert _approx(store.get("throttle"), want["throttle"])
    assert _approx(store.get("thrustf"), want["thrustf"])
    assert store.get("mfreeze_prop") == want["mfreeze_prop"]
    if power_integrated:
        for name in ("power", "powerd", "power_com", "tpower", "idle", "mil", "max"):
            assert _approx(store.get(name), want[name])


def test_name_is_propulsion():
    assert Plane6Propulsion(_deck()).name == "propulsion"


def test_define_registers_cpp_fields_not_externals():
    vehicle = _Vehicle()
    Plane6Propulsion(_deck()).define(vehicle)
    store = vehicle.store
    for name in DEFINED:
        assert name in store.names()
        if name in INT_FIELDS:
            assert store.get(name) == 0
            assert store.field(name).type == "int"
        else:
            assert store.get(name) == 0.0
            assert store.field(name).type == "real"
    for name in EXTERNALS:
        assert name not in store.names()
    for name in FUEL_FIELDS:
        assert name not in store.names()
    assert store.field("throttle").role == "data"
    assert store.field("throttle").outputs == ("scrn", "plot")
    assert store.field("thrust").role == "out"
    assert store.field("thrust").outputs == ("scrn", "plot")
    assert store.field("power").role == "state"
    assert store.field("power").outputs == ("plot",)
    assert store.field("idle").outputs == ("plot",)
    assert store.field("mil").outputs == ("plot",)
    assert store.field("max").outputs == ("plot",)
    assert store.field("powerd").role == "state"
    assert store.field("thrustf").role == "save"
    assert store.field("mfreeze_prop").role == "save"


def test_foot_and_nt_not_in_cadac_constants():
    assert not hasattr(cadac_constants, "FOOT")
    assert not hasattr(cadac_constants, "NT")


def test_initialize_leaves_power_zero():
    deck, vehicle, prop = _ready(mprop=2)
    store = vehicle.store
    assert store.get("power") == 0.0
    assert store.get("powerd") == 0.0
    assert store.get("thrust") == 0.0


def test_mprop_2_one_step_throttle_in_unit_interval():
    _, vehicle, prop = _ready(mprop=2, vmach=0.58)
    prop.execute(vehicle, _ctx(0.001))
    throttle = vehicle.store.get("throttle")
    assert 0 < throttle <= 1


def test_mprop_2_matches_cadac_formulas():
    deck, vehicle, prop = _ready(mprop=2, vmach=0.58)
    want = _expected(deck, vehicle.store, DT)
    assert want["throttle"] == pytest.approx(0.6, rel=RTOL, abs=ATOL)
    prop.execute(vehicle, _ctx())
    _assert_step(vehicle.store, want, power_integrated=True)
    assert 0 < vehicle.store.get("throttle") <= 1


def test_mprop_2_clips_throttle_high_to_military():
    deck, vehicle, prop = _ready(mprop=2, vmach=0.50)
    want = _expected(deck, vehicle.store, DT)
    assert want["throttle"] == 0.77
    prop.execute(vehicle, _ctx())
    assert vehicle.store.get("throttle") == 0.77
    _assert_step(vehicle.store, want, power_integrated=True)


def test_mprop_2_clips_throttle_low_to_zero():
    deck, vehicle, prop = _ready(mprop=2, vmach=0.70)
    want = _expected(deck, vehicle.store, DT)
    assert want["throttle"] == 0.0
    prop.execute(vehicle, _ctx())
    assert vehicle.store.get("throttle") == 0.0
    _assert_step(vehicle.store, want, power_integrated=True)


def test_mprop_1_uses_store_throttle_no_clip():
    deck, vehicle, prop = _ready(mprop=1, throttle=0.9)
    want = _expected(deck, vehicle.store, DT)
    assert want["throttle"] == 0.9
    assert want["power_com"] == pytest.approx(217.38 * 0.9 - 117.38, rel=RTOL, abs=ATOL)
    assert want["tpower"] == 0.2
    prop.execute(vehicle, _ctx())
    _assert_step(vehicle.store, want, power_integrated=True)
    assert vehicle.store.get("thrust_req") == 0.0


def test_mprop_1_military_power_com_and_tpower():
    deck, vehicle, prop = _ready(mprop=1, throttle=0.6)
    want = _expected(deck, vehicle.store, DT)
    assert want["power_com"] == pytest.approx(64.94 * 0.6, rel=RTOL, abs=ATOL)
    assert want["tpower"] == 1.0
    assert want["power"] < 50
    prop.execute(vehicle, _ctx())
    _assert_step(vehicle.store, want, power_integrated=True)


def test_thrust_blend_above_fifty_percent_power():
    deck, vehicle, prop = _ready(mprop=1, throttle=0.9)
    vehicle.store.set("power", 60.0)
    want = _expected(deck, vehicle.store, DT)
    assert want["power"] >= 50
    prop.execute(vehicle, _ctx())
    _assert_step(vehicle.store, want, power_integrated=True)


def test_mprop_0_zeros_thrust_without_integrating_power():
    _, vehicle, prop = _ready(mprop=0, throttle=0.5)
    store = vehicle.store
    store.set("power", 12.3)
    store.set("powerd", 4.5)
    store.set("idle", 7.0)
    store.set("thrust_req", 99.0)
    prop.execute(vehicle, _ctx())
    assert store.get("thrust") == 0.0
    assert store.get("thrust_req") == 0.0
    assert store.get("throttle") == 0.5
    assert store.get("power") == 12.3
    assert store.get("powerd") == 4.5
    assert store.get("idle") == 7.0


def test_mprop_else_zeros_thrust_without_integrating_power():
    _, vehicle, prop = _ready(mprop=3, throttle=0.5)
    store = vehicle.store
    store.set("power", 8.0)
    prop.execute(vehicle, _ctx())
    assert store.get("thrust") == 0.0
    assert store.get("power") == 8.0
    assert store.get("throttle") == 0.5


def test_idle_mil_max_converted_lb_to_newton():
    deck, vehicle, prop = _ready(mprop=1, throttle=0.6)
    hbe_ft = HBE * FOOT
    idle_lb = deck.look_up("idle_vs_mach_alt", VMACH, hbe_ft)
    mil_lb = deck.look_up("mil_vs_mach_alt", VMACH, hbe_ft)
    max_lb = deck.look_up("max_vs_mach_alt", VMACH, hbe_ft)
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("idle"), idle_lb * NT)
    assert _approx(store.get("mil"), mil_lb * NT)
    assert _approx(store.get("max"), max_lb * NT)


def test_altitude_lookup_uses_feet():
    deck, vehicle, prop = _ready(mprop=1, throttle=0.6, hbe=HBE)
    wrong_idle = deck.look_up("idle_vs_mach_alt", VMACH, HBE) * NT
    want = _expected(deck, vehicle.store, DT)
    prop.execute(vehicle, _ctx())
    assert vehicle.store.get("idle") != pytest.approx(wrong_idle, rel=RTOL, abs=ATOL)
    assert _approx(vehicle.store.get("idle"), want["idle"])


def test_stored_slope_second_step():
    deck, vehicle, prop = _ready(mprop=1, throttle=0.6)
    ctx = _ctx()
    want1 = _expected(deck, vehicle.store, DT)
    prop.execute(vehicle, ctx)
    _assert_step(vehicle.store, want1, power_integrated=True)
    want2 = _expected(deck, vehicle.store, DT)
    prop.execute(vehicle, ctx)
    _assert_step(vehicle.store, want2, power_integrated=True)
    assert vehicle.store.get("power") != want1["power"]


def test_skip_mfreeze_when_not_on_store():
    deck, vehicle, prop = _ready(mprop=2, vmach=0.58)
    want = _expected(deck, vehicle.store, DT, mfreeze=None)
    prop.execute(vehicle, _ctx())
    assert "mfreeze" not in vehicle.store.names()
    _assert_step(vehicle.store, want, power_integrated=True)
    assert vehicle.store.get("thrust") != 0.0


def test_mfreeze_latches_thrust_when_present():
    deck, vehicle, prop = _ready(mprop=1, throttle=0.6, mfreeze=1)
    want1 = _expected(deck, vehicle.store, DT, mfreeze=1)
    prop.execute(vehicle, _ctx())
    live = vehicle.store.get("thrust")
    assert vehicle.store.get("mfreeze_prop") == 1
    assert _approx(vehicle.store.get("thrustf"), live)
    _assert_step(vehicle.store, want1, power_integrated=True)
    vehicle.store.set("throttle", 0.9)
    want2 = _expected(deck, vehicle.store, DT, mfreeze=1)
    prop.execute(vehicle, _ctx())
    assert _approx(vehicle.store.get("thrust"), live)
    _assert_step(vehicle.store, want2, power_integrated=True)


def test_mfreeze_zero_clears_latch():
    _, vehicle, prop = _ready(mprop=1, throttle=0.6, mfreeze=1)
    prop.execute(vehicle, _ctx())
    held = vehicle.store.get("thrust")
    vehicle.store.set("mfreeze", 0)
    vehicle.store.set("throttle", 0.9)
    prop.execute(vehicle, _ctx())
    assert vehicle.store.get("mfreeze_prop") == 0
    assert vehicle.store.get("thrust") != pytest.approx(held, rel=RTOL, abs=ATOL)


def test_thrust_req_uses_rad_from_constants():
    deck, vehicle, prop = _ready(mprop=2, vmach=0.58, alphax=5.0)
    want = (
        CDRAG * PDYNMC * REFA / cos(5.0 * RAD)
    )
    prop.execute(vehicle, _ctx())
    assert _approx(vehicle.store.get("thrust_req"), want)
    wrong = CDRAG * PDYNMC * REFA / cos(5.0)
    assert vehicle.store.get("thrust_req") != pytest.approx(wrong, rel=1e-6)
