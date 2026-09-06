from pathlib import Path

import pytest

from cadac.constants import AGRAV
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.hyper5.propulsion import Hyper5Propulsion

HYPER5 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/HYPER5_250113/HYPER5"
PROP = HYPER5 / "hyper5_prop_deck.asc"

RTOL = 1e-12
ATOL = 1e-14

MACH = 5.0
ALPHAX = 0.0
RHO = 0.0889
DVBE = 1475.0
PDYNMC = 72000.0
CA = 0.05
AREA = 11.6986
AINTAKE = 0.184
MASS0 = 1976.0
FMASS0 = 624.0
PHI_CONST = 1.0
PHI_MIN = 0.5
PHI_MAX = 1.2
QHOLD = 72000.0
TQ = 1.0
TLAG = 10.0
INT_STEP = 0.05

DEFINED = (
    "phi_const",
    "tlag",
    "phis",
    "phisd",
    "mprop",
    "aintake",
    "phi",
    "phi_max",
    "qhold",
    "mass",
    "mass0",
    "cin",
    "tq",
    "phi_min",
    "fmass0",
    "fmasse",
    "fmassd",
    "thrst_stoch",
    "spi",
    "thrust",
    "mass_flow",
    "fmassr",
    "thrst_req",
)
ROLES = {
    "phi_const": "data",
    "tlag": "data",
    "phis": "state",
    "phisd": "state",
    "mprop": "data",
    "aintake": "data",
    "phi": "data/diag",
    "phi_max": "data",
    "qhold": "data",
    "mass": "out",
    "mass0": "data",
    "cin": "diag",
    "tq": "data",
    "phi_min": "data",
    "fmass0": "data",
    "fmasse": "state",
    "fmassd": "state",
    "thrst_stoch": "diag",
    "spi": "diag",
    "thrust": "out",
    "mass_flow": "diag",
    "fmassr": "diag",
    "thrst_req": "diag",
}
OUTPUTS = {
    "phi_const": (),
    "tlag": (),
    "phis": ("plot",),
    "phisd": ("plot",),
    "mprop": ("plot",),
    "aintake": (),
    "phi": ("scrn", "plot"),
    "phi_max": (),
    "qhold": (),
    "mass": ("scrn", "plot"),
    "mass0": (),
    "cin": ("scrn", "plot"),
    "tq": (),
    "phi_min": (),
    "fmass0": (),
    "fmasse": (),
    "fmassd": (),
    "thrst_stoch": ("plot",),
    "spi": ("scrn", "plot"),
    "thrust": ("scrn", "plot"),
    "mass_flow": (),
    "fmassr": ("scrn", "plot"),
    "thrst_req": ("scrn", "plot"),
}
PLANT = ("rho", "pdynmc", "mach", "dvbe", "ca", "area", "alphax")
NOT_DEFINED = ("time", "rho", "pdynmc", "mach", "dvbe", "ca", "area", "alphax")


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


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _plant(store, **overrides):
    vals = {
        "rho": RHO,
        "pdynmc": PDYNMC,
        "mach": MACH,
        "dvbe": DVBE,
        "ca": CA,
        "area": AREA,
        "alphax": ALPHAX,
    }
    vals.update(overrides)
    store.define(Field("rho", vals["rho"], "real", "out", "environment"))
    store.define(Field("pdynmc", vals["pdynmc"], "real", "out", "environment"))
    store.define(Field("mach", vals["mach"], "real", "out", "environment"))
    store.define(Field("dvbe", vals["dvbe"], "real", "out", "newton"))
    store.define(Field("ca", vals["ca"], "real", "diag", "aerodynamics"))
    store.define(Field("area", vals["area"], "real", "data", "aerodynamics"))
    store.define(Field("alphax", vals["alphax"], "real", "out", "control"))


def _prop_data(store, *, mprop, **overrides):
    vals = {
        "phi_const": PHI_CONST,
        "tlag": TLAG,
        "aintake": AINTAKE,
        "phi": 0.0,
        "phi_max": PHI_MAX,
        "qhold": QHOLD,
        "mass0": MASS0,
        "tq": TQ,
        "phi_min": PHI_MIN,
        "fmass0": FMASS0,
        "fmasse": 0.0,
        "fmassd": 0.0,
        "phis": 0.0,
        "phisd": 0.0,
    }
    vals.update(overrides)
    store.set("mprop", mprop)
    for name, value in vals.items():
        store.set(name, value)


def _ready(deck, *, mprop, plant=None, **prop_kw):
    vehicle = _Vehicle()
    prop = Hyper5Propulsion(deck)
    prop.define(vehicle)
    _plant(vehicle.store, **(plant or {}))
    _prop_data(vehicle.store, mprop=mprop, **prop_kw)
    prop.initialize(vehicle, _ctx())
    return vehicle, prop


def _fuel(thrust, spi, fmasse, fmassd, dt):
    fmassd_next = thrust / (spi * AGRAV)
    fmasse = integrate(fmassd_next, fmassd, fmasse, dt)
    mass = MASS0 - fmasse
    fmassr = FMASS0 - fmasse
    mass_flow = thrust / (AGRAV * spi)
    return fmassd_next, fmasse, mass, fmassr, mass_flow


def test_name_is_propulsion():
    assert Hyper5Propulsion(None).name == "propulsion"


def test_define_registers_cpp_fields_not_plant_or_time():
    vehicle = _Vehicle()
    Hyper5Propulsion(None).define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    for name in DEFINED:
        field = store.field(name)
        assert field.role == ROLES[name]
        assert field.module == "propulsion"
        assert field.outputs == OUTPUTS[name]
        if name == "mprop":
            assert field.type == "int"
            assert store.get(name) == 0
        else:
            assert field.type == "real"
            assert store.get(name) == 0.0
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_initialize_sets_mass_from_mass0():
    vehicle = _Vehicle()
    prop = Hyper5Propulsion(None)
    prop.define(vehicle)
    vehicle.store.set("mass0", MASS0)
    prop.initialize(vehicle, _ctx())
    assert vehicle.store.get("mass") == MASS0


def test_mprop_0_zero_thrust_without_lookup():
    vehicle, prop = _ready(None, mprop=0, fmassd=1.0)
    store = vehicle.store
    store.set("phi", 0.8)
    prop.execute(vehicle, _ctx())
    assert store.get("thrust") == 0.0
    assert store.get("fmassd") == 0.0
    assert store.get("fmasse") == 0.0
    assert store.get("mass") == MASS0
    assert store.get("mprop") == 0
    assert store.get("cin") == 0.0
    assert store.get("spi") == 0.0
    assert store.get("phi") == 0.0
    assert store.get("mass_flow") == 0.0
    assert store.get("thrst_stoch") == 0.0
    assert store.get("thrst_req") == 0.0
    assert "time" not in store.names()


def test_mprop_0_boom_deck_does_not_look_up():
    vehicle, prop = _ready(_BoomDeck(), mprop=0)
    prop.execute(vehicle, _ctx())
    assert vehicle.store.get("thrust") == 0.0


@pytest.mark.parametrize("mprop", [-1, 4, 99])
def test_mprop_not_0_1_2_3_raises(mprop):
    vehicle = _Vehicle()
    prop = Hyper5Propulsion(None)
    prop.define(vehicle)
    vehicle.store.set("mprop", mprop)
    with pytest.raises(ValueError, match="unknown mprop"):
        prop.execute(vehicle, _ctx())


def test_mprop_1_one_step_matches_cpp_with_local_phi_zero_lookup():
    deck = _deck()
    vehicle, prop = _ready(deck, mprop=1, phi=0.8)
    store = vehicle.store

    phi_local = 0.0
    cin = deck.look_up("cin_vs_alphax_mach", ALPHAX, MACH)
    spi = deck.look_up("spi_vs_mach_phi_alphax", MACH, phi_local, ALPHAX)
    spi_at_const = deck.look_up("spi_vs_mach_phi_alphax", MACH, PHI_CONST, ALPHAX)
    assert spi != spi_at_const
    thrust = 0.0676 * PHI_CONST * spi * AGRAV * RHO * DVBE * cin * AINTAKE
    fmassd_next, fmasse, mass, fmassr, mass_flow = _fuel(
        thrust, spi, 0.0, 0.0, INT_STEP
    )

    prop.execute(vehicle, _ctx())

    assert _approx(store.get("cin"), cin)
    assert _approx(store.get("spi"), spi)
    assert _approx(store.get("thrust"), thrust)
    assert _approx(store.get("fmassd"), fmassd_next)
    assert _approx(store.get("fmasse"), fmasse)
    assert _approx(store.get("mass"), mass)
    assert _approx(store.get("fmassr"), fmassr)
    assert _approx(store.get("mass_flow"), mass_flow)
    assert store.get("mprop") == 1
    assert store.get("phi") == 0.0
    assert store.get("thrst_stoch") == 0.0
    assert store.get("thrst_req") == 0.0
    assert store.get("phi_const") == PHI_CONST
    assert "time" not in store.names()


def test_mprop_1_shuts_down_when_fuel_expended():
    deck = _deck()
    vehicle, prop = _ready(deck, mprop=1, fmasse=FMASS0)
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("fmassr") <= 0.0
    assert store.get("mprop") == 0
    assert store.get("thrust") == 0.0
    assert store.get("fmassd") == 0.0


def _expected_mprop2(
    deck,
    *,
    phis=0.0,
    phisd=0.0,
    fmasse=0.0,
    fmassd=0.0,
    mass=MASS0,
    pdynmc=PDYNMC,
    phi_min=PHI_MIN,
    phi_max=PHI_MAX,
    tlag=TLAG,
    dt=INT_STEP,
):
    phi = 0.0
    cin = deck.look_up("cin_vs_alphax_mach", ALPHAX, MACH)
    spi = deck.look_up("spi_vs_mach_phi_alphax", MACH, phi, ALPHAX)
    thrst_stoch = 0.0676 * spi * AGRAV * RHO * DVBE * cin * AINTAKE
    thrst_req = AREA * CA * QHOLD
    phi_req = thrst_req / thrst_stoch
    gainq = 2 * mass / (RHO * DVBE * thrst_stoch * TQ)
    ephi = gainq * (QHOLD - pdynmc)
    phi = phi_req + ephi
    phisd_new = (phi - phis) / tlag
    phis = integrate(phisd_new, phisd, phis, dt)
    phisd = phisd_new
    phi = phis
    if phi < phi_min:
        phi = phi_min
    if phi > phi_max:
        phi = phi_max
    spi = deck.look_up("spi_vs_mach_phi_alphax", MACH, phi / 0.0676, ALPHAX)
    thrust = 0.0676 * phi * spi * AGRAV * RHO * DVBE * cin * AINTAKE
    fmassd_next, fmasse, mass, fmassr, mass_flow = _fuel(
        thrust, spi, fmasse, fmassd, dt
    )
    return {
        "cin": cin,
        "spi": spi,
        "thrust": thrust,
        "phi": phi,
        "phis": phis,
        "phisd": phisd,
        "thrst_stoch": thrst_stoch,
        "thrst_req": thrst_req,
        "fmassd": fmassd_next,
        "fmasse": fmasse,
        "mass": mass,
        "fmassr": fmassr,
        "mass_flow": mass_flow,
    }


def test_mprop_2_qhold_lag_and_min_clip_vs_cpp():
    deck = _deck()
    vehicle, prop = _ready(deck, mprop=2)
    want = _expected_mprop2(deck)
    assert want["phis"] < PHI_MIN
    assert want["phi"] == PHI_MIN

    prop.execute(vehicle, _ctx())
    store = vehicle.store
    for name, value in want.items():
        assert _approx(store.get(name), value), name
    assert store.get("mprop") == 2
    assert store.get("phis") != store.get("phi")


def test_mprop_2_max_clip_keeps_unclipped_phis():
    deck = _deck()
    phis0 = 5.0
    tlag = 1.0e6
    vehicle, prop = _ready(deck, mprop=2, phis=phis0, tlag=tlag)
    want = _expected_mprop2(deck, phis=phis0, tlag=tlag)
    assert want["phis"] > PHI_MAX
    assert want["phi"] == PHI_MAX

    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("phis"), want["phis"])
    assert store.get("phi") == PHI_MAX
    assert _approx(store.get("spi"), want["spi"])
    assert _approx(store.get("thrust"), want["thrust"])


def test_mprop_2_second_spi_uses_phi_over_0_0676():
    deck = _deck()
    vehicle, prop = _ready(deck, mprop=2)
    want = _expected_mprop2(deck)
    spi_wrong = deck.look_up(
        "spi_vs_mach_phi_alphax", MACH, want["phi"], ALPHAX
    )
    assert want["spi"] != spi_wrong
    prop.execute(vehicle, _ctx())
    assert _approx(vehicle.store.get("spi"), want["spi"])


def test_mprop_2_stored_slope_second_step():
    deck = _deck()
    vehicle, prop = _ready(deck, mprop=2)
    first = _expected_mprop2(deck)
    prop.execute(vehicle, _ctx())
    second = _expected_mprop2(
        deck,
        phis=first["phis"],
        phisd=first["phisd"],
        fmasse=first["fmasse"],
        fmassd=first["fmassd"],
        mass=first["mass"],
    )
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("phis"), second["phis"])
    assert _approx(store.get("phisd"), second["phisd"])
    assert _approx(store.get("thrust"), second["thrust"])
    assert _approx(store.get("fmasse"), second["fmasse"])


def test_mprop_3_does_not_read_saved_phi():
    deck = _deck()
    vehicle, prop = _ready(deck, mprop=3, phi=0.8)
    store = vehicle.store
    phi_local = 0.0
    cin = deck.look_up("cin_vs_alphax_mach", ALPHAX, MACH)
    spi0 = deck.look_up("spi_vs_mach_phi_alphax", MACH, phi_local, ALPHAX)
    spi = deck.look_up(
        "spi_vs_mach_phi_alphax", MACH, phi_local / 0.0676, ALPHAX
    )
    thrust = 0.0676 * phi_local * spi * AGRAV * RHO * DVBE * cin * AINTAKE
    assert thrust == 0.0
    spi_if_saved = deck.look_up(
        "spi_vs_mach_phi_alphax", MACH, 0.8 / 0.0676, ALPHAX
    )
    assert spi == spi0
    assert spi != spi_if_saved

    prop.execute(vehicle, _ctx())
    assert store.get("phi") == 0.0
    assert store.get("thrust") == 0.0
    assert _approx(store.get("cin"), cin)
    assert _approx(store.get("spi"), spi)
    assert store.get("mprop") == 3
    assert store.get("thrst_stoch") == 0.0
    assert store.get("thrst_req") == 0.0
