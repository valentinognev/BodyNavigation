from math import cos, sin
from pathlib import Path

import pytest

from cadac.constants import RAD
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.hyper5.aero import Hyper5Aero

HYPER5 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/HYPER5_250113/HYPER5"
AERO = HYPER5 / "hyper5_aero_deck.asc"

RTOL = 1e-12
ATOL = 1e-14

MACH = 4.0
ALPHAX = 2.0
AREA = 11.6986

DEFINED = ("cl", "cd", "cl_ov_cd", "area", "cla", "cn", "ca")
EXTERNALS = ("mach", "alphax", "time")
ROLES = {
    "cl": "out",
    "cd": "out",
    "cl_ov_cd": "diag",
    "area": "data",
    "cla": "out",
    "cn": "diag",
    "ca": "diag",
}
PLOT_FLAGGED = ("cl", "cd", "cl_ov_cd", "cla", "cn", "ca")


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.05,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _deck():
    _, tables = parse_asc_deck(AERO)
    return Datadeck.from_tables(tables)


def _expected(deck, alphax, mach):
    cn = deck.look_up("cn_rr3x_vs_alphax_mach", alphax, mach)
    ca = deck.look_up("ca_rr3x_vs_alphax_mach", alphax, mach)
    alpha = alphax * RAD
    cd = cn * sin(alpha) + ca * cos(alpha)
    cl = cn * cos(alpha) - ca * sin(alpha)
    cnp = deck.look_up("cn_rr3x_vs_alphax_mach", alphax + 2, mach)
    cnn = deck.look_up("cn_rr3x_vs_alphax_mach", alphax - 2, mach)
    cap = deck.look_up("ca_rr3x_vs_alphax_mach", alphax + 2, mach)
    can = deck.look_up("ca_rr3x_vs_alphax_mach", alphax - 2, mach)
    cna = (cnp - cnn) / 4
    caa = (cap - can) / 4
    cla = cna * cos(alpha) - caa * sin(alpha)
    return cn, ca, cl, cd, cla, cl / cd


def _ready(deck=None, *, mach=MACH, alphax=ALPHAX, area=AREA):
    if deck is None:
        deck = _deck()
    vehicle = _Vehicle()
    aero = Hyper5Aero(deck)
    aero.define(vehicle)
    store = vehicle.store
    store.define(Field("mach", mach, "real", "out", "environment"))
    store.define(Field("alphax", alphax, "real", "out", "control"))
    store.set("area", area)
    aero.initialize(vehicle, _ctx())
    return vehicle, aero, deck


def test_name_is_aerodynamics():
    assert Hyper5Aero(_deck()).name == "aerodynamics"


def test_define_registers_cpp_fields_not_alphax_mach_or_time():
    vehicle = _Vehicle()
    Hyper5Aero(_deck()).define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    for name in DEFINED:
        field = store.field(name)
        assert store.get(name) == 0.0
        assert field.type == "real"
        assert field.role == ROLES[name]
        assert field.module == "aerodynamics"
        if name in PLOT_FLAGGED:
            assert field.outputs == ("scrn", "plot")
        else:
            assert field.outputs == ()
    for name in EXTERNALS:
        assert name not in store.names()


def test_initialize_is_pass():
    vehicle, _aero, _deck = _ready()
    store = vehicle.store
    assert store.get("cl") == 0.0
    assert store.get("cd") == 0.0
    assert store.get("cla") == 0.0
    assert store.get("cn") == 0.0
    assert store.get("ca") == 0.0
    assert store.get("cl_ov_cd") == 0.0
    assert store.get("area") == AREA


def test_execute_matches_lookup_then_cpp_at_mach_4_alphax_2():
    vehicle, aero, deck = _ready()
    store = vehicle.store
    cn, ca, cl, cd, cla, cl_ov_cd = _expected(deck, ALPHAX, MACH)
    assert AREA != 0.0
    assert cn != 0.0
    assert ca != 0.0

    aero.execute(vehicle, _ctx())

    assert store.get("cn") == pytest.approx(cn, rel=RTOL, abs=ATOL)
    assert store.get("ca") == pytest.approx(ca, rel=RTOL, abs=ATOL)
    assert store.get("cl") == pytest.approx(cl, rel=RTOL, abs=ATOL)
    assert store.get("cd") == pytest.approx(cd, rel=RTOL, abs=ATOL)
    assert store.get("cla") == pytest.approx(cla, rel=RTOL, abs=ATOL)
    assert store.get("cl_ov_cd") == pytest.approx(cl_ov_cd, rel=RTOL, abs=ATOL)
    assert store.get("area") == AREA
    assert "time" not in store.names()
