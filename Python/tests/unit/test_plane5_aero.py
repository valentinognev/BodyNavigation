from pathlib import Path

import pytest

from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.flat3.falcon5.aero import Plane5Aero

FALCON5 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/FALCON5_250116/FALCON5"
AERO = FALCON5 / "Falcon5_aero_deck.asc"


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


def _expected(deck, mac, mach, alphax):
    cl_name = f"cl_{mac}MAC_vs_mach_alphax"
    cd_name = f"cd_{mac}MAC_vs_mach_alphax"
    cl = deck.look_up(cl_name, mach, alphax)
    cd = deck.look_up(cd_name, mach, alphax)
    clp = deck.look_up(cl_name, mach, alphax + 2)
    cln = deck.look_up(cl_name, mach, alphax - 2)
    cla = (clp - cln) / 4
    return cl, cd, cla, cl / cd


def test_name_is_aerodynamics():
    assert Plane5Aero(_deck()).name == "aerodynamics"


def test_define_registers_cpp_fields_not_alphax_or_mach():
    vehicle = _Vehicle()
    Plane5Aero(_deck()).define(vehicle)
    store = vehicle.store
    assert store.get("cl") == 0.0
    assert store.get("cd") == 0.0
    assert store.get("cl_ov_cd") == 0.0
    assert store.get("cla") == 0.0
    assert store.get("area") == 27.87
    assert store.get("mac") == 0
    assert store.field("mac").type == "int"
    assert "alphax" not in store.names()
    assert "mach" not in store.names()


@pytest.mark.parametrize("mac", [30, 35, 40])
def test_execute_matches_lookup_at_mach_0_6(mac):
    deck = _deck()
    vehicle = _Vehicle()
    aero = Plane5Aero(deck)
    aero.define(vehicle)
    store = vehicle.store
    store.define(Field("mach", 0.6, "real", "out", "environment"))
    store.define(Field("alphax", 5.0, "real", "out", "control"))
    store.set("mac", mac)

    mach = 0.6
    alphax = store.get("alphax")
    cl, cd, cla, cl_ov_cd = _expected(deck, mac, mach, alphax)

    aero.execute(vehicle, _ctx())

    assert store.get("cl") == pytest.approx(cl, rel=1e-12)
    assert store.get("cd") == pytest.approx(cd, rel=1e-12)
    assert store.get("cla") == pytest.approx(cla, rel=1e-12)
    assert store.get("cl_ov_cd") == pytest.approx(cl_ov_cd, rel=1e-12)
