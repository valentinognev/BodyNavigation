from pathlib import Path

import pytest

from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.cruise5.aero import Cruise5Aero

CRUISE5 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/CRUISE5_250115/CRUISE5"
RTOL = 1e-12
ATOL = 1e-14
MACH = 0.7
ALPHAX = 0.0


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _deck():
    _, tables = parse_asc_deck(CRUISE5 / "cruise3_aero_deck.asc")
    return Datadeck.from_tables(tables)


def test_name_is_aerodynamics():
    assert Cruise5Aero(_deck()).name == "aerodynamics"


def test_define_area_default_0_929_and_no_alphax():
    vehicle = _Vehicle()
    Cruise5Aero(_deck()).define(vehicle)
    assert vehicle.store.get("area") == pytest.approx(0.929, rel=RTOL, abs=ATOL)
    assert "alphax" not in vehicle.store.names()
    field = vehicle.store.field("cl_ov_cd")
    assert field.module == "aerodynamics"
    assert field.outputs == ("scrn", "plot")


def test_execute_drag_polar_vs_lookup_formulas():
    deck = _deck()
    vehicle = _Vehicle()
    aero = Cruise5Aero(deck)
    aero.define(vehicle)
    vehicle.store.define(Field("mach", MACH, "real", "out", "environment"))
    vehicle.store.define(Field("alphax", ALPHAX, "real", "out", "control"))
    cd0 = deck.look_up("cd0_vs_mach", MACH)
    cl0 = deck.look_up("cl0_vs_mach", MACH)
    cla = deck.look_up("cla_vs_mach", MACH)
    ckk = deck.look_up("ckk_vs_mach", MACH)
    cla0 = deck.look_up("cla0_vs_mach", MACH)
    cl = cla0 + cla * ALPHAX
    cd = cd0 + ckk * (cl - cl0) ** 2
    aero.execute(vehicle, None)
    store = vehicle.store
    assert store.get("cl") == pytest.approx(cl, rel=RTOL, abs=ATOL)
    assert store.get("cd") == pytest.approx(cd, rel=RTOL, abs=ATOL)
    assert store.get("cla") == pytest.approx(cla, rel=RTOL, abs=ATOL)
    assert store.get("cl_ov_cd") == pytest.approx(cl / cd, rel=RTOL, abs=ATOL)
