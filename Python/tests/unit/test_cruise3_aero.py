from pathlib import Path

from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.cruise3.aero import Cruise3Aero

HYPER3 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/HYPER3_250114/HYPER3"
AERO = HYPER3 / "ghame3_aero_deck.asc"


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.1,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _deck():
    _, tables = parse_asc_deck(AERO)
    return Datadeck.from_tables(tables)


def test_name_is_aerodynamics():
    assert Cruise3Aero(_deck()).name == "aerodynamics"


def test_define_registers_spec_fields():
    vehicle = _Vehicle()
    Cruise3Aero(_deck()).define(vehicle)
    for name in ("alphax", "area", "cl", "cd", "cla", "cl_ov_cd"):
        assert vehicle.store.get(name) == 0.0


def test_execute_drag_polar_at_mach_0_760854_alphax_7():
    deck = _deck()
    vehicle = _Vehicle()
    aero = Cruise3Aero(deck)
    aero.define(vehicle)
    store = vehicle.store
    store.define(Field("mach", 0.760854, "real", "out", "environment"))
    store.set("alphax", 7.0)

    mach = 0.760854
    alphax = 7.0
    cd0 = deck.look_up("cd0_vs_mach", mach)
    cl0 = deck.look_up("cl0_vs_mach", mach)
    cla = deck.look_up("cla_vs_mach", mach)
    ckk = deck.look_up("ckk_vs_mach", mach)
    cla0 = deck.look_up("cla0_vs_mach", mach)
    cl = cla0 + cla * alphax
    cd = cd0 + ckk * (cl - cl0) ** 2
    cl_ov_cd = cl / cd

    aero.execute(vehicle, _ctx())

    assert store.get("cl") == cl
    assert store.get("cd") == cd
    assert store.get("cla") == cla
    assert store.get("cl_ov_cd") == cl_ov_cd
    assert store.get("alphax") == alphax
    assert store.get("area") == 0.0
