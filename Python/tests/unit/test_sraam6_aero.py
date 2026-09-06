import math
from pathlib import Path

import numpy as np

from cadac.constants import R
from cadac.env.us76 import atmosphere76
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.sraam6.aero import Sraam6Aero

SRAAM6 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/SRAAM6_250130/SRAAM6"
AERO = SRAAM6 / "sraam6_aero_deck.asc"

RTOL = 1e-12
ATOL = 1e-14

DEFINED = (
    "refl",
    "refa",
    "ca",
    "cy",
    "cn",
    "cll",
    "clm",
    "cln",
    "cn0",
    "cnp",
    "clm0",
    "clmp",
    "cyp",
    "clnp",
    "ca0",
    "caa",
    "cad",
    "cndq",
    "clmdq",
    "clmq",
    "cllap",
    "clldp",
    "cllp",
    "dna",
    "dnd",
    "dma",
    "dmq",
    "dmd",
    "dlp",
    "dld",
    "cna",
    "cya",
    "clma",
    "clna",
    "stmarg",
    "realq1",
    "realq2",
    "wnq",
    "zetq",
    "realp",
    "alplimx",
    "gavail",
    "gmax",
    "pqreal",
    "trcond",
    "trcvel",
    "trmach",
    "trdynm",
    "trload",
    "tralp",
    "trtht",
    "trthtd",
    "trphid",
    "trate",
)

ROLES = {
    "refl": "init",
    "refa": "init",
    "ca": "out",
    "cy": "out",
    "cn": "out",
    "cll": "out",
    "clm": "out",
    "cln": "out",
    "cn0": "diag",
    "cnp": "diag",
    "clm0": "diag",
    "clmp": "diag",
    "cyp": "diag",
    "clnp": "diag",
    "ca0": "diag",
    "caa": "diag",
    "cad": "diag",
    "cndq": "diag",
    "clmdq": "diag",
    "clmq": "diag",
    "cllap": "diag",
    "clldp": "diag",
    "cllp": "diag",
    "dna": "out",
    "dnd": "out",
    "dma": "out",
    "dmq": "out",
    "dmd": "out",
    "dlp": "out",
    "dld": "out",
    "cna": "diag",
    "cya": "diag",
    "clma": "diag",
    "clna": "diag",
    "stmarg": "diag",
    "realq1": "diag",
    "realq2": "diag",
    "wnq": "diag",
    "zetq": "diag",
    "realp": "diag",
    "alplimx": "data",
    "gavail": "diag",
    "gmax": "diag",
    "pqreal": "diag",
    "trcond": "diag",
    "trcvel": "data",
    "trmach": "data",
    "trdynm": "data",
    "trload": "data",
    "tralp": "data",
    "trtht": "data",
    "trthtd": "data",
    "trphid": "data",
    "trate": "data",
}

PLOT_FLAGGED = ("gavail", "gmax")
EXTERNALS = (
    "vmach",
    "alppx",
    "phip",
    "ppx",
    "qqx",
    "rrx",
    "mprop",
    "vmass",
    "xcgref",
    "xcg",
    "alimit",
    "dpx",
    "dqx",
    "drx",
    "pdynmc",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(0.0, 0.001, 0.0, 0.0, None, 0)


def _deck():
    _, tables = parse_asc_deck(AERO)
    return Datadeck.from_tables(tables)


def _us76_air(alt=5000.0, dvbe=250.0):
    rho, _press, tempk = atmosphere76(alt)
    vsound = math.sqrt(1.4 * R * tempk)
    vmach = abs(dvbe / vsound)
    pdynmc = 0.5 * rho * dvbe**2
    return vmach, pdynmc


def _externals(
    store,
    *,
    vmach,
    pdynmc,
    alppx=0.0,
    phip=0.0,
    ppx=0.0,
    qqx=0.0,
    rrx=0.0,
    mprop=1,
    vmass=92.0,
    xcgref=1.536,
    xcg=1.536,
    alimit=50.0,
    dpx=0.0,
    dqx=0.0,
    drx=0.0,
    dvbe=250.0,
    alplimx=46.0,
):
    store.define(Field("vmach", vmach, "real", "out", "environment"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("alppx", alppx, "real", "out", "kinematics"))
    store.define(Field("phip", phip, "real", "out", "kinematics"))
    store.define(Field("ppx", ppx, "real", "out", "euler"))
    store.define(Field("qqx", qqx, "real", "out", "euler"))
    store.define(Field("rrx", rrx, "real", "out", "euler"))
    store.define(Field("mprop", mprop, "int", "out", "propulsion"))
    store.define(Field("vmass", vmass, "real", "out", "propulsion"))
    store.define(Field("xcgref", xcgref, "real", "out", "propulsion"))
    store.define(Field("xcg", xcg, "real", "out", "propulsion"))
    store.define(Field("alimit", alimit, "real", "data", "control"))
    store.define(Field("dpx", dpx, "real", "out", "actuator"))
    store.define(Field("dqx", dqx, "real", "out", "actuator"))
    store.define(Field("drx", drx, "real", "out", "actuator"))
    store.define(Field("dvbe", dvbe, "real", "out", "newton"))
    store.set("alplimx", alplimx)


def _ready(**kw):
    vmach, pdynmc = _us76_air()
    kw.setdefault("vmach", vmach)
    kw.setdefault("pdynmc", pdynmc)
    deck = _deck()
    vehicle = _Vehicle()
    aero = Sraam6Aero(deck)
    aero.define(vehicle)
    aero.initialize(vehicle, _ctx())
    _externals(vehicle.store, **kw)
    return deck, vehicle, aero


def _expected_ca_cn_cy(deck, store):
    look_up = deck.look_up
    vmach = store.get("vmach")
    alppx = store.get("alppx")
    phip = store.get("phip")
    mprop = store.get("mprop")
    dqx = store.get("dqx")
    drx = store.get("drx")
    cphip = math.cos(phip)
    sphip = math.sin(phip)
    dqax = dqx * cphip - drx * sphip
    drax = dqx * sphip + drx * cphip
    ca0 = look_up("ca0_vs_mach", vmach)
    caa = look_up("caa_vs_mach", vmach)
    cad = look_up("cad_vs_mach", vmach)
    caoff = look_up("caoff_vs_mach", vmach)
    deff = (math.fabs(dqax) + math.fabs(drax)) / 2.0
    ca = ca0 + caa * alppx + cad * deff * deff + float(1 - mprop) * caoff
    cyp = look_up("cyp_vs_mach_alpha", vmach, alppx)
    cydr = look_up("cndq_vs_mach_alpha", vmach, alppx)
    s4phi = math.sin(4.0 * phip)
    cya = cyp * s4phi + cydr * drax
    cn0 = look_up("cn0_vs_mach_alpha", vmach, alppx)
    cnp = look_up("cnp_vs_mach_alpha", vmach, alppx)
    cndq = look_up("cndq_vs_mach_alpha", vmach, alppx)
    s2phi = math.sin(2.0 * phip) ** 2
    cna = cn0 + cnp * s2phi + cndq * dqax
    cy = cya * cphip - cna * sphip
    cn = cya * sphip + cna * cphip
    return ca, cy, cn


def test_name_is_aerodynamics():
    assert Sraam6Aero(_deck()).name == "aerodynamics"


def test_define_registers_cpp_fields_not_externals():
    vehicle = _Vehicle()
    Sraam6Aero(_deck()).define(vehicle)
    store = vehicle.store
    for name in DEFINED:
        assert name in store.names(), name
        field = store.field(name)
        assert field.module == "aerodynamics"
        assert field.role == ROLES[name], name
        if name == "trcond":
            assert field.type == "int"
            assert store.get(name) == 0
            assert field.outputs == ("scrn", "plot")
        else:
            assert field.type == "real"
            assert store.get(name) == 0.0
            if name in PLOT_FLAGGED:
                assert field.outputs == ("plot",), name
            else:
                assert field.outputs == (), name
    for name in EXTERNALS:
        assert name not in store.names()


def test_initialize_overwrites_trcvel_and_sets_refs():
    vehicle = _Vehicle()
    aero = Sraam6Aero(_deck())
    aero.define(vehicle)
    store = vehicle.store
    store.set("trcvel", 99.0)
    aero.initialize(vehicle, _ctx())
    np.testing.assert_allclose(store.get("trcvel"), 1e-3, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("refl"), 0.1524, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("refa"), 0.01824, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("trmach"), 0.5, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("trdynm"), 10e3, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("trload"), 3.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("tralp"), 1.0, rtol=RTOL, atol=ATOL)
    assert store.get("trcond") == 0


def test_execute_1v1_ca_cn_cy_match_lookup_formulas():
    deck, vehicle, aero = _ready(
        alppx=0.0,
        phip=0.0,
        ppx=0.0,
        qqx=0.0,
        rrx=0.0,
        mprop=1,
        vmass=92.0,
        xcgref=1.536,
        xcg=1.536,
        alimit=50.0,
        dpx=0.0,
        dqx=0.0,
        drx=0.0,
        alplimx=46.0,
    )
    store = vehicle.store
    vmach, pdynmc = _us76_air()
    np.testing.assert_allclose(store.get("vmach"), vmach, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("pdynmc"), pdynmc, rtol=RTOL, atol=ATOL)
    want_ca, want_cy, want_cn = _expected_ca_cn_cy(deck, store)
    aero.execute(vehicle, _ctx())
    np.testing.assert_allclose(store.get("ca"), want_ca, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("cy"), want_cy, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("cn"), want_cn, rtol=RTOL, atol=ATOL)


def test_gmax_below_trload_sets_trcond_4():
    _deck_obj, vehicle, aero = _ready(pdynmc=0.0)
    aero.execute(vehicle, _ctx())
    assert vehicle.store.get("trcond") == 4


def test_aero_deck_splits_glued_clnp_tail():
    deck = _deck()
    np.testing.assert_allclose(
        deck.look_up("clnp_vs_mach_alpha", 0.85, 75.0), -9.582, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        deck.look_up("clnp_vs_mach_alpha", 0.85, 90.0), -14.563, rtol=RTOL, atol=ATOL
    )
