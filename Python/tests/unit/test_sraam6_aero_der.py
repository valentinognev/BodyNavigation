import math
from pathlib import Path

import numpy as np

from cadac.constants import DEG, R
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
    ai11=0.308,
    ai33=59.80,
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
    store.define(Field("ai11", ai11, "real", "out", "propulsion"))
    store.define(Field("ai33", ai33, "real", "out", "propulsion"))
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


def _expected_dna_dma(deck, store):
    look_up = deck.look_up
    vmach = store.get("vmach")
    alppx = store.get("alppx")
    pdynmc = store.get("pdynmc")
    refl = store.get("refl")
    refa = store.get("refa")
    vmass = store.get("vmass")
    xcgref = store.get("xcgref")
    xcg = store.get("xcg")
    ai33 = store.get("ai33")
    alpp = alppx + 3.0
    if alpp < 3.0:
        alpp = 3.0
    alpm = alppx - 3.0
    if alpm < 0.0:
        alpm = 0.0
    cn0p = look_up("cn0_vs_mach_alpha", vmach, alpp)
    cn0m = look_up("cn0_vs_mach_alpha", vmach, alpm)
    cna = DEG * (cn0p - cn0m) / (alpp - alpm)
    clm0p = look_up("clm0_vs_mach_alpha", vmach, alpp)
    clm0m = look_up("clm0_vs_mach_alpha", vmach, alpm)
    cma = DEG * (clm0p - clm0m) / (alpp - alpm) - cna * (xcgref - xcg) / refl
    dna = (pdynmc * refa / vmass) * cna
    dma = (pdynmc * refa * refl / ai33) * cma
    return dna, dma


def test_execute_1v1_dna_dma_match_cpp_finite_difference():
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
        ai11=0.308,
        ai33=59.80,
    )
    store = vehicle.store
    want_dna, want_dma = _expected_dna_dma(deck, store)
    aero.execute(vehicle, _ctx())
    np.testing.assert_allclose(store.get("dna"), want_dna, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("dma"), want_dma, rtol=RTOL, atol=ATOL)
    assert math.isfinite(store.get("dna"))
    assert math.isfinite(store.get("dma"))
    assert store.get("dna") != 0.0
    assert store.get("dma") != 0.0


def test_bypass_keeps_saved_dna_when_alppx_near_alplimx():
    _deck, vehicle, aero = _ready(alppx=44.0, alplimx=46.0, ai11=0.308, ai33=59.80)
    store = vehicle.store
    saved = 12.345
    store.set("dna", saved)
    aero.execute(vehicle, _ctx())
    np.testing.assert_allclose(store.get("dna"), saved, rtol=RTOL, atol=ATOL)
