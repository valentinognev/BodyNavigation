from math import cos, fabs, sin, sqrt
from pathlib import Path

import numpy as np
import pytest

from cadac.constants import AGRAV, DEG
from cadac.env.us76 import atmosphere76
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.flat6.agm6.aero import Agm6Aero

AGM6 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/AGM6_250217/AGM6"
AERO = AGM6 / "AGM6_aero_deck.asc"

RTOL = 1e-12
ATOL = 1e-14

VMACH = 0.85
ALPPX = 3.0
PHIP = 0.0
DVBA = 293.0
VMASS = 1360.0
AI11 = 42.5
AI33 = 2632.0
ALPLIMX = 20.0
HBE = 7000.0

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
    "tmcode",
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
    "trcond": "init",
    "tmcode": "data",
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

PLOT_FLAGGED = (
    "stmarg",
    "realq1",
    "realq2",
    "wnq",
    "zetq",
    "gavail",
    "gmax",
    "trcond",
)

EXTERNALS = (
    "vmach",
    "pdynmc",
    "alppx",
    "phip",
    "ppx",
    "qqx",
    "rrx",
    "dvba",
    "vmass",
    "ai11",
    "ai33",
    "dpx",
    "dqx",
    "drx",
    "alimit",
)

DER_FIELDS = (
    "dna",
    "dnd",
    "dma",
    "dmq",
    "dmd",
    "dlp",
    "dld",
    "stmarg",
    "realq1",
    "realq2",
    "wnq",
    "zetq",
    "realp",
    "pqreal",
)

FORMULA_NAMES = (
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
    "cna",
    "cya",
    "clma",
    "clna",
    "gmax",
    "gavail",
) + DER_FIELDS


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


def _pdynmc(hbe=HBE, dvba=DVBA):
    rho, _press, _tempk = atmosphere76(hbe)
    return 0.5 * rho * dvba * dvba


def _optional(store, name, default=0.0):
    if name in store.names():
        return store.get(name)
    return default


def _externals(
    store,
    *,
    vmach=VMACH,
    alppx=ALPPX,
    phip=PHIP,
    dvba=DVBA,
    vmass=VMASS,
    ai11=AI11,
    ai33=AI33,
    pdynmc=None,
    ppx=0.0,
    qqx=0.0,
    rrx=0.0,
    dpx=0.0,
    dqx=0.0,
    drx=0.0,
    alimit=0.0,
    include_controls=True,
):
    if pdynmc is None:
        pdynmc = _pdynmc(dvba=dvba)
    store.define(Field("vmach", vmach, "real", "out", "environment"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("alppx", alppx, "real", "out", "kinematics"))
    store.define(Field("phip", phip, "real", "out", "kinematics"))
    store.define(Field("ppx", ppx, "real", "init/out", "euler"))
    store.define(Field("qqx", qqx, "real", "init/out", "euler"))
    store.define(Field("rrx", rrx, "real", "init/out", "euler"))
    store.define(Field("dvba", dvba, "real", "out", "environment"))
    store.define(Field("vmass", vmass, "real", "out", "propulsion"))
    store.define(Field("ai11", ai11, "real", "data", "propulsion"))
    store.define(Field("ai33", ai33, "real", "data", "propulsion"))
    if include_controls:
        store.define(Field("dpx", dpx, "real", "out", "actuator"))
        store.define(Field("dqx", dqx, "real", "out", "actuator"))
        store.define(Field("drx", drx, "real", "out", "actuator"))
        store.define(Field("alimit", alimit, "real", "data", "control"))


def _ready(*, alplimx=ALPLIMX, include_controls=True, **kw):
    deck = _deck()
    vehicle = _Vehicle()
    aero = Agm6Aero(deck)
    aero.define(vehicle)
    aero.initialize(vehicle, _ctx())
    _externals(vehicle.store, include_controls=include_controls, **kw)
    vehicle.store.set("alplimx", alplimx)
    return deck, vehicle, aero


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _expected(deck, store):
    look_up = deck.look_up
    vmach = store.get("vmach")
    pdynmc = store.get("pdynmc")
    alppx = store.get("alppx")
    phip = store.get("phip")
    ppx = store.get("ppx")
    qqx = store.get("qqx")
    rrx = store.get("rrx")
    dvba = store.get("dvba")
    vmass = store.get("vmass")
    ai11 = store.get("ai11")
    ai33 = store.get("ai33")
    refl = store.get("refl")
    refa = store.get("refa")
    alplimx = store.get("alplimx")
    trload = store.get("trload")
    trcond = store.get("trcond")
    dpx = _optional(store, "dpx")
    dqx = _optional(store, "dqx")
    drx = _optional(store, "drx")
    alimit = _optional(store, "alimit")

    cphip = cos(phip)
    sphip = sin(phip)
    dqax = dqx * cphip - drx * sphip
    drax = dqx * sphip + drx * cphip
    qqax = qqx * cphip - rrx * sphip
    rrax = qqx * sphip + rrx * cphip

    ca0 = look_up("ca0_vs_mach", vmach)
    caa = look_up("caa_vs_mach", vmach)
    cad = look_up("cad_vs_mach", vmach)
    deff = (fabs(dqax) + fabs(drax)) / 2
    ca = ca0 + caa * alppx + cad * deff * deff

    cyp = look_up("cyp_vs_mach_alpha", vmach, alppx)
    cydr = look_up("cndq_vs_mach", vmach)
    s4phi = sin(4 * phip)
    cya = cyp * s4phi + cydr * drax

    cn0 = look_up("cn0_vs_mach_alpha", vmach, alppx)
    cnp = look_up("cnp_vs_mach_alpha", vmach, alppx)
    cndq = look_up("cndq_vs_mach", vmach)
    s2phi = sin(2 * phip) ** 2
    cna = cn0 + cnp * s2phi + cndq * dqax

    cllap = look_up("cllap_vs_mach", vmach)
    cllp = look_up("cllp_vs_mach", vmach)
    clldp = look_up("clldp_vs_mach", vmach)
    cll = cllap * alppx * alppx * s4phi + cllp * ppx * refl / (2 * dvba) + clldp * dpx

    clm0 = look_up("clm0_vs_mach_alpha", vmach, alppx)
    clmp = look_up("clmp_vs_mach_alpha", vmach, alppx)
    clmq = look_up("clmq_vs_mach", vmach)
    clmdq = look_up("clmdq_vs_mach", vmach)
    clma = clm0 + clmp * s2phi + clmq * qqax * refl / (2 * dvba) + clmdq * dqax

    clnp = look_up("clnp_vs_mach_alpha", vmach, alppx)
    clnr = clmq
    clndr = clmdq
    clna = clnp * s4phi + clnr * rrax * refl / (2 * dvba) + clndr * drax

    cy = cya * cphip - cna * sphip
    cn = cya * sphip + cna * cphip
    clm = clma * cphip + clna * sphip
    cln = clna * cphip - clma * sphip

    cn0mx = look_up("cn0_vs_mach_alpha", vmach, alplimx)
    almx = cn0mx * pdynmc * refa
    weight = vmass * AGRAV
    gmax = almx / weight
    if gmax >= alimit:
        gmax = alimit
    aload = cn0 * pdynmc * refa
    gg = aload / weight
    gavail = gmax - gg
    if gmax < trload:
        trcond = 4

    dna = store.get("dna")
    dnd = store.get("dnd")
    dma = store.get("dma")
    dmq = store.get("dmq")
    dmd = store.get("dmd")
    dlp = store.get("dlp")
    dld = store.get("dld")
    stmarg = store.get("stmarg")

    if alppx < (alplimx - 3.0):
        alpp = alppx + 3
        if alpp < 3:
            alpp = 3
        alpm = alppx - 3
        if alpm < 0:
            alpm = 0
        cn0p = look_up("cn0_vs_mach_alpha", vmach, alpp)
        cn0m = look_up("cn0_vs_mach_alpha", vmach, alpm)
        dum = cn0p - cn0m
        cna_der = DEG * dum / (alpp - alpm)
        cnd = DEG * cndq
        clm0p = look_up("clm0_vs_mach_alpha", vmach, alpp)
        clm0m = look_up("clm0_vs_mach_alpha", vmach, alpm)
        cma = DEG * (clm0p - clm0m) / (alpp - alpm)
        cmq = DEG * clmq
        cmd = DEG * clmdq
        clp = DEG * cllp
        cld = DEG * clldp
        dumn = pdynmc * refa / vmass
        dna = dumn * cna_der
        dnd = dumn * cnd
        dumm = pdynmc * refa * refl / ai33
        dma = dumm * cma
        dmq = dumm * (refl / (2 * dvba)) * cmq
        dmd = dumm * cmd
        duml = pdynmc * refa * refl / ai11
        dlp = duml * (refl / (2.0 * dvba)) * clp
        dld = duml * cld
        stmarg = -cma / cna_der

    a11 = dmq
    a12 = dma / dna
    a21 = dna
    a22 = -dna / dvba
    arg = (a11 + a22) ** 2 - 4.0 * (a11 * a22 - a12 * a21)
    if arg >= 0.0:
        wnq = 0.0
        zetq = 0.0
        dum = a11 + a22
        realq1 = (dum + sqrt(arg)) / 2.0
        realq2 = (dum - sqrt(arg)) / 2.0
        pqreal = (realq1 + realq2) / 2.0
    else:
        realq1 = 0.0
        realq2 = 0.0
        wnq = sqrt(a11 * a22 - a12 * a21)
        zetq = -(a11 + a22) / (2.0 * wnq)
        pqreal = -zetq * wnq
    realp = dlp

    return {
        "ca": ca,
        "cy": cy,
        "cn": cn,
        "cll": cll,
        "clm": clm,
        "cln": cln,
        "cn0": cn0,
        "cnp": cnp,
        "clm0": clm0,
        "clmp": clmp,
        "cyp": cyp,
        "clnp": clnp,
        "ca0": ca0,
        "caa": caa,
        "cad": cad,
        "cndq": cndq,
        "clmdq": clmdq,
        "clmq": clmq,
        "cllap": cllap,
        "clldp": clldp,
        "cllp": cllp,
        "cna": cna,
        "cya": cya,
        "clma": clma,
        "clna": clna,
        "gmax": gmax,
        "gavail": gavail,
        "trcond": trcond,
        "dna": dna,
        "dnd": dnd,
        "dma": dma,
        "dmq": dmq,
        "dmd": dmd,
        "dlp": dlp,
        "dld": dld,
        "stmarg": stmarg,
        "realq1": realq1,
        "realq2": realq2,
        "wnq": wnq,
        "zetq": zetq,
        "realp": realp,
        "pqreal": pqreal,
    }


def test_name_is_aerodynamics():
    assert Agm6Aero(_deck()).name == "aerodynamics"


def test_define_registers_cpp_fields_not_externals():
    vehicle = _Vehicle()
    Agm6Aero(_deck()).define(vehicle)
    store = vehicle.store
    for name in DEFINED:
        assert name in store.names(), name
        field = store.field(name)
        assert field.module == "aerodynamics"
        assert field.role == ROLES[name], name
        if name == "trcond":
            assert field.type == "int"
            assert store.get(name) == 0
        else:
            assert field.type == "real"
            assert store.get(name) == 0.0
        if name in PLOT_FLAGGED:
            assert field.outputs == ("plot",), name
        else:
            assert field.outputs == (), name
    for name in EXTERNALS:
        assert name not in store.names()
    assert "vmach" not in store.names()
    assert "pdynmc" not in store.names()
    assert "alppx" not in store.names()


def test_initialize_sets_refs_and_termination():
    vehicle = _Vehicle()
    aero = Agm6Aero(_deck())
    aero.define(vehicle)
    aero.initialize(vehicle, _ctx())
    store = vehicle.store
    assert store.get("refl") == 0.5
    assert store.get("refa") == 0.196
    assert store.get("trmach") == 0.4
    assert store.get("trdynm") == 10e3
    assert store.get("trload") == 0.5
    assert store.get("tralp") == 1
    assert store.get("trcond") == 0
    assert store.get("trcvel") == 10e-5
    assert store.get("alplimx") == 0.0


def test_execute_matches_cadac_formulas():
    deck, vehicle, aero = _ready(
        vmach=VMACH,
        alppx=ALPPX,
        phip=PHIP,
        dvba=DVBA,
        vmass=VMASS,
        ai11=AI11,
        ai33=AI33,
        dpx=0.0,
        dqx=0.0,
        drx=0.0,
        alimit=0.0,
    )
    store = vehicle.store
    assert _approx(store.get("pdynmc"), _pdynmc())
    assert store.get("vmach") == VMACH
    assert store.get("alppx") == ALPPX
    want = _expected(deck, store)
    aero.execute(vehicle, _ctx())
    assert np.isfinite(store.get("dna"))
    assert store.get("dna") != 0.0
    for name in FORMULA_NAMES:
        assert np.isfinite(store.get(name)), name
        assert _approx(store.get(name), want[name]), name
    assert store.get("trcond") == want["trcond"]
    assert _approx(store.get("ca"), want["ca"])
    assert _approx(store.get("cn"), want["cn"])
    assert _approx(store.get("clm"), want["clm"])


def test_execute_matches_with_phip_and_rates():
    deck, vehicle, aero = _ready(
        phip=0.4,
        ppx=10.0,
        qqx=5.0,
        rrx=8.0,
        dpx=2.0,
        dqx=3.0,
        drx=-1.0,
        alimit=3.0,
    )
    store = vehicle.store
    want = _expected(deck, store)
    aero.execute(vehicle, _ctx())
    for name in FORMULA_NAMES:
        assert np.isfinite(store.get(name)), name
        assert _approx(store.get(name), want[name]), name


def test_absent_controls_are_free_flight_zero():
    deck, vehicle, aero = _ready(include_controls=False)
    store = vehicle.store
    assert "dpx" not in store.names()
    assert "dqx" not in store.names()
    assert "drx" not in store.names()
    assert "alimit" not in store.names()
    want = _expected(deck, store)
    aero.execute(vehicle, _ctx())
    assert _approx(store.get("ca"), want["ca"])
    assert _approx(store.get("cn"), want["cn"])
    assert _approx(store.get("clm"), want["clm"])
    assert np.isfinite(store.get("dna"))


def test_der_skip_when_alppx_exceeds_alplimx_minus_3():
    _deck_obj, vehicle, aero = _ready(alppx=3.0, alplimx=20.0)
    store = vehicle.store
    aero.execute(vehicle, _ctx())
    assert store.get("stmarg") != 0.0
    store.set("stmarg", 123.456)
    store.set("alppx", 18.0)
    aero.execute(vehicle, _ctx())
    assert store.get("stmarg") == 123.456
