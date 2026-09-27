from math import cos, fabs, sin, sqrt
from pathlib import Path

import numpy as np
import pytest

from cadac.constants import AGRAV, R, RAD
from cadac.env.us76 import atmosphere76
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.round6.hyper6.aero import Hyper6Aero

HYPER6 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/HYPER6_250125/HYPER6"
AERO = HYPER6 / "ghame6_aero_deck.asc"

RTOL = 1e-12
ATOL = 1e-14

ALPHAX = 2.5
BETAX = 0.0
DELEX = 0.0
DELAX = 0.0
DELRX = 0.0
PPX = 0.0
QQX = 0.0
RRX = 0.0
DVBA = 1000.0
ALT = 10000.0
VMASS = 136077.0
ALPPLIMX = 21.0
ALPNLIMX = -3.0
STRCT_POS = 3.0
STRCT_NEG = -2.0

GHAME_IBBB0 = np.array(
    [
        [1.573e6, 0.0, 0.38e6],
        [0.0, 31.6e6, 0.0],
        [0.38e6, 0.0, 32.54e6],
    ],
    dtype=float,
)

DEFINED = (
    "maero",
    "refa",
    "refb",
    "refc",
    "cd",
    "cl",
    "cy",
    "cll",
    "clm",
    "cln",
    "cx",
    "cz",
    "cd0",
    "cda",
    "cl0",
    "cla",
    "clde",
    "cyb",
    "cyda",
    "cydr",
    "cllb",
    "cllda",
    "cllp",
    "cllr",
    "cm0",
    "cma",
    "cmde",
    "cmq",
    "clnb",
    "clnda",
    "clndr",
    "clnp",
    "clnr",
    "clldr",
    "clovercd",
    "stmarg",
    "dla",
    "dlde",
    "dma",
    "dmq",
    "dmde",
    "dyb",
    "dydr",
    "dnb",
    "dnr",
    "dndr",
    "dllp",
    "dllda",
    "alpplimx",
    "strct_pos_limitx",
    "gavail_pos",
    "gmax",
    "alpnlimx",
    "strct_neg_limitx",
    "gavail_neg",
    "gminx",
    "realp1",
    "realp2",
    "wnp",
    "zetp",
    "rpreal",
    "realy1",
    "realy2",
    "wny",
    "zety",
    "ryreal",
    "trcode",
    "tmcode",
    "trmach",
    "trdynm",
    "trload",
    "tralp",
    "refa_st",
    "caa",
)

ROLES = {
    "maero": "data",
    "refa": "init",
    "refb": "init",
    "refc": "init",
    "cd": "dia",
    "cl": "dia",
    "cy": "out",
    "cll": "out",
    "clm": "out",
    "cln": "out",
    "cx": "out",
    "cz": "out",
    "cd0": "diag",
    "cda": "diag",
    "cl0": "diag",
    "cla": "diag",
    "clde": "diag",
    "cyb": "diag",
    "cyda": "diag",
    "cydr": "diag",
    "cllb": "diag",
    "cllda": "diag",
    "cllp": "diag",
    "cllr": "diag",
    "cm0": "diag",
    "cma": "diag",
    "cmde": "diag",
    "cmq": "diag",
    "clnb": "diag",
    "clnda": "diag",
    "clndr": "diag",
    "clnp": "diag",
    "clnr": "diag",
    "clldr": "diag",
    "clovercd": "diag",
    "stmarg": "diag",
    "dla": "out",
    "dlde": "out",
    "dma": "out",
    "dmq": "out",
    "dmde": "out",
    "dyb": "out",
    "dydr": "out",
    "dnb": "out",
    "dnr": "out",
    "dndr": "out",
    "dllp": "out",
    "dllda": "out",
    "alpplimx": "data",
    "strct_pos_limitx": "data",
    "gavail_pos": "diag",
    "gmax": "out",
    "alpnlimx": "data",
    "strct_neg_limitx": "data",
    "gavail_neg": "diag",
    "gminx": "out",
    "realp1": "diag",
    "realp2": "diag",
    "wnp": "diag",
    "zetp": "diag",
    "rpreal": "diag",
    "realy1": "diag",
    "realy2": "diag",
    "wny": "diag",
    "zety": "diag",
    "ryreal": "diag",
    "trcode": "init",
    "tmcode": "data",
    "trmach": "data",
    "trdynm": "data",
    "trload": "data",
    "tralp": "data",
    "refa_st": "init",
    "caa": "init",
}

PLOT_FLAGGED = ("stmarg", "gavail_pos", "gmax", "gavail_neg", "gminx", "wnp")

EXTERNALS = (
    "time",
    "alphax",
    "betax",
    "vmach",
    "pdynmc",
    "dvba",
    "ppx",
    "qqx",
    "rrx",
    "vmass",
    "IBBB",
    "delax",
    "delex",
    "delrx",
)

DER_FIELDS = (
    "dla",
    "dlde",
    "dma",
    "dmq",
    "dmde",
    "dyb",
    "dydr",
    "dnb",
    "dnr",
    "dndr",
    "dllp",
    "dllda",
    "stmarg",
    "realp1",
    "realp2",
    "wnp",
    "zetp",
    "rpreal",
    "realy1",
    "realy2",
    "wny",
    "zety",
    "ryreal",
)

FORMULA_NAMES = (
    "cd",
    "cl",
    "cy",
    "cll",
    "clm",
    "cln",
    "cx",
    "cz",
    "cd0",
    "cda",
    "cl0",
    "cla",
    "clde",
    "cyb",
    "cyda",
    "cydr",
    "cllb",
    "cllda",
    "cllp",
    "cllr",
    "cm0",
    "cma",
    "cmde",
    "cmq",
    "clnb",
    "clnda",
    "clndr",
    "clnp",
    "clnr",
    "clldr",
    "clovercd",
    "gmax",
    "gminx",
    "gavail_pos",
    "gavail_neg",
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


def _us76_air(alt=ALT, dvba=DVBA):
    rho, _press, tempk = atmosphere76(alt)
    vsound = (1.4 * R * tempk) ** 0.5
    vmach = abs(dvba / vsound)
    pdynmc = 0.5 * rho * dvba * dvba
    return vmach, pdynmc


def _externals(
    store,
    *,
    alphax=ALPHAX,
    betax=BETAX,
    delex=DELEX,
    delax=DELAX,
    delrx=DELRX,
    ppx=PPX,
    qqx=QQX,
    rrx=RRX,
    dvba=DVBA,
    vmach=None,
    pdynmc=None,
    vmass=VMASS,
    ibbb=None,
):
    if vmach is None or pdynmc is None:
        climb_vmach, climb_pdynmc = _us76_air(dvba=dvba)
        if vmach is None:
            vmach = climb_vmach
        if pdynmc is None:
            pdynmc = climb_pdynmc
    if ibbb is None:
        ibbb = GHAME_IBBB0
    store.define(Field("alphax", alphax, "real", "init/diag", "kinematics"))
    store.define(Field("betax", betax, "real", "diag", "kinematics"))
    store.define(Field("vmach", vmach, "real", "out", "environment"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("dvba", dvba, "real", "out", "environment"))
    store.define(Field("ppx", ppx, "real", "init/out", "euler"))
    store.define(Field("qqx", qqx, "real", "init/out", "euler"))
    store.define(Field("rrx", rrx, "real", "init/out", "euler"))
    store.define(Field("vmass", vmass, "real", "out", "propulsion"))
    store.define(Field("IBBB", ibbb, "mat", "out", "propulsion"))
    store.define(Field("delax", delax, "real", "out", "actuator"))
    store.define(Field("delex", delex, "real", "out", "actuator"))
    store.define(Field("delrx", delrx, "real", "out", "actuator"))


def _ready(*, maero=1, **kw):
    data = {
        "alpplimx": ALPPLIMX,
        "alpnlimx": ALPNLIMX,
        "strct_pos_limitx": STRCT_POS,
        "strct_neg_limitx": STRCT_NEG,
        "maero": maero,
    }
    for key in ("alpplimx", "alpnlimx", "strct_pos_limitx", "strct_neg_limitx", "maero"):
        if key in kw:
            data[key] = kw.pop(key)
    deck = _deck()
    vehicle = _Vehicle()
    aero = Hyper6Aero(deck)
    aero.define(vehicle)
    aero.initialize(vehicle, _ctx())
    _externals(vehicle.store, **kw)
    store = vehicle.store
    store.set("maero", data["maero"])
    store.set("alpplimx", data["alpplimx"])
    store.set("alpnlimx", data["alpnlimx"])
    store.set("strct_pos_limitx", data["strct_pos_limitx"])
    store.set("strct_neg_limitx", data["strct_neg_limitx"])
    return deck, vehicle, aero


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _expected(deck, store):
    look_up = deck.look_up
    alphax = store.get("alphax")
    betax = store.get("betax")
    vmach = store.get("vmach")
    pdynmc = store.get("pdynmc")
    dvba = store.get("dvba")
    ppx = store.get("ppx")
    qqx = store.get("qqx")
    rrx = store.get("rrx")
    vmass = store.get("vmass")
    refa = store.get("refa")
    refb = store.get("refb")
    refc = store.get("refc")
    alpplimx = store.get("alpplimx")
    strct_pos_limitx = store.get("strct_pos_limitx")
    alpnlimx = store.get("alpnlimx")
    strct_neg_limitx = store.get("strct_neg_limitx")
    delax = store.get("delax")
    delex = store.get("delex")
    delrx = store.get("delrx")
    ibbb = store.get("IBBB")

    cd0 = look_up("cd0_vs_alpha_mach", alphax, vmach)
    cda = look_up("cda_vs_alpha_mach", alphax, vmach)
    cd = cd0 + cda * alphax

    cl0 = look_up("cl0_vs_alpha_mach", alphax, vmach)
    cla = look_up("cla_vs_alpha_mach", alphax, vmach)
    clde = look_up("clde_vs_alpha_mach", alphax, vmach)
    cl = cl0 + cla * alphax + clde * delex
    clovercd = fabs(cl / cd)

    cyb = look_up("cyb_vs_alpha_mach", alphax, vmach)
    cyda = look_up("cyda_vs_alpha_mach", alphax, vmach)
    cydr = look_up("cydr_vs_alpha_mach", alphax, vmach)
    cy = cyb * betax + cyda * delax + cydr * delrx

    cllb = look_up("cllb_vs_alpha_mach", alphax, vmach)
    cllda = look_up("cllda_vs_alpha_mach", alphax, vmach)
    clldr = look_up("clldr_vs_alpha_mach", alphax, vmach)
    cllp = look_up("cllp_vs_alpha_mach", alphax, vmach)
    cllr = look_up("cllr_vs_alpha_mach", alphax, vmach)
    cll = (
        cllb * betax
        + cllda * delax
        + clldr * delrx
        + cllp * ppx * RAD * refb / (2 * dvba)
        + cllr * rrx * RAD * refb / (2 * dvba)
    )

    cm0 = look_up("cm0_vs_alpha_mach", alphax, vmach)
    cma = look_up("cma_vs_alpha_mach", alphax, vmach)
    cmde = look_up("cmde_vs_alpha_mach", alphax, vmach)
    cmq = look_up("cmq_vs_alpha_mach", alphax, vmach)
    clm = cm0 + cma * alphax + cmde * delex + cmq * qqx * RAD * refc / (2.0 * dvba)

    clnb = look_up("clnb_vs_alpha_mach", alphax, vmach)
    clnda = look_up("clnda_vs_alpha_mach", alphax, vmach)
    clndr = look_up("clndr_vs_alpha_mach", alphax, vmach)
    clnp = look_up("clnp_vs_alpha_mach", alphax, vmach)
    clnr = look_up("clnr_vs_alpha_mach", alphax, vmach)
    cln = (
        clnb * betax
        + clnda * delax
        + clndr * delrx
        + clnp * ppx * RAD * refb / (2.0 * dvba)
        + clnr * rrx * RAD * refb / (2.0 * dvba)
    )

    cosa = cos(alphax * RAD)
    sina = sin(alphax * RAD)
    cx = -cd * cosa + cl * sina
    cz = -cd * sina - cl * cosa

    cl0max = look_up("cl0_vs_alpha_mach", alpplimx, vmach)
    clamax = look_up("cla_vs_alpha_mach", alpplimx, vmach)
    clmax = cl0max + clamax * alpplimx
    almax = clmax * pdynmc * refa
    weight = vmass * AGRAV
    gmax = almax / weight
    if gmax >= strct_pos_limitx:
        gmax = strct_pos_limitx
    aload = cl * pdynmc * refa
    gg = aload / weight
    gavail_pos = gmax - gg

    cl0min = look_up("cl0_vs_alpha_mach", alpnlimx, vmach)
    clamin = look_up("cla_vs_alpha_mach", alpnlimx, vmach)
    clmin = cl0min + clamin * alpnlimx
    almin = clmin * pdynmc * refa
    gminx = almin / weight
    if gminx <= strct_neg_limitx:
        gminx = strct_neg_limitx
    gavail_neg = gminx - gg

    ibbb11 = ibbb[0, 0]
    ibbb22 = ibbb[1, 1]
    ibbb33 = ibbb[2, 2]
    duml = (pdynmc * refa / vmass) / RAD
    dla = duml * cla
    dlde = duml * clde
    dumm = pdynmc * refa * refc / ibbb22
    dma = dumm * cma / RAD
    dmq = dumm * (refc / (2.0 * dvba)) * cmq
    dmde = dumm * cmde / RAD

    dumy = pdynmc * refa / vmass
    dyb = dumy * cyb / RAD
    dydr = dumy * cydr / RAD
    dumn = pdynmc * refa * refb / ibbb33
    dnb = dumn * clnb / RAD
    dnr = dumn * (refb / (2.0 * dvba)) * clnr
    dndr = dumn * clndr / RAD

    dumll = pdynmc * refa * refb / ibbb11
    dllp = dumll * (refb / (2.0 * dvba)) * cllp
    dllda = dumll * cllda / RAD

    stmarg = 0.0
    if cla:
        stmarg = -cma / cla

    a11 = dmq
    a12 = dma / dla
    a21 = dla
    a22 = -dla / dvba
    arg = (a11 + a22) ** 2 - 4.0 * (a11 * a22 - a12 * a21)
    if arg >= 0.0:
        wnp = 0.0
        zetp = 0.0
        dum = a11 + a22
        realp1 = (dum + sqrt(arg)) / 2.0
        realp2 = (dum - sqrt(arg)) / 2.0
        rpreal = (realp1 + realp2) / 2.0
    else:
        realp1 = 0.0
        realp2 = 0.0
        wnp = sqrt(a11 * a22 - a12 * a21)
        zetp = -(a11 + a22) / (2.0 * wnp)
        rpreal = -zetp * wnp

    a11 = dnr
    a12 = dnb / dyb
    a21 = -dyb
    a22 = dyb / dvba
    arg = (a11 + a22) ** 2 - 4.0 * (a11 * a22 - a12 * a21)
    if arg >= 0.0:
        wny = 0.0
        zety = 0.0
        dum = a11 + a22
        realy1 = (dum + sqrt(arg)) / 2.0
        realy2 = (dum - sqrt(arg)) / 2.0
        ryreal = (realy1 + realy2) / 2.0
    else:
        realy1 = 0.0
        realy2 = 0.0
        wny = sqrt(a11 * a22 - a12 * a21)
        zety = -(a11 + a22) / (2.0 * wny)
        ryreal = -zety * wny

    return {
        "cd": cd,
        "cl": cl,
        "cy": cy,
        "cll": cll,
        "clm": clm,
        "cln": cln,
        "cx": cx,
        "cz": cz,
        "cd0": cd0,
        "cda": cda,
        "cl0": cl0,
        "cla": cla,
        "clde": clde,
        "cyb": cyb,
        "cyda": cyda,
        "cydr": cydr,
        "cllb": cllb,
        "cllda": cllda,
        "cllp": cllp,
        "cllr": cllr,
        "cm0": cm0,
        "cma": cma,
        "cmde": cmde,
        "cmq": cmq,
        "clnb": clnb,
        "clnda": clnda,
        "clndr": clndr,
        "clnp": clnp,
        "clnr": clnr,
        "clldr": clldr,
        "clovercd": clovercd,
        "gmax": gmax,
        "gminx": gminx,
        "gavail_pos": gavail_pos,
        "gavail_neg": gavail_neg,
        "stmarg": stmarg,
        "dla": dla,
        "dlde": dlde,
        "dma": dma,
        "dmq": dmq,
        "dmde": dmde,
        "dyb": dyb,
        "dydr": dydr,
        "dnb": dnb,
        "dnr": dnr,
        "dndr": dndr,
        "dllp": dllp,
        "dllda": dllda,
        "realp1": realp1,
        "realp2": realp2,
        "wnp": wnp,
        "zetp": zetp,
        "rpreal": rpreal,
        "realy1": realy1,
        "realy2": realy2,
        "wny": wny,
        "zety": zety,
        "ryreal": ryreal,
    }


def test_name_is_aerodynamics():
    assert Hyper6Aero(_deck()).name == "aerodynamics"


def test_define_registers_cpp_fields_not_externals():
    vehicle = _Vehicle()
    Hyper6Aero(_deck()).define(vehicle)
    store = vehicle.store
    for name in DEFINED:
        assert name in store.names(), name
        field = store.field(name)
        assert field.module == "aerodynamics"
        assert field.role == ROLES[name], name
        if name == "maero":
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
    assert "tralppx" not in store.names()
    assert "tralpnx" not in store.names()
    assert "trbetx" not in store.names()
    assert "alphax" not in store.names()
    assert "vmach" not in store.names()
    assert "pdynmc" not in store.names()


def test_initialize_sets_ghame_refs_and_termination():
    vehicle = _Vehicle()
    aero = Hyper6Aero(_deck())
    aero.define(vehicle)
    aero.initialize(vehicle, _ctx())
    store = vehicle.store
    assert store.get("refa") == 557.42
    assert store.get("refb") == 24.38
    assert store.get("refc") == 22.86
    assert store.get("trmach") == 0.8
    assert store.get("trdynm") == 10.0e3
    assert store.get("trload") == 3.0
    assert store.get("tralp") == 21.0
    assert store.get("trcode") == 0.0
    assert store.get("tmcode") == 0.0
    assert store.get("refa_st") == 7.0
    assert store.get("caa") == 0.4
    assert store.get("maero") == 0
    assert store.get("alpplimx") == 0.0
    assert store.get("alpnlimx") == 0.0


def test_climb_maero_1_cx_cz_and_der_finite():
    deck, vehicle, aero = _ready(maero=1, alphax=2.5)
    store = vehicle.store
    vmach, pdynmc = _us76_air()
    assert _approx(store.get("vmach"), vmach)
    assert _approx(store.get("dvba"), 1000.0)
    assert store.get("alphax") == 2.5
    aero.execute(vehicle, _ctx())
    assert np.isfinite(store.get("cx"))
    assert np.isfinite(store.get("cz"))
    assert store.get("cx") != 0.0
    assert store.get("cz") != 0.0
    for name in DER_FIELDS:
        assert np.isfinite(store.get(name)), name
    want = _expected(deck, store)
    assert _approx(store.get("cx"), want["cx"])
    assert _approx(store.get("cz"), want["cz"])


_MAERO2_ZERO = (
    "cy",
    "cll",
    "clm",
    "cln",
    "cz",
    "gmax",
    "gminx",
    "cd",
    "cl",
    "cd0",
    "cda",
    "cl0",
    "cla",
    "clde",
    "cyb",
    "cyda",
    "cydr",
    "cllb",
    "cllda",
    "cllp",
    "cllr",
    "cm0",
    "cma",
    "cmde",
    "cmq",
    "clnb",
    "clnda",
    "clndr",
    "clnp",
    "clnr",
    "clldr",
    "clovercd",
    "gavail_pos",
    "gavail_neg",
)


def test_maero_2_after_maero_1_zeros_coefficients_keeps_derivatives():
    _deck_obj, vehicle, aero = _ready(maero=1)
    aero.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("cd") != 0.0
    dla = store.get("dla")
    assert dla != 0.0
    store.set("maero", 2)
    aero.execute(vehicle, _ctx())
    assert store.get("refa") == store.get("refa_st")
    assert _approx(store.get("cx"), -store.get("caa"))
    for name in _MAERO2_ZERO:
        assert store.get(name) == 0.0, name
    assert store.get("dla") == dla


def test_other_maero_raises():
    for maero in (0, 3, 99, -1):
        _deck_obj, vehicle, aero = _ready(maero=maero)
        with pytest.raises(ValueError):
            aero.execute(vehicle, _ctx())


def test_execute_matches_cadac_formulas():
    deck, vehicle, aero = _ready(
        alphax=2.5,
        betax=1.0,
        delex=4.0,
        delax=5.0,
        delrx=3.0,
        ppx=10.0,
        qqx=5.0,
        rrx=8.0,
    )
    store = vehicle.store
    want = _expected(deck, store)
    aero.execute(vehicle, _ctx())
    for name in FORMULA_NAMES:
        assert np.isfinite(store.get(name)), name
        assert _approx(store.get(name), want[name]), name
    assert store.get("trcode") == 0.0
    assert store.get("refa") == 557.42
