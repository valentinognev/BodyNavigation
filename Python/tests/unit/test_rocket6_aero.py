from math import cos, sin, sqrt
from pathlib import Path

import numpy as np
import pytest

from cadac.constants import AGRAV, RAD
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.round6.rocket6.aero import Rocket6Aero

ROCKET6 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/ROCKET6_250122/ROCKET6"
AERO = ROCKET6 / "aero_deck_SLV.asc"

RTOL = 1e-12
ATOL = 1e-14

# insertion stage-1 aero data (input.asc)
REFA = 3.243
REFD = 2.032
XCG_REF = 8.6435
ALPLIMX = 20.0
ALIMITX = 5.0
VMASS = 48984.0
XCG = 10.53
VMACH = 0.5
ALPPX = 2.0
PHIPX = 0.0
MPROP = 3
DVBA = 170.0
PDYNMC = 5000.0
QQX = 0.0
RRX = 0.0
ALPHAX = 2.0

STAGE1_IBBB = np.array(
    [
        [21.94e3, 0.0, 0.0],
        [0.0, 671.62e3, 0.0],
        [0.0, 0.0, 671.62e3],
    ],
    dtype=float,
)

DEFINED = (
    "maero",
    "refa",
    "refd",
    "xcg_ref",
    "cy",
    "cll",
    "clm",
    "cln",
    "cx",
    "cz",
    "ca0",
    "caa",
    "cn0",
    "clm0",
    "clmq",
    "cla",
    "clde",
    "cyb",
    "cydr",
    "cllda",
    "cllp",
    "cma",
    "cmde",
    "cmq",
    "cnb",
    "cndr",
    "cnr",
    "stmarg_yaw",
    "stmarg_pitch",
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
    "alplimx",
    "alimitx",
    "gnavail",
    "gyavail",
    "gnmax",
    "gymax",
)

ROLES = {
    "maero": "data",
    "refa": "init",
    "refd": "init",
    "xcg_ref": "init",
    "cy": "out",
    "cll": "out",
    "clm": "out",
    "cln": "out",
    "cx": "out",
    "cz": "out",
    "ca0": "diag",
    "caa": "diag",
    "cn0": "diag",
    "clm0": "diag",
    "clmq": "diag",
    "cla": "out",
    "clde": "diag",
    "cyb": "diag",
    "cydr": "diag",
    "cllda": "diag",
    "cllp": "diag",
    "cma": "diag",
    "cmde": "diag",
    "cmq": "diag",
    "cnb": "diag",
    "cndr": "diag",
    "cnr": "diag",
    "stmarg_yaw": "diag",
    "stmarg_pitch": "diag",
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
    "alplimx": "data",
    "alimitx": "data",
    "gnavail": "diag",
    "gyavail": "diag",
    "gnmax": "out",
    "gymax": "out",
}

PLOT_FLAGGED = (
    "stmarg_yaw",
    "stmarg_pitch",
    "realp1",
    "wnp",
    "zetp",
    "realy1",
    "wny",
    "zety",
    "gnmax",
    "gymax",
)

SCRN_FLAGGED = ("maero",)

INT_FIELDS = ("maero",)

# kinematics / env / propulsion / actuator / TVC names aero reads but does not own
EXTERNALS = (
    "alphax",
    "alppx",
    "phipx",
    "vmach",
    "pdynmc",
    "dvba",
    "qqx",
    "rrx",
    "vmass",
    "IBBB",
    "xcg",
    "mprop",
    "thrust",
    "mtvc",
    "gtvc",
    "parm",
    "time",
    "betax",
    "ppx",
    "rho",
    "alt",
    "delax",
    "delex",
    "delrx",
)

GHAME_NOT_OWNED = ("refb", "refc")

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
    "stmarg_yaw",
    "stmarg_pitch",
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
    "cx",
    "cy",
    "cz",
    "cll",
    "clm",
    "cln",
    "ca0",
    "caa",
    "cn0",
    "clm0",
    "clmq",
    "cla",
    "cma",
    "clde",
    "cyb",
    "cydr",
    "cllda",
    "cllp",
    "cmde",
    "cmq",
    "cnb",
    "cndr",
    "cnr",
    "gnmax",
    "gymax",
    "gnavail",
    "gyavail",
) + DER_FIELDS


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.001,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _deck():
    _, tables = parse_asc_deck(AERO)
    return Datadeck.from_tables(tables)


def _slv(maero):
    return {13: 3, 12: 2, 11: 1}[maero]


def _externals(
    store,
    *,
    alphax=ALPHAX,
    alppx=ALPPX,
    phipx=PHIPX,
    vmach=VMACH,
    pdynmc=PDYNMC,
    dvba=DVBA,
    qqx=QQX,
    rrx=RRX,
    vmass=VMASS,
    xcg=XCG,
    mprop=MPROP,
    ibbb=None,
    thrust=0.0,
    mtvc=0,
    gtvc=1.0,
    parm=16.84,
):
    if ibbb is None:
        ibbb = STAGE1_IBBB
    store.define(Field("alphax", alphax, "real", "init/diag", "kinematics"))
    store.define(Field("alppx", alppx, "real", "diag", "kinematics"))
    store.define(Field("phipx", phipx, "real", "diag", "kinematics"))
    store.define(Field("vmach", vmach, "real", "out", "environment"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("dvba", dvba, "real", "out", "environment"))
    store.define(Field("qqx", qqx, "real", "init/out", "euler"))
    store.define(Field("rrx", rrx, "real", "init/out", "euler"))
    store.define(Field("vmass", vmass, "real", "out", "propulsion"))
    store.define(Field("IBBB", ibbb, "mat", "out", "propulsion"))
    store.define(Field("xcg", xcg, "real", "out", "propulsion"))
    store.define(Field("mprop", mprop, "int", "data", "propulsion"))
    store.define(Field("thrust", thrust, "real", "out", "propulsion"))
    store.define(Field("mtvc", mtvc, "int", "data", "tvc"))
    store.define(Field("gtvc", gtvc, "real", "data", "tvc"))
    store.define(Field("parm", parm, "real", "data", "tvc"))


def _ready(*, maero=13, **kw):
    data = {
        "maero": maero,
        "refa": REFA,
        "refd": REFD,
        "xcg_ref": XCG_REF,
        "alplimx": ALPLIMX,
        "alimitx": ALIMITX,
    }
    for key in list(data):
        if key in kw:
            data[key] = kw.pop(key)
    deck = _deck()
    vehicle = _Vehicle()
    aero = Rocket6Aero(deck)
    aero.define(vehicle)
    aero.initialize(vehicle, _ctx())
    _externals(vehicle.store, **kw)
    store = vehicle.store
    store.set("maero", data["maero"])
    store.set("refa", data["refa"])
    store.set("refd", data["refd"])
    store.set("xcg_ref", data["xcg_ref"])
    store.set("alplimx", data["alplimx"])
    store.set("alimitx", data["alimitx"])
    return deck, vehicle, aero


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _expected(deck, store):
    look_up = deck.look_up
    maero = store.get("maero")
    n = _slv(maero)
    alppx = store.get("alppx")
    phipx = store.get("phipx")
    vmach = store.get("vmach")
    pdynmc = store.get("pdynmc")
    dvba = store.get("dvba")
    qqx = store.get("qqx")
    rrx = store.get("rrx")
    vmass = store.get("vmass")
    xcg = store.get("xcg")
    mprop = store.get("mprop")
    refa = store.get("refa")
    refd = store.get("refd")
    xcg_ref = store.get("xcg_ref")
    alplimx = store.get("alplimx")
    alimitx = store.get("alimitx")
    ibbb = store.get("IBBB")
    cla = store.get("cla")
    cma = store.get("cma")
    names = store.names()
    mtvc = store.get("mtvc") if "mtvc" in names else 0
    thrust = store.get("thrust") if "thrust" in names else 0.0
    gtvc = store.get("gtvc") if "gtvc" in names else 0.0
    parm = store.get("parm") if "parm" in names else 0.0

    phip = phipx * RAD
    cphip = cos(phip)
    sphip = sin(phip)
    qqax = qqx * cphip - rrx * sphip

    ca0 = look_up(f"ca0slv{n}_vs_mach", vmach)
    caa = look_up(f"caaslv{n}_vs_mach", vmach)
    ca0b = look_up(f"ca0bslv{n}_vs_mach", vmach)
    thrust_on = 1.0 if mprop else 0.0
    ca = ca0 + caa * alppx + thrust_on * ca0b

    cn0 = look_up(f"cn0slv{n}_vs_mach_alpha", vmach, alppx)
    cna = cn0
    clm0 = look_up(f"clm0slv{n}_vs_mach_alpha", vmach, alppx)
    clmq = look_up(f"clmqslv{n}_vs_mach", vmach)
    clmaref = clm0 + clmq * qqax * refd / (2.0 * dvba)
    clma = clmaref - cna * (xcg_ref - xcg) / refd

    alplx = alppx + 3.0
    alpmx = alppx - 3.0
    if alpmx < 0.0:
        alpmx = 0.0
    cn0p = look_up(f"cn0slv{n}_vs_mach_alpha", vmach, alplx)
    cn0m = look_up(f"cn0slv{n}_vs_mach_alpha", vmach, alpmx)
    if alplx < alplimx:
        cla = (cn0p - cn0m) / (alplx - alpmx)
    clm0p = look_up(f"clm0slv{n}_vs_mach_alpha", vmach, alplx)
    clm0m = look_up(f"clm0slv{n}_vs_mach_alpha", vmach, alpmx)
    if alppx < alplimx:
        cma = (clm0p - clm0m) / (alplx - alpmx) - cla * (xcg_ref - xcg) / refd

    cx = -ca
    cy = -cna * sphip
    cz = -cna * cphip
    cll = 0.0
    clm = clma * cphip
    cln = -clma * sphip

    cn0mx = look_up(f"cn0slv{n}_vs_mach_alpha", vmach, alplimx)
    anlmx = cn0mx * pdynmc * refa
    weight = vmass * AGRAV
    gnmax = anlmx / weight
    if gnmax >= alimitx:
        gnmax = alimitx
    cn = 0.0
    aloadn = cn * pdynmc * refa
    gng = aloadn / weight
    gnavail = gnmax - gng
    gymax = gnmax
    gyavail = gnavail

    clde = 0.0
    cyb = -cla
    cydr = 0.0
    cllda = 0.0
    cllp = 0.0
    cmde = 0.0
    cmq = clmq
    cnb = -cma
    cndr = 0.0
    cnr = clmq

    ibbb11 = ibbb[0, 0]
    ibbb22 = ibbb[1, 1]
    ibbb33 = ibbb[2, 2]
    duml = (pdynmc * refa / vmass) / RAD
    dla = duml * cla
    dlde = duml * clde
    dumm = pdynmc * refa * refd / ibbb22
    dma = dumm * cma / RAD
    dmq = dumm * (refd / (2 * dvba)) * cmq
    dmde = dumm * cmde / RAD

    dumy = pdynmc * refa / vmass
    dyb = dumy * cyb / RAD
    dydr = dumy * cydr / RAD
    dumn = pdynmc * refa * refd / ibbb33
    dnb = dumn * cnb / RAD
    dnr = dumn * (refd / (2 * dvba)) * cnr
    dndr = dumn * cndr / RAD

    dumll = pdynmc * refa * refd / ibbb11
    dllp = dumll * (refd / (2 * dvba)) * cllp
    dllda = dumll * cllda / RAD

    if mtvc == 1 or mtvc == 2 or mtvc == 3:
        dlde = gtvc * thrust / vmass
        dmde = -(parm - xcg) * gtvc * thrust / ibbb[2, 2]
        dydr = dlde
        dndr = dmde

    stmarg_pitch = 0.0
    if cla:
        stmarg_pitch = -cma / cla
    stmarg_yaw = 0.0
    if cyb:
        stmarg_yaw = -cnb / cyb

    a11 = dmq
    a12 = 0.0
    if dla:
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
    if dyb:
        a12 = dnb / dyb
    else:
        a12 = 0.0
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
        "ca": ca,
        "ca0": ca0,
        "caa": caa,
        "ca0b": ca0b,
        "cn0": cn0,
        "clm0": clm0,
        "clmq": clmq,
        "cla": cla,
        "cma": cma,
        "cx": cx,
        "cy": cy,
        "cz": cz,
        "cll": cll,
        "clm": clm,
        "cln": cln,
        "clde": clde,
        "cyb": cyb,
        "cydr": cydr,
        "cllda": cllda,
        "cllp": cllp,
        "cmde": cmde,
        "cmq": cmq,
        "cnb": cnb,
        "cndr": cndr,
        "cnr": cnr,
        "gnmax": gnmax,
        "gymax": gymax,
        "gnavail": gnavail,
        "gyavail": gyavail,
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
        "stmarg_pitch": stmarg_pitch,
        "stmarg_yaw": stmarg_yaw,
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
    assert Rocket6Aero(_deck()).name == "aerodynamics"


def test_define_registers_cpp_fields_not_externals():
    vehicle = _Vehicle()
    Rocket6Aero(_deck()).define(vehicle)
    store = vehicle.store
    for name in DEFINED:
        assert name in store.names(), name
        field = store.field(name)
        assert field.module == "aerodynamics"
        assert field.role == ROLES[name], name
        if name in INT_FIELDS:
            assert field.type == "int"
            assert store.get(name) == 0
        else:
            assert field.type == "real"
            assert store.get(name) == 0.0
        if name in PLOT_FLAGGED:
            assert field.outputs == ("plot",), name
        elif name in SCRN_FLAGGED:
            assert field.outputs == ("scrn",), name
        else:
            assert field.outputs == (), name
    for name in EXTERNALS:
        assert name not in store.names()
    for name in GHAME_NOT_OWNED:
        assert name not in store.names()
    assert "alphax" not in store.names()
    assert "vmach" not in store.names()
    assert "pdynmc" not in store.names()
    assert "dvba" not in store.names()
    assert "vmass" not in store.names()
    assert "IBBB" not in store.names()
    assert "xcg" not in store.names()


def test_initialize_sets_termination_not_refs():
    vehicle = _Vehicle()
    aero = Rocket6Aero(_deck())
    aero.define(vehicle)
    aero.initialize(vehicle, _ctx())
    store = vehicle.store
    assert store.get("trmach") == 0.8
    assert store.get("trdynm") == 10.0e3
    assert store.get("trload") == 3.0
    assert store.get("tralp") == 21.0
    assert store.get("trcode") == 0.0
    assert store.get("tmcode") == 0.0
    assert store.get("refa") == 0.0
    assert store.get("refd") == 0.0
    assert store.get("xcg_ref") == 0.0
    assert store.get("maero") == 0
    assert store.get("alplimx") == 0.0
    assert store.get("alimitx") == 0.0
    assert "refb" not in store.names()
    assert "refc" not in store.names()


def test_maero_13_finite_cx_cz_clm_and_replica_cx_neg_ca():
    # Break: wrong SLV3 tables, missing ca0b with mprop, or cx=+ca.
    deck, vehicle, aero = _ready(
        maero=13,
        vmach=VMACH,
        alppx=ALPPX,
        phipx=PHIPX,
        mprop=MPROP,
    )
    store = vehicle.store
    aero.execute(vehicle, _ctx())
    assert np.isfinite(store.get("cx"))
    assert np.isfinite(store.get("cz"))
    assert np.isfinite(store.get("clm"))
    assert store.get("cx") != 0.0
    assert store.get("cz") != 0.0
    look_up = deck.look_up
    ca0 = look_up("ca0slv3_vs_mach", VMACH)
    caa = look_up("caaslv3_vs_mach", VMACH)
    ca0b = look_up("ca0bslv3_vs_mach", VMACH)
    ca = ca0 + caa * ALPPX + ca0b
    assert _approx(store.get("cx"), -ca)
    assert store.get("cx") == -ca


def test_maero_1_raises():
    _deck_obj, vehicle, aero = _ready(maero=1)
    with pytest.raises(ValueError):
        aero.execute(vehicle, _ctx())


def test_other_maero_raises():
    for maero in (0, 2, 10, 14, -1):
        _deck_obj, vehicle, aero = _ready(maero=maero)
        with pytest.raises(ValueError):
            aero.execute(vehicle, _ctx())


def test_execute_matches_cadac_formulas():
    deck, vehicle, aero = _ready(
        maero=13,
        alppx=2.0,
        phipx=30.0,
        qqx=5.0,
        rrx=8.0,
        mprop=3,
    )
    store = vehicle.store
    want = _expected(deck, store)
    aero.execute(vehicle, _ctx())
    for name in FORMULA_NAMES:
        assert np.isfinite(store.get(name)), name
        assert _approx(store.get(name), want[name]), name
    assert store.get("trcode") == 0.0
    assert store.get("refa") == REFA
    assert store.get("refd") == REFD


def test_maero_11_uses_slv1_tables():
    # Break: last-stage still looking up slv3/slv2.
    deck, vehicle, aero = _ready(maero=11, vmach=VMACH, alppx=ALPPX, mprop=3)
    store = vehicle.store
    aero.execute(vehicle, _ctx())
    want = _expected(deck, store)
    assert _approx(store.get("cx"), want["cx"])
    slv3 = deck.look_up("ca0slv3_vs_mach", VMACH)
    slv1 = deck.look_up("ca0slv1_vs_mach", VMACH)
    assert slv1 != slv3
    assert not _approx(store.get("ca0"), slv3)
    assert _approx(store.get("ca0"), slv1)


def test_mtvc2_overrides_control_derivatives():
    deck, vehicle, aero = _ready(
        maero=13,
        mtvc=2,
        thrust=1.4e6,
        gtvc=1.0,
        parm=16.84,
    )
    store = vehicle.store
    want = _expected(deck, store)
    aero.execute(vehicle, _ctx())
    assert _approx(store.get("dlde"), want["dlde"])
    assert _approx(store.get("dmde"), want["dmde"])
    assert _approx(store.get("dydr"), want["dydr"])
    assert _approx(store.get("dndr"), want["dndr"])
    assert store.get("dlde") != 0.0


def test_terminate_is_pass():
    _deck_obj, vehicle, aero = _ready()
    aero.terminate(vehicle, _ctx())
