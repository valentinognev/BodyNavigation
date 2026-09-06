from math import cos, sin
from pathlib import Path

import numpy as np
import pytest

from cadac.constants import AGRAV, RAD
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.plane6.aero import Plane6Aero

FALCON6 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/FALCON6_250201/FALCON6"
AERO = FALCON6 / "f16_aero_deck.asc"

RTOL = 1e-12
ATOL = 1e-14

ALPHAX = 1.0
BETAX = 0.0
DELEX = 0.0
DELAX = 0.0
DELRX = 0.0
PPX = 0.0
QQX = 0.0
RRX = 0.0
DVBA = 180.0
PDYNMC = 17000.0
VMACH = 0.53
XCG = 1.0
XCGR = 1.0
ALPLIMPX = 16.0
ALPLIMNX = -6.0

F16_IBBB = np.array(
    [
        [12875.0, 0.0, -1331.4],
        [0.0, 75673.0, 0.0],
        [-1331.4, 0.0, 85551.0],
    ],
    dtype=float,
)

DEFINED = (
    "refa",
    "refb",
    "refc",
    "cd",
    "cl",
    "cxt",
    "cyt",
    "czt",
    "clt",
    "cmt",
    "cnt",
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
    "cdrag",
    "clift",
    "alplimpx",
    "gmax",
    "alplimnx",
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
    "tralppx",
    "tralpnx",
    "trbetx",
    "vmass",
    "IBBB",
    "eng_ang_mom",
    "xcg",
    "xcgr",
)

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
    "delax",
    "delex",
    "delrx",
)

UNASSIGNED_LOCALS = (
    "cd",
    "cd0",
    "cda",
    "cl0",
    "cla",
    "cyda",
    "cllb",
    "cllr",
    "cm0",
    "cmde",
    "clnda",
    "clnp",
    "clldr",
)

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
    pdynmc=PDYNMC,
    vmach=VMACH,
):
    store.define(Field("time", 0.0, "real", "exec", "newton"))
    store.define(Field("alphax", alphax, "real", "diag", "kinematics"))
    store.define(Field("betax", betax, "real", "diag", "kinematics"))
    store.define(Field("vmach", vmach, "real", "out", "environment"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("dvba", dvba, "real", "out", "environment"))
    store.define(Field("ppx", ppx, "real", "init/out", "euler"))
    store.define(Field("qqx", qqx, "real", "init/out", "euler"))
    store.define(Field("rrx", rrx, "real", "init/out", "euler"))
    store.define(Field("delax", delax, "real", "out", "actuator"))
    store.define(Field("delex", delex, "real", "out", "actuator"))
    store.define(Field("delrx", delrx, "real", "out", "actuator"))


def _ready(**kw):
    data = {
        "xcg": XCG,
        "xcgr": XCGR,
        "alplimpx": ALPLIMPX,
        "alplimnx": ALPLIMNX,
    }
    for key in ("xcg", "xcgr", "alplimpx", "alplimnx"):
        if key in kw:
            data[key] = kw.pop(key)
    deck = _deck()
    vehicle = _Vehicle()
    aero = Plane6Aero(deck)
    aero.define(vehicle)
    aero.initialize(vehicle, _ctx())
    _externals(vehicle.store, **kw)
    store = vehicle.store
    store.set("xcg", data["xcg"])
    store.set("xcgr", data["xcgr"])
    store.set("alplimpx", data["alplimpx"])
    store.set("alplimnx", data["alplimnx"])
    return deck, vehicle, aero


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _expected(deck, store):
    refa = store.get("refa")
    refb = store.get("refb")
    refc = store.get("refc")
    vmass = store.get("vmass")
    alphax = store.get("alphax")
    betax = store.get("betax")
    pdynmc = store.get("pdynmc")
    dvba = store.get("dvba")
    ppx = store.get("ppx")
    qqx = store.get("qqx")
    rrx = store.get("rrx")
    delax = store.get("delax")
    delex = store.get("delex")
    delrx = store.get("delrx")
    xcg = store.get("xcg")
    xcgr = store.get("xcgr")
    alplimpx = store.get("alplimpx")
    alplimnx = store.get("alplimnx")
    trload = store.get("trload")
    trcode = store.get("trcode")

    c2v = refc / (2 * dvba)
    b2v = refb / (2 * dvba)

    cx = deck.look_up("cx_vs_elev_alpha", delex, alphax)
    cxq = deck.look_up("cxq_vs_alpha", alphax)
    cxt = cx + c2v * cxq * qqx * RAD

    cyr = deck.look_up("cyr_vs_alpha", alphax)
    cyp = deck.look_up("cyp_vs_alpha", alphax)
    cyt = (
        -0.02 * betax
        + 0.021 * delax / 20
        + 0.086 * delrx / 30
        + b2v * (cyr * rrx * RAD + cyp * ppx * RAD)
    )

    cz = deck.look_up("cz_vs_alpha", alphax)
    czq = deck.look_up("czq_vs_alpha", alphax)
    czt = cz * (1 - (betax * RAD) ** 2) - 0.19 * delex / 25 + c2v * czq * qqx * RAD

    cl = deck.look_up("cl_vs_beta_alpha", betax, alphax)
    cldr = deck.look_up("cldr_vs_beta_alpha", betax, alphax)
    clda = deck.look_up("clda_vs_beta_alpha", betax, alphax)
    clda = -clda
    clr = deck.look_up("clr_vs_alpha", alphax)
    clp = deck.look_up("clp_vs_alpha", alphax)
    cllr = 0.0
    clt = cl + clda * delax / 20 + cldr * delrx / 30 + b2v * (cllr * rrx * RAD + clp * ppx * RAD)

    cm = deck.look_up("cm_vs_elev_alpha", delex, alphax)
    cmq = deck.look_up("cmq_vs_alpha", alphax)
    cmt = cm + c2v * cmq * qqx * RAD + czt * (xcgr - xcg) / refc

    cn = deck.look_up("cn_vs_beta_alpha", betax, alphax)
    cnda = deck.look_up("cnda_vs_beta_alpha", betax, alphax)
    cndr = deck.look_up("cndr_vs_beta_alpha", betax, alphax)
    cnr = deck.look_up("cnr_vs_alpha", alphax)
    cnp = deck.look_up("cnp_vs_alpha", alphax)
    cnt = (
        cn
        + cnda * delax / 20
        + cndr * delrx / 30
        - cyt * (xcgr - xcg) / refb
        + b2v * (cnr * rrx * RAD + cnp * ppx * RAD)
    )

    czp = deck.look_up("cz_vs_alpha", alplimpx)
    czn = deck.look_up("cz_vs_alpha", alplimnx)
    alpx = -czp * pdynmc * refa
    alnx = -czn * pdynmc * refa
    weight = vmass * AGRAV
    gmax = alpx / weight
    gminx = alnx / weight

    cosa = cos(alphax * RAD)
    sina = sin(alphax * RAD)
    cdrag = -cxt * cosa - czt * sina
    clift = cxt * sina - czt * cosa
    clovercd = clift / cdrag

    clde = 0.19 / 25
    cyb = -0.02 * RAD
    cydr = -0.086 / 30
    clnr = b2v * cnr
    clndr = cndr / 30
    cllp = b2v * clp
    cllda = clda / 20

    if gmax < trload:
        trcode = 4

    return {
        "cxt": cxt,
        "cyt": cyt,
        "czt": czt,
        "clt": clt,
        "cmt": cmt,
        "cnt": cnt,
        "cdrag": cdrag,
        "clift": clift,
        "clovercd": clovercd,
        "gmax": gmax,
        "gminx": gminx,
        "cl": cl,
        "clde": clde,
        "cyb": cyb,
        "cydr": cydr,
        "cmq": cmq,
        "clnr": clnr,
        "clndr": clndr,
        "cllp": cllp,
        "cllda": cllda,
        "trcode": trcode,
        "cx": cx,
        "cz": cz,
        "clr": clr,
        "cllr": cllr,
    }


def test_name_is_aerodynamics():
    assert Plane6Aero(_deck()).name == "aerodynamics"


def test_define_registers_cpp_fields_not_externals():
    vehicle = _Vehicle()
    Plane6Aero(_deck()).define(vehicle)
    store = vehicle.store
    for name in DEFINED:
        assert name in store.names()
        if name == "IBBB":
            np.testing.assert_array_equal(store.get(name), np.zeros((3, 3)))
            assert store.field(name).type == "mat"
        else:
            assert store.get(name) == 0.0
    for name in EXTERNALS:
        assert name not in store.names()
    assert store.field("gmax").outputs == ("plot",)
    assert store.field("gminx").outputs == ("plot",)
    assert store.field("stmarg").outputs == ("plot",)
    assert store.field("dma").outputs == ("plot",)
    assert store.field("dmde").outputs == ("plot",)


def test_initialize_sets_f16_mass_geometry_and_limits():
    vehicle = _Vehicle()
    aero = Plane6Aero(_deck())
    aero.define(vehicle)
    aero.initialize(vehicle, _ctx())
    store = vehicle.store
    assert store.get("refa") == 27.87
    assert store.get("refb") == 9.14
    assert store.get("refc") == 3.45
    assert store.get("vmass") == 9496.0
    np.testing.assert_allclose(store.get("IBBB"), F16_IBBB, rtol=0, atol=0)
    assert store.get("eng_ang_mom") == 70000.0
    assert store.get("trmach") == 0.8
    assert store.get("trdynm") == 10e3
    assert store.get("trload") == 3.0
    assert store.get("tralppx") == 21.0
    assert store.get("tralpnx") == -6.0
    assert store.get("trbetx") == 5.0
    assert store.get("trcode") == 0.0
    assert store.get("tmcode") == 0.0
    assert store.get("xcg") == 0.0
    assert store.get("xcgr") == 0.0
    assert store.get("alplimpx") == 0.0
    assert store.get("alplimnx") == 0.0


def test_cxt_czt_at_alpha_1_elev_0():
    deck, vehicle, aero = _ready(alphax=1.0, delex=0.0, qqx=5.0)
    store = vehicle.store
    want = _expected(deck, store)
    aero.execute(vehicle, _ctx())
    assert _approx(store.get("cxt"), want["cxt"])
    assert _approx(store.get("czt"), want["czt"])
    assert _approx(store.get("cxt"), want["cx"] + (3.45 / (2 * 180.0)) * deck.look_up("cxq_vs_alpha", 1.0) * 5.0 * RAD)
    assert want["cz"] == deck.look_up("cz_vs_alpha", 1.0)


def test_execute_matches_cadac_formulas():
    deck, vehicle, aero = _ready(
        alphax=1.0,
        betax=2.0,
        delex=4.0,
        delax=5.0,
        delrx=3.0,
        ppx=10.0,
        qqx=5.0,
        rrx=8.0,
        xcg=0.9,
        xcgr=1.0,
    )
    store = vehicle.store
    want = _expected(deck, store)
    aero.execute(vehicle, _ctx())
    for name in (
        "cxt",
        "cyt",
        "czt",
        "clt",
        "cmt",
        "cnt",
        "cdrag",
        "clift",
        "clovercd",
        "gmax",
        "gminx",
        "cl",
        "clde",
        "cyb",
        "cydr",
        "cmq",
        "clnr",
        "clndr",
        "cllp",
        "cllda",
        "trcode",
    ):
        assert _approx(store.get(name), want[name]), name


def test_clt_uses_zero_cllr_not_clr_lookup():
    deck, vehicle, aero = _ready(alphax=1.0, betax=2.0, rrx=8.0, ppx=0.0, delax=0.0, delrx=0.0)
    store = vehicle.store
    want = _expected(deck, store)
    aero.execute(vehicle, _ctx())
    b2v = store.get("refb") / (2 * store.get("dvba"))
    clt_if_clr = want["cl"] + b2v * (want["clr"] * 8.0 * RAD)
    assert want["clr"] != 0.0
    assert not _approx(want["clt"], clt_if_clr)
    assert _approx(store.get("clt"), want["clt"])
    assert store.get("cllr") == 0.0


def test_diagnostic_cl_is_rolling_moment_lookup_not_lift():
    deck, vehicle, aero = _ready(alphax=1.0, betax=2.0, delex=4.0)
    store = vehicle.store
    want = _expected(deck, store)
    aero.execute(vehicle, _ctx())
    assert _approx(store.get("cl"), want["cl"])
    assert not _approx(store.get("cl"), want["clift"])
    assert _approx(store.get("clift"), want["clift"])


def test_unassigned_table_locals_stay_zero():
    _, vehicle, aero = _ready(alphax=1.0, betax=2.0, delex=4.0, delax=5.0, delrx=3.0, ppx=10.0, qqx=5.0, rrx=8.0)
    aero.execute(vehicle, _ctx())
    store = vehicle.store
    for name in UNASSIGNED_LOCALS:
        assert store.get(name) == 0.0, name


def test_trcode_4_when_gmax_below_trload():
    _, vehicle, aero = _ready(alphax=1.0, pdynmc=100.0, alplimpx=16.0)
    store = vehicle.store
    deck = aero.deck
    want = _expected(deck, store)
    assert want["gmax"] < store.get("trload")
    aero.execute(vehicle, _ctx())
    assert store.get("trcode") == 4.0
    assert _approx(store.get("gmax"), want["gmax"])


def test_trcode_stays_zero_when_gmax_meets_trload():
    _, vehicle, aero = _ready(alphax=1.0, pdynmc=PDYNMC, alplimpx=16.0)
    store = vehicle.store
    want = _expected(aero.deck, store)
    assert want["gmax"] >= store.get("trload")
    aero.execute(vehicle, _ctx())
    assert store.get("trcode") == 0.0
