from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from cadac.constants import RAD
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.flat6.falcon6.aero import Plane6Aero

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

DER_OUTPUTS = (
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
    "cma",
    "clnb",
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

AERO_UNASSIGNED = (
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


def _expected_der(deck, store):
    look_up = deck.look_up
    refa = store.get("refa")
    refb = store.get("refb")
    refc = store.get("refc")
    xcg = store.get("xcg")
    xcgr = store.get("xcgr")
    pdynmc = store.get("pdynmc")
    dvba = store.get("dvba")
    alphax = store.get("alphax")
    betax = store.get("betax")
    vmass = store.get("vmass")
    ibbb = store.get("IBBB")
    delex = store.get("delex")

    b2v = refb / (2 * dvba)

    clde = 0.19 / 25
    cyb = -0.02 * RAD
    cydr = -0.086 / 30
    cnr = look_up("cnr_vs_alpha", alphax)
    cndr = look_up("cndr_vs_beta_alpha", betax, alphax)
    clp = look_up("clp_vs_alpha", alphax)
    clda = -look_up("clda_vs_beta_alpha", betax, alphax)
    clnr = b2v * cnr
    clndr = cndr / 30
    cllp = b2v * clp
    cllda = clda / 20
    cmq = look_up("cmq_vs_alpha", alphax)

    czp = look_up("cz_vs_alpha", alphax + 1.5)
    czn = look_up("cz_vs_alpha", alphax - 1.5)
    cza = (czp - czn) / 3
    cla = -cza

    cmp = look_up("cm_vs_elev_alpha", delex, alphax + 1.5)
    cmn = look_up("cm_vs_elev_alpha", delex, alphax - 1.5)
    dum = (cmp - cmn) / 3
    cma = dum + cza * (xcgr - xcg) / refc

    cmp = look_up("cm_vs_elev_alpha", delex + 1.5, alphax)
    cmn = look_up("cm_vs_elev_alpha", delex - 1.5, alphax)
    cmde = (cmp - cmn) / 3

    cnp = look_up("cn_vs_beta_alpha", betax + 1.5, alphax)
    cnn = look_up("cn_vs_beta_alpha", betax - 1.5, alphax)
    dum = (cnp - cnn) / 3
    clnb = dum - cyb * (xcgr - xcg) / refb

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
        "cla_local": cla,
        "cza": cza,
        "cmde_local": cmde,
        "clde": clde,
        "cyb": cyb,
        "cydr": cydr,
        "cmq": cmq,
        "clnr": clnr,
        "clndr": clndr,
        "cllp": cllp,
        "cllda": cllda,
        "duml": duml,
        "dumm": dumm,
        "cma": cma,
        "clnb": clnb,
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
        "stmarg": stmarg,
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


def test_derivatives_finite_at_task7_flight_condition():
    deck, vehicle, aero = _ready(alphax=1.0, delex=0.0)
    store = vehicle.store
    want = _expected_der(deck, store)
    aero.execute(vehicle, _ctx())

    dla = store.get("dla")
    dma = store.get("dma")
    cma = store.get("cma")
    assert np.isfinite(dla)
    assert np.isfinite(dma)
    assert np.isfinite(cma)
    assert dla != 0.0
    assert dma != 0.0
    assert cma != 0.0

    assert _approx(dla, want["duml"] * (-want["cza"]))
    assert _approx(dma, want["dumm"] * cma / RAD)
    assert _approx(store.get("cla"), 0.0)


def test_execute_matches_cadac_der_formulas():
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
    want = _expected_der(deck, store)
    aero.execute(vehicle, _ctx())
    for name in DER_OUTPUTS:
        assert np.isfinite(store.get(name)), name
        assert _approx(store.get(name), want[name]), name
    assert _approx(store.get("dla"), want["duml"] * want["cla_local"])
    assert _approx(store.get("dma"), want["dumm"] * want["cma"] / RAD)
    assert store.get("cla") == 0.0
    assert store.get("cmde") == 0.0
    assert _approx(store.get("clde"), want["clde"])
    assert _approx(store.get("cyb"), want["cyb"])
    assert _approx(store.get("cydr"), want["cydr"])
    assert _approx(store.get("cmq"), want["cmq"])
    assert _approx(store.get("clnr"), want["clnr"])
    assert _approx(store.get("clndr"), want["clndr"])
    assert _approx(store.get("cllp"), want["cllp"])
    assert _approx(store.get("cllda"), want["cllda"])


def test_store_cla_and_cmde_stay_zero_after_der():
    _, vehicle, aero = _ready(alphax=1.0, delex=4.0, betax=2.0)
    aero.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("cla") == 0.0
    assert store.get("cmde") == 0.0
    assert store.get("cma") != 0.0
    assert store.get("clnb") != 0.0
    czp = aero.deck.look_up("cz_vs_alpha", 1.0 + 1.5)
    czn = aero.deck.look_up("cz_vs_alpha", 1.0 - 1.5)
    assert (czp - czn) / 3 != 0.0


def test_cma_and_clnb_include_cg_shift():
    deck, vehicle, aero = _ready(alphax=1.0, betax=2.0, delex=0.0, xcg=0.7, xcgr=1.1)
    store = vehicle.store
    want = _expected_der(deck, store)
    aero.execute(vehicle, _ctx())
    cza = want["cza"]
    cyb = want["cyb"]
    refc = store.get("refc")
    refb = store.get("refb")
    shift_m = cza * (1.1 - 0.7) / refc
    shift_n = -cyb * (1.1 - 0.7) / refb
    assert shift_m != 0.0
    assert shift_n != 0.0
    assert _approx(store.get("cma"), want["cma"])
    assert _approx(store.get("clnb"), want["clnb"])
    cmp = deck.look_up("cm_vs_elev_alpha", 0.0, 1.0 + 1.5)
    cmn = deck.look_up("cm_vs_elev_alpha", 0.0, 1.0 - 1.5)
    dum_m = (cmp - cmn) / 3
    assert not _approx(store.get("cma"), dum_m)
    cnp = deck.look_up("cn_vs_beta_alpha", 2.0 + 1.5, 1.0)
    cnn = deck.look_up("cn_vs_beta_alpha", 2.0 - 1.5, 1.0)
    dum_n = (cnp - cnn) / 3
    assert not _approx(store.get("clnb"), dum_n)


def test_stmarg_uses_local_cla_not_store():
    deck, vehicle, aero = _ready(alphax=1.0, delex=0.0)
    store = vehicle.store
    want = _expected_der(deck, store)
    aero.execute(vehicle, _ctx())
    assert store.get("cla") == 0.0
    assert want["cla_local"] != 0.0
    assert _approx(store.get("stmarg"), -store.get("cma") / want["cla_local"])
    assert _approx(store.get("stmarg"), want["stmarg"])


def test_aero_unassigned_locals_stay_zero_with_der():
    _, vehicle, aero = _ready(
        alphax=1.0, betax=2.0, delex=4.0, delax=5.0, delrx=3.0, ppx=10.0, qqx=5.0, rrx=8.0
    )
    aero.execute(vehicle, _ctx())
    store = vehicle.store
    for name in AERO_UNASSIGNED:
        assert store.get(name) == 0.0, name
    for name in ("dla", "dma", "cma", "stmarg", "dllda"):
        assert store.get(name) != 0.0, name
        assert np.isfinite(store.get(name)), name
