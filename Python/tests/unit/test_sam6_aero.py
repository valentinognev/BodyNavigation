from math import fabs, sqrt
from pathlib import Path

import numpy as np
import pytest

from cadac.constants import AGRAV, DEG, RAD
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.flat6.sam6.aero import Sam6Aero

SAM6 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/SAM6_250217/SAM6"
AERO = SAM6 / "SAM_aero_deck.asc"

RTOL = 1e-12
ATOL = 1e-14
SMALL = 1e-7

VMACH = 2.0
ALPHAX = 10.0
BETAX = 0.0
MASS = 300.0
XCGREF = 2.5
XCG = 2.5
MPROP = 1
PDYNMC = 50000.0
PHIP = 0.0
PPX = 0.0
QQX = 0.0
RRX = 0.0
DVBE = 680.0
ALIMITX = 50.0
DPX = 0.0
DQX = 0.0
DRX = 0.0
DELX1 = 0.0
DELX2 = 0.0
DELX3 = 0.0
DELX4 = 0.0
ALPPX = 10.0
AI11 = 2.9
AI33 = 440.0
THRUST = 0.0

DEFINED = (
    "refl",
    "refa",
    "ca",
    "cy",
    "cn",
    "cll",
    "clm",
    "cln",
    "cyb",
    "clnb",
    "clnr",
    "clndr",
    "ca0",
    "cad",
    "cndq",
    "clmdq",
    "clmq",
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
    "clma",
    "dlnd",
    "stmarg_pitch",
    "realq1",
    "realq2",
    "wnq",
    "zetq",
    "realp",
    "alplimx",
    "gavail",
    "gmax",
    "pqreal",
    "dnr",
    "dyb",
    "dnb",
    "wnr",
    "zetr",
    "stmarg_yaw",
    "realr1",
    "realr2",
    "prreal",
    "trcond",
    "trortho",
    "tralp",
    "trdynm",
    "trload",
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
    "cyb": "save",
    "clnb": "save",
    "clnr": "diag",
    "clndr": "diag",
    "ca0": "diag",
    "cad": "diag",
    "cndq": "diag",
    "clmdq": "diag",
    "clmq": "diag",
    "clldp": "diag",
    "cllp": "diag",
    "dna": "out",
    "dnd": "out",
    "dma": "out",
    "dmq": "out",
    "dmd": "out",
    "dlp": "out",
    "dld": "out",
    "cna": "save",
    "clma": "save",
    "dlnd": "out",
    "stmarg_pitch": "save",
    "realq1": "diag",
    "realq2": "diag",
    "wnq": "diag",
    "zetq": "diag",
    "realp": "diag",
    "alplimx": "data",
    "gavail": "diag",
    "gmax": "diag",
    "pqreal": "diag",
    "dnr": "out",
    "dyb": "out",
    "dnb": "out",
    "wnr": "diag",
    "zetr": "diag",
    "stmarg_yaw": "save",
    "realr1": "diag",
    "realr2": "diag",
    "prreal": "diag",
    "trcond": "diag",
    "trortho": "data",
    "tralp": "data",
    "trdynm": "data",
    "trload": "data",
}

PLOT_FLAGGED = ("cyb", "clnb", "cna", "clma", "stmarg_pitch", "gavail", "gmax", "stmarg_yaw")
SCRN_FLAGGED = ("realq1", "realq2")

NOT_DEFINED = ("vmach", "alphax", "betax", "dpx")

EXTERNALS = (
    "vmach",
    "pdynmc",
    "phip",
    "ppx",
    "qqx",
    "rrx",
    "dvbe",
    "mprop",
    "mass",
    "xcgref",
    "xcg",
    "alimitx",
    "dpx",
    "dqx",
    "drx",
    "delx1",
    "delx2",
    "delx3",
    "delx4",
    "alphax",
    "betax",
    "alppx",
    "ai11",
    "ai33",
    "thrust",
)

FORMULA_NAMES = (
    "ca",
    "cy",
    "cn",
    "cll",
    "clm",
    "cln",
    "ca0",
    "cad",
    "cndq",
    "clmdq",
    "clmq",
    "clldp",
    "cllp",
    "clnr",
    "clndr",
    "gmax",
    "gavail",
    "dna",
    "dnd",
    "dma",
    "dmq",
    "dmd",
    "dlp",
    "dld",
    "dlnd",
    "dnr",
    "dyb",
    "dnb",
    "cna",
    "clma",
    "cyb",
    "clnb",
    "stmarg_pitch",
    "stmarg_yaw",
    "realq1",
    "realq2",
    "wnq",
    "zetq",
    "realp",
    "pqreal",
    "wnr",
    "zetr",
    "realr1",
    "realr2",
    "prreal",
)


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


def _externals(
    store,
    *,
    vmach=VMACH,
    pdynmc=PDYNMC,
    phip=PHIP,
    ppx=PPX,
    qqx=QQX,
    rrx=RRX,
    dvbe=DVBE,
    mprop=MPROP,
    mass=MASS,
    xcgref=XCGREF,
    xcg=XCG,
    alimitx=ALIMITX,
    dpx=DPX,
    dqx=DQX,
    drx=DRX,
    delx1=DELX1,
    delx2=DELX2,
    delx3=DELX3,
    delx4=DELX4,
    alphax=ALPHAX,
    betax=BETAX,
    alppx=ALPPX,
    ai11=AI11,
    ai33=AI33,
    thrust=THRUST,
):
    store.define(Field("vmach", vmach, "real", "out", "environment"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("phip", phip, "real", "out", "kinematics"))
    store.define(Field("ppx", ppx, "real", "out", "euler"))
    store.define(Field("qqx", qqx, "real", "out", "euler"))
    store.define(Field("rrx", rrx, "real", "out", "euler"))
    store.define(Field("dvbe", dvbe, "real", "out", "newton"))
    store.define(Field("mprop", mprop, "int", "data", "propulsion"))
    store.define(Field("mass", mass, "real", "out", "propulsion"))
    store.define(Field("xcgref", xcgref, "real", "data", "propulsion"))
    store.define(Field("xcg", xcg, "real", "out", "propulsion"))
    store.define(Field("alimitx", alimitx, "real", "data", "control"))
    store.define(Field("dpx", dpx, "real", "out", "actuator"))
    store.define(Field("dqx", dqx, "real", "out", "actuator"))
    store.define(Field("drx", drx, "real", "out", "actuator"))
    store.define(Field("delx1", delx1, "real", "out", "actuator"))
    store.define(Field("delx2", delx2, "real", "out", "actuator"))
    store.define(Field("delx3", delx3, "real", "out", "actuator"))
    store.define(Field("delx4", delx4, "real", "out", "actuator"))
    store.define(Field("alphax", alphax, "real", "out", "kinematics"))
    store.define(Field("betax", betax, "real", "out", "kinematics"))
    store.define(Field("alppx", alppx, "real", "out", "kinematics"))
    store.define(Field("ai11", ai11, "real", "out", "propulsion"))
    store.define(Field("ai33", ai33, "real", "out", "propulsion"))
    store.define(Field("thrust", thrust, "real", "out", "propulsion"))


def _ready(**kw):
    deck = _deck()
    vehicle = _Vehicle()
    aero = Sam6Aero(deck)
    aero.define(vehicle)
    aero.initialize(vehicle, _ctx())
    _externals(vehicle.store, **kw)
    return deck, vehicle, aero


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _expected(deck, store):
    look_up = deck.look_up
    mach = store.get("vmach")
    pdynmc = store.get("pdynmc")
    ppx = store.get("ppx")
    qqx = store.get("qqx")
    rrx = store.get("rrx")
    dvbe = store.get("dvbe")
    mprop = store.get("mprop")
    mass = store.get("mass")
    xcgref = store.get("xcgref")
    xcg = store.get("xcg")
    alimitx = store.get("alimitx")
    dpx = store.get("dpx")
    dqx = store.get("dqx")
    drx = store.get("drx")
    delx1 = store.get("delx1")
    delx2 = store.get("delx2")
    delx3 = store.get("delx3")
    delx4 = store.get("delx4")
    alphax = store.get("alphax")
    betax = store.get("betax")
    alppx = store.get("alppx")
    ai11 = store.get("ai11")
    ai33 = store.get("ai33")
    refl = store.get("refl")
    refa = store.get("refa")
    alplimx = store.get("alplimx")
    trcond = store.get("trcond")
    trload = store.get("trload")
    cyb = store.get("cyb")
    clnb = store.get("clnb")
    cna = store.get("cna")
    clma = store.get("clma")
    stmarg_pitch = store.get("stmarg_pitch")
    stmarg_yaw = store.get("stmarg_yaw")

    deffx = (fabs(delx1) + fabs(delx2) + fabs(delx3) + fabs(delx4)) / 4
    ca0 = look_up("ca0_vs_mach,betax,alphax", mach, betax, alphax)
    cad = look_up("cad_vs_mach", mach)
    ca = ca0 + cad * deffx
    if mprop == 0:
        ca += look_up("cab_vs_mach", mach)

    cy0 = look_up("cy0_vs_mach,betax,alphax", mach, betax, alphax)
    cydr = look_up("cydr_vs_mach,betax,alphax", mach, betax, alphax)
    cy = cy0 + cydr * drx

    cn0 = look_up("cn0_vs_mach,betax,alphax", mach, betax, alphax)
    cndq = look_up("cndq_vs_mach,betax,alphax", mach, betax, alphax)
    cn = cn0 + cndq * dqx

    cll0 = look_up("cll0_vs_mach,betax,alphax", mach, betax, alphax)
    cllp = look_up("cllp_vs_mach", mach)
    clldp = look_up("clldp_vs_mach,betax,alphax", mach, betax, alphax)
    cll = cll0 + cllp * ppx * RAD * refl / (2 * dvbe) + clldp * dpx

    clm0 = look_up("clm0_vs_mach,betax,alphax", mach, betax, alphax)
    clmq = look_up("clmq_vs_mach", mach)
    clmdq = look_up("clmdq_vs_mach,betax,alphax", mach, betax, alphax)
    clm = clm0 + clmq * qqx * RAD * refl / (2 * dvbe) + clmdq * dqx - cn / refl * (xcgref - xcg)

    cln0 = look_up("cln0_vs_mach,betax,alphax", mach, betax, alphax)
    clnr = look_up("clnr_vs_mach", mach)
    clndr = look_up("clndr_vs_mach,betax,alphax", mach, betax, alphax)
    cln = cln0 + clnr * rrx * RAD * refl / (2 * dvbe) + clndr * drx - cy / refl * (xcgref - xcg)

    cn0mx = look_up("cn0_vs_mach,betax,alphax", mach, 0, alplimx)
    almx = cn0mx * pdynmc * refa
    weight = mass * AGRAV
    gmax = almx / weight
    if gmax >= alimitx:
        gmax = alimitx
    aload = sqrt(cn0 * cn0 + cy0 * cy0) * pdynmc * refa
    gg = aload / weight
    gavail = gmax - gg
    if gavail > alimitx:
        gavail = alimitx
    if gavail < 0:
        gavail = 0
    if gmax < trload:
        trcond = 4

    if alppx < (alplimx - 3):
        alphapx = fabs(alphax) + 3
        if alphapx < 3:
            alphapx = 3
        alphamx = fabs(alphax) - 3
        if alphamx < 0:
            alphamx = 0
        cn0p = look_up("cn0_vs_mach,betax,alphax", mach, betax, alphapx)
        cn0m = look_up("cn0_vs_mach,betax,alphax", mach, betax, alphamx)
        cna = (cn0p - cn0m) / ((alphapx - alphamx) * RAD)
        clm0p = look_up("clm0_vs_mach,betax,alphax", mach, betax, alphapx)
        clm0m = look_up("clm0_vs_mach,betax,alphax", mach, betax, alphamx)
        clma = (clm0p - clm0m) / ((alphapx - alphamx) * RAD) - cna / refl * (xcgref - xcg)

        betapx = fabs(betax) + 3
        if betapx < 3:
            betapx = 3
        betamx = fabs(betax) - 3
        if betamx < 0:
            betamx = 0
        cy0p = look_up("cy0_vs_mach,betax,alphax", mach, betapx, alphax)
        cy0m = look_up("cy0_vs_mach,betax,alphax", mach, betamx, alphax)
        cyb = (cy0p - cy0m) / ((betapx - betamx) * RAD)
        cln0p = look_up("cln0_vs_mach,betax,alphax", mach, betapx, alphax)
        cln0m = look_up("cln0_vs_mach,betax,alphax", mach, betamx, alphax)
        clnb = (cln0p - cln0m) / ((betapx - betamx) * RAD) - cyb / refl * (xcgref - xcg)

    dna = (pdynmc * refa / mass) * cna
    dma = (pdynmc * refa * refl / ai33) * clma
    dmq = DEG * (pdynmc * refa * refl / ai33) * (refl / (2 * dvbe)) * clmq
    dmd = DEG * (pdynmc * refa * refl / ai33) * clmdq
    dyb = (pdynmc * refa / mass) * cyb
    dnb = (pdynmc * refa * refl / ai33) * clnb
    dnr = DEG * (pdynmc * refa * refl / ai33) * (refl / (2 * dvbe)) * clnr
    dnd = (pdynmc * refa / mass) * cndq
    dlnd = DEG * (pdynmc * refa * refl / ai33) * clndr
    dlp = DEG * (pdynmc * refa * refl / ai11) * (refl / (2 * dvbe)) * cllp
    dld = DEG * (pdynmc * refa * refl / ai11) * clldp

    wnq = 0.0
    zetq = 0.0
    realq1 = 0.0
    realq2 = 0.0
    pqreal = 0.0
    if fabs(dna) >= SMALL:
        stmarg_pitch = -(dma / dna) * (ai33 / (refl * mass))
        a11 = dmq
        a12 = dma / dna
        a21 = dna
        a22 = -dna / dvbe
        arg = (a11 + a22) ** 2 - 4 * (a11 * a22 - a12 * a21)
        if arg >= 0:
            wnq = 0
            zetq = 0
            dum = a11 + a22
            realq1 = (dum + sqrt(arg)) / 2
            realq2 = (dum - sqrt(arg)) / 2
            pqreal = (realq1 + realq2) / 2
        else:
            realq1 = 0
            realq2 = 0
            wnq = sqrt(a11 * a22 - a12 * a21)
            zetq = -(a11 + a22) / (2 * wnq)
            pqreal = -zetq * wnq

    wnr = 0.0
    zetr = 0.0
    realr1 = 0.0
    realr2 = 0.0
    prreal = 0.0
    if fabs(dyb) >= SMALL:
        stmarg_yaw = -(dnb / dyb) * (ai33 / (refl * mass))
        a11 = dnr
        a12 = dnb / dyb
        a21 = -dyb
        a22 = dyb / dvbe
        arg = (a11 + a22) ** 2 - 4 * (a11 * a22 - a12 * a21)
        if arg >= 0:
            wnr = 0
            zetr = 0
            dum = a11 + a22
            realr1 = (dum + sqrt(arg)) / 2
            realr2 = (dum - sqrt(arg)) / 2
            prreal = (realr1 + realr2) / 2
        else:
            realr1 = 0
            realr2 = 0
            wnr = sqrt(a11 * a22 - a12 * a21)
            zetr = -(a11 + a22) / (2 * wnr)
            prreal = -zetr * wnr

    realp = dlp
    return {
        "ca": ca,
        "cy": cy,
        "cn": cn,
        "cll": cll,
        "clm": clm,
        "cln": cln,
        "ca0": ca0,
        "cad": cad,
        "cndq": cndq,
        "clmdq": clmdq,
        "clmq": clmq,
        "clldp": clldp,
        "cllp": cllp,
        "clnr": clnr,
        "clndr": clndr,
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
        "dlnd": dlnd,
        "dnr": dnr,
        "dyb": dyb,
        "dnb": dnb,
        "cna": cna,
        "clma": clma,
        "cyb": cyb,
        "clnb": clnb,
        "stmarg_pitch": stmarg_pitch,
        "stmarg_yaw": stmarg_yaw,
        "realq1": realq1,
        "realq2": realq2,
        "wnq": wnq,
        "zetq": zetq,
        "realp": realp,
        "pqreal": pqreal,
        "wnr": wnr,
        "zetr": zetr,
        "realr1": realr1,
        "realr2": realr2,
        "prreal": prreal,
    }


def test_name_is_aerodynamics():
    assert Sam6Aero(_deck()).name == "aerodynamics"


def test_define_cpp_fields():
    vehicle = _Vehicle()
    Sam6Aero(_deck()).define(vehicle)
    store = vehicle.store
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        assert field.module == "aerodynamics"
        assert field.role == ROLES[name]
        if name in PLOT_FLAGGED:
            assert "plot" in field.outputs
        if name in SCRN_FLAGGED:
            assert "scrn" in field.outputs
    assert store.field("trcond").type == "int"
    assert store.get("refl") == pytest.approx(0.25, rel=RTOL, abs=ATOL)
    assert store.get("refa") == pytest.approx(0.0491, rel=RTOL, abs=ATOL)
    assert store.get("alplimx") == pytest.approx(40.0, rel=RTOL, abs=ATOL)


def test_does_not_define_vmach_alphax_betax_dpx():
    vehicle = _Vehicle()
    Sam6Aero(_deck()).define(vehicle)
    for name in NOT_DEFINED + EXTERNALS:
        assert name not in vehicle.store.names()


def test_initialize_termination_numbers():
    vehicle = _Vehicle()
    aero = Sam6Aero(_deck())
    aero.define(vehicle)
    aero.initialize(vehicle, _ctx())
    store = vehicle.store
    assert store.get("trcond") == 0
    assert store.get("trortho") == pytest.approx(1e-4, rel=RTOL, abs=ATOL)
    assert store.get("tralp") == pytest.approx(1.047, rel=RTOL, abs=ATOL)
    assert store.get("trdynm") == pytest.approx(1e4, rel=RTOL, abs=ATOL)
    assert store.get("trload") == pytest.approx(0.001, rel=RTOL, abs=ATOL)
    assert store.get("refl") == pytest.approx(0.25, rel=RTOL, abs=ATOL)
    assert store.get("refa") == pytest.approx(0.0491, rel=RTOL, abs=ATOL)


def test_ca_cn_match_lookup_then_cpp_sums():
    deck, vehicle, aero = _ready()
    aero.execute(vehicle, _ctx())
    store = vehicle.store
    mach = store.get("vmach")
    alphax = store.get("alphax")
    betax = store.get("betax")
    dqx = store.get("dqx")
    deffx = (
        fabs(store.get("delx1"))
        + fabs(store.get("delx2"))
        + fabs(store.get("delx3"))
        + fabs(store.get("delx4"))
    ) / 4
    ca0 = deck.look_up("ca0_vs_mach,betax,alphax", mach, betax, alphax)
    cad = deck.look_up("cad_vs_mach", mach)
    ca = ca0 + cad * deffx
    cn0 = deck.look_up("cn0_vs_mach,betax,alphax", mach, betax, alphax)
    cndq = deck.look_up("cndq_vs_mach,betax,alphax", mach, betax, alphax)
    cn = cn0 + cndq * dqx
    assert _approx(store.get("ca"), ca)
    assert _approx(store.get("cn"), cn)
    assert np.isfinite(store.get("dna"))


def test_dna_finite():
    _, vehicle, aero = _ready()
    aero.execute(vehicle, _ctx())
    assert np.isfinite(vehicle.store.get("dna"))
    assert vehicle.store.get("dna") != 0.0


def test_mprop_zero_adds_cab():
    deck, vehicle, aero = _ready(mprop=0)
    aero.execute(vehicle, _ctx())
    store = vehicle.store
    mach = store.get("vmach")
    alphax = store.get("alphax")
    betax = store.get("betax")
    ca0 = deck.look_up("ca0_vs_mach,betax,alphax", mach, betax, alphax)
    cad = deck.look_up("cad_vs_mach", mach)
    cab = deck.look_up("cab_vs_mach", mach)
    assert _approx(store.get("ca"), ca0 + cab)
    _, burning, aero_b = _ready(mprop=1)
    aero_b.execute(burning, _ctx())
    assert store.get("ca") == pytest.approx(
        burning.store.get("ca") + cab, rel=RTOL, abs=ATOL
    )


def test_execute_matches_cpp_replica():
    deck, vehicle, aero = _ready()
    aero.execute(vehicle, _ctx())
    want = _expected(deck, vehicle.store)
    for name in FORMULA_NAMES:
        assert _approx(vehicle.store.get(name), want[name]), name


def test_skip_tvc_gains_when_gtvc_parm_absent():
    _, vehicle, aero = _ready()
    assert "gtvc" not in vehicle.store.names()
    assert "parm" not in vehicle.store.names()
    aero.execute(vehicle, _ctx())
    assert np.isfinite(vehicle.store.get("dna"))
    assert np.isfinite(vehicle.store.get("dma"))


def test_comma_table_names():
    deck = _deck()
    assert "ca0_vs_mach,betax,alphax" in {t.name for t in (
        parse_asc_deck(AERO)[1]
    )}
    ca0 = deck.look_up("ca0_vs_mach,betax,alphax", VMACH, BETAX, ALPHAX)
    cn0 = deck.look_up("cn0_vs_mach,betax,alphax", VMACH, BETAX, ALPHAX)
    assert np.isfinite(ca0)
    assert np.isfinite(cn0)


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.flat6.sam6.aero as mod
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src


def test_terminate_exists_and_is_pass():
    _, vehicle, aero = _ready()
    aero.terminate(vehicle, _ctx())
    assert aero.terminate(vehicle, _ctx()) is None
