import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat3.aim5.control import Aim5Control

RTOL = 1e-12
ATOL = 1e-14
# input_hori Missile
TA = 2.0
TR = 0.1
GACP = 40.0
ALPMAX = 35.0
ALPHAX = 0.0
BETAX = 0.0
AREA = 0.01767
MASS = 63.8
DVBE = 269.0
ALT = 10000.0
CNALP = 0.123 * DEG
CYBET = -0.123 * DEG
CNAIM = 0.2
CYAIM = 0.0
THRUST = 28075.0
ANCOMX = 1.0
ALCOMX = 0.0
INT_STEP = 0.002

CONTROL_FIELDS = {
    "ta": ("real", "data", ()),
    "tr": ("real", "data", ()),
    "gacp": ("real", "data", ()),
    "tip": ("real", "diag", ("scrn", "plot")),
    "xi": ("real", "state", ()),
    "xid": ("real", "state", ()),
    "ratep": ("real", "state", ()),
    "ratepd": ("real", "state", ()),
    "alp": ("real", "state", ()),
    "alpd": ("real", "state", ()),
    "yi": ("real", "state", ()),
    "yid": ("real", "state", ()),
    "ratey": ("real", "state", ()),
    "rateyd": ("real", "state", ()),
    "bet": ("real", "state", ()),
    "betd": ("real", "state", ()),
    "alphax": ("real", "in/out", ("scrn", "plot")),
    "betax": ("real", "in/out", ("scrn", "plot")),
}

EXTERNALS = (
    "dvbe",
    "mass",
    "area",
    "pdynmc",
    "cnalp",
    "cybet",
    "cnaim",
    "cyaim",
    "thrust",
    "ancomx",
    "alcomx",
    "grav",
    "alpmax",
)


def _cadac_sign(variable):
    if variable < 0:
        return -1
    return 1


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=INT_STEP,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _pdynmc():
    rho, _, _ = atmosphere76(ALT)
    return 0.5 * rho * DVBE**2


def _expected_control(
    *,
    dvbe,
    mass,
    area,
    pdynmc,
    cnalp,
    cybet,
    cnaim,
    cyaim,
    thrust,
    ancomx,
    alcomx,
    grav,
    ta,
    tr,
    gacp,
    alpmax,
    xi,
    xid,
    ratep,
    ratepd,
    alp,
    alpd,
    yi,
    yid,
    ratey,
    rateyd,
    bet,
    betd,
    int_step,
):
    # Pitch acceleration controller (Aim::control)
    tip = dvbe * mass / (pdynmc * area * abs(cnalp) + thrust)
    fspz = -pdynmc * area * cnaim / mass
    gr = gacp * tip * tr / dvbe
    gi = gr / ta
    abez = -ancomx * grav
    ep = abez - fspz
    xid_new = gi * ep
    xi = integrate(xid_new, xid, xi, int_step)
    xid = xid_new
    ratepc = -(ep * gr + xi)
    ratepd_new = (ratepc - ratep) / tr
    ratep = integrate(ratepd_new, ratepd, ratep, int_step)
    ratepd = ratepd_new
    alpd_new = (tip * ratep - alp) / tip
    alp = integrate(alpd_new, alpd, alp, int_step)
    alpd = alpd_new
    alphax = alp * DEG
    if abs(alphax) > alpmax:
        alphax = alpmax * _cadac_sign(alphax)

    # Yaw: C++ recomputes gr/gi with tiy
    tiy = dvbe * mass / (pdynmc * area * abs(cybet) + thrust)
    fspy = pdynmc * area * cyaim / mass
    gr = gacp * tiy * tr / dvbe
    gi = gr / ta
    abey = alcomx * grav
    ey = abey - fspy
    yid_new = gi * ey
    yi = integrate(yid_new, yid, yi, int_step)
    yid = yid_new
    rateyc = ey * gr + yi
    rateyd_new = (rateyc - ratey) / tr
    ratey = integrate(rateyd_new, rateyd, ratey, int_step)
    rateyd = rateyd_new
    betd_new = -(tiy * ratey + bet) / tiy
    bet = integrate(betd_new, betd, bet, int_step)
    betd = betd_new
    betax = bet * DEG
    if abs(betax) > alpmax:
        betax = alpmax * _cadac_sign(betax)

    return {
        "tip": tip,
        "xi": xi,
        "xid": xid,
        "ratep": ratep,
        "ratepd": ratepd,
        "alp": alp,
        "alpd": alpd,
        "alphax": alphax,
        "yi": yi,
        "yid": yid,
        "ratey": ratey,
        "rateyd": rateyd,
        "bet": bet,
        "betd": betd,
        "betax": betax,
    }


def _snapshot(store):
    return {
        "xi": store.get("xi"),
        "xid": store.get("xid"),
        "ratep": store.get("ratep"),
        "ratepd": store.get("ratepd"),
        "alp": store.get("alp"),
        "alpd": store.get("alpd"),
        "yi": store.get("yi"),
        "yid": store.get("yid"),
        "ratey": store.get("ratey"),
        "rateyd": store.get("rateyd"),
        "bet": store.get("bet"),
        "betd": store.get("betd"),
    }


def _ready(*, alphax=ALPHAX, betax=BETAX, ancomx=ANCOMX, alcomx=ALCOMX):
    pdynmc = _pdynmc()
    grav = gravity(ALT)
    vehicle = _Vehicle()
    control = Aim5Control()
    control.define(vehicle)
    store = vehicle.store
    for name, value, module in (
        ("dvbe", DVBE, "newton"),
        ("mass", MASS, "propulsion"),
        ("area", AREA, "aerodynamics"),
        ("pdynmc", pdynmc, "environment"),
        ("cnalp", CNALP, "aerodynamics"),
        ("cybet", CYBET, "aerodynamics"),
        ("cnaim", CNAIM, "aerodynamics"),
        ("cyaim", CYAIM, "aerodynamics"),
        ("thrust", THRUST, "propulsion"),
        ("ancomx", ancomx, "guidance"),
        ("alcomx", alcomx, "guidance"),
        ("grav", grav, "environment"),
        ("alpmax", ALPMAX, "aerodynamics"),
    ):
        store.define(Field(name, value, "real", "out", module))
    store.set("ta", TA)
    store.set("tr", TR)
    store.set("gacp", GACP)
    store.set("alphax", alphax)
    store.set("betax", betax)
    control.initialize(vehicle, _ctx())
    return vehicle, control, pdynmc, grav


def test_name_is_control():
    assert Aim5Control().name == "control"


def test_define_registers_cpp_control_fields():
    vehicle = _Vehicle()
    Aim5Control().define(vehicle)
    store = vehicle.store
    plot = ("scrn", "plot")
    for name, (ftype, role, outputs) in CONTROL_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "control"
        assert field.outputs == outputs
        assert field.value == pytest.approx(0.0, abs=ATOL)
    assert store.field("alphax").outputs == plot
    assert store.field("betax").outputs == plot
    assert store.field("betax").outputs != ("scrn.plot",)
    for name in EXTERNALS:
        assert name not in store.names()
    assert "dvae" not in store.names()


def test_initialize_hori_alp_bet_zero():
    vehicle = _Vehicle()
    control = Aim5Control()
    control.define(vehicle)
    store = vehicle.store
    store.set("alphax", ALPHAX)
    store.set("betax", BETAX)
    control.initialize(vehicle, _ctx())
    assert store.get("alp") == 0.0
    assert store.get("bet") == 0.0


def test_one_step_matches_cpp_pitch_yaw():
    vehicle, control, pdynmc, grav = _ready()
    want = _expected_control(
        dvbe=DVBE,
        mass=MASS,
        area=AREA,
        pdynmc=pdynmc,
        cnalp=CNALP,
        cybet=CYBET,
        cnaim=CNAIM,
        cyaim=CYAIM,
        thrust=THRUST,
        ancomx=ANCOMX,
        alcomx=ALCOMX,
        grav=grav,
        ta=TA,
        tr=TR,
        gacp=GACP,
        alpmax=ALPMAX,
        int_step=INT_STEP,
        **_snapshot(vehicle.store),
    )

    control.execute(vehicle, _ctx())

    store = vehicle.store
    for name in ("alphax", "betax", "xi", "ratep", "alp"):
        np.testing.assert_allclose(store.get(name), want[name], rtol=RTOL, atol=ATOL)


def test_two_step_uses_stored_slope():
    vehicle, control, pdynmc, grav = _ready()
    control.execute(vehicle, _ctx())
    want = _expected_control(
        dvbe=DVBE,
        mass=MASS,
        area=AREA,
        pdynmc=pdynmc,
        cnalp=CNALP,
        cybet=CYBET,
        cnaim=CNAIM,
        cyaim=CYAIM,
        thrust=THRUST,
        ancomx=ANCOMX,
        alcomx=ALCOMX,
        grav=grav,
        ta=TA,
        tr=TR,
        gacp=GACP,
        alpmax=ALPMAX,
        int_step=INT_STEP,
        **_snapshot(vehicle.store),
    )

    control.execute(vehicle, _ctx())

    store = vehicle.store
    for name in ("alphax", "betax", "xi", "ratep", "alp"):
        np.testing.assert_allclose(store.get(name), want[name], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("xid"), want["xid"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ratepd"), want["ratepd"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("alpd"), want["alpd"], rtol=RTOL, atol=ATOL)


def test_alphax_clip_does_not_write_alp():
    vehicle, control, pdynmc, grav = _ready(ancomx=100.0)
    store = vehicle.store
    store.set("alp", ALPMAX * RAD)
    store.set("ratep", 50.0)
    want = _expected_control(
        dvbe=DVBE,
        mass=MASS,
        area=AREA,
        pdynmc=pdynmc,
        cnalp=CNALP,
        cybet=CYBET,
        cnaim=CNAIM,
        cyaim=CYAIM,
        thrust=THRUST,
        ancomx=100.0,
        alcomx=ALCOMX,
        grav=grav,
        ta=TA,
        tr=TR,
        gacp=GACP,
        alpmax=ALPMAX,
        int_step=INT_STEP,
        **_snapshot(vehicle.store),
    )
    assert abs(want["alp"] * DEG) > ALPMAX

    control.execute(vehicle, _ctx())

    assert abs(store.get("alphax")) <= ALPMAX
    np.testing.assert_allclose(store.get("alphax"), want["alphax"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("alp"), want["alp"], rtol=RTOL, atol=ATOL)
    assert abs(store.get("alp") * DEG) > ALPMAX


def test_terminate_is_pass():
    vehicle, control, _, _ = _ready()
    control.terminate(vehicle, _ctx())
    assert vehicle.store.get("alp") == 0.0
    assert vehicle.store.get("bet") == 0.0
