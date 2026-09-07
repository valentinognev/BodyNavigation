from math import fabs, isfinite, pow
from pathlib import Path

import pytest

from cadac.constants import DEG
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.sam6.rocket import Sam6RocketControl

RTOL = 1e-12
ATOL = 1e-14
DT = 0.001
ALT = 1000.0
ALT_ENDO = 30000.0
AREA = 0.636
ALPMAX = 20.0
CYTGT = 0.0
CNTGT = 0.0
CNALP = 7.468
CYBET = -7.468
THRUST = 128600.0
MASS = 6000.0
GRAV = 9.81
PDYNMC = 50000.0
DVAE = 300.0
ANCOMX_BIAS = 2.0
ANCOMX = 0.0
ALCOMX = 0.0

DEFINED = (
    "maut",
    "flag_exo",
    "ancomx_bias",
    "alt_endo",
    "tip",
    "xi",
    "xid",
    "ratep",
    "ratepd",
    "alp",
    "alpd",
    "yi",
    "yid",
    "ratey",
    "rateyd",
    "bet",
    "betd",
    "alphax",
    "betax",
)
ROLES = {
    "maut": "data",
    "flag_exo": "data",
    "ancomx_bias": "data",
    "alt_endo": "data",
    "tip": "diag",
    "xi": "state",
    "xid": "state",
    "ratep": "state",
    "ratepd": "state",
    "alp": "state",
    "alpd": "state",
    "yi": "state",
    "yid": "state",
    "ratey": "state",
    "rateyd": "state",
    "bet": "state",
    "betd": "state",
    "alphax": "out",
    "betax": "out",
}
OUTPUTS = {name: () for name in DEFINED}
OUTPUTS["alphax"] = ("com",)
OUTPUTS["betax"] = ("com",)
INT_FIELDS = ("maut", "flag_exo")
NOT_DEFINED = (
    "grav",
    "pdynmc",
    "dvae",
    "alt",
    "area",
    "alpmax",
    "cytgt",
    "cntgt",
    "cnalp",
    "cybet",
    "thrust",
    "mass",
    "mguide",
    "ancomx",
    "alcomx",
    "SAEL",
    "VAEL",
    "TAL",
    "time",
    "FSPA",
    "gmax",
    "dvta",
    "UTAA",
    "WOEA",
)
STATES = (
    "xi",
    "xid",
    "ratep",
    "ratepd",
    "alp",
    "alpd",
    "yi",
    "yid",
    "ratey",
    "rateyd",
    "bet",
    "betd",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ctx(int_step=DT):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _plant(
    store,
    *,
    grav=GRAV,
    pdynmc=PDYNMC,
    dvae=DVAE,
    alt=ALT,
    area=AREA,
    alpmax=ALPMAX,
    cytgt=CYTGT,
    cntgt=CNTGT,
    cnalp=CNALP,
    cybet=CYBET,
    thrust=THRUST,
    mass=MASS,
    ancomx=ANCOMX,
    alcomx=ALCOMX,
):
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("dvae", dvae, "real", "out", "newton"))
    store.define(Field("alt", alt, "real", "out", "newton"))
    store.define(Field("area", area, "real", "data", "aerodynamics"))
    store.define(Field("alpmax", alpmax, "real", "data", "aerodynamics"))
    store.define(Field("cytgt", cytgt, "real", "out", "aerodynamics"))
    store.define(Field("cntgt", cntgt, "real", "out", "aerodynamics"))
    store.define(Field("cnalp", cnalp, "real", "out", "aerodynamics"))
    store.define(Field("cybet", cybet, "real", "out", "aerodynamics"))
    store.define(Field("thrust", thrust, "real", "out", "propulsion"))
    store.define(Field("mass", mass, "real", "out", "propulsion"))
    store.define(Field("ancomx", ancomx, "real", "out", "guidance"))
    store.define(Field("alcomx", alcomx, "real", "out", "guidance"))


def _defined():
    vehicle = _Vehicle()
    control = Sam6RocketControl()
    control.define(vehicle)
    return vehicle, control


def _ready(
    *,
    maut=1,
    flag_exo=0,
    ancomx_bias=ANCOMX_BIAS,
    alt_endo=ALT_ENDO,
    alt=ALT,
    ancomx=ANCOMX,
    alcomx=ALCOMX,
    alpmax=ALPMAX,
    plant=True,
    **plant_kw,
):
    vehicle, control = _defined()
    store = vehicle.store
    store.set("maut", maut)
    store.set("flag_exo", flag_exo)
    store.set("ancomx_bias", ancomx_bias)
    store.set("alt_endo", alt_endo)
    if plant:
        _plant(store, alt=alt, ancomx=ancomx, alcomx=alcomx, alpmax=alpmax, **plant_kw)
    return vehicle, control


def _cpp_step(store, int_step):
    maut = store.get("maut")
    flag_exo = store.get("flag_exo")
    ancomx_bias = store.get("ancomx_bias")
    alt_endo = store.get("alt_endo")
    grav = store.get("grav")
    pdynmc = store.get("pdynmc")
    dvae = store.get("dvae")
    alt = store.get("alt")
    area = store.get("area")
    alpmax = store.get("alpmax")
    cytgt = store.get("cytgt")
    cntgt = store.get("cntgt")
    cnalp = store.get("cnalp")
    cybet = store.get("cybet")
    thrust = store.get("thrust")
    mass = store.get("mass")
    ancomx = store.get("ancomx")
    alcomx = store.get("alcomx")
    xi = store.get("xi")
    xid = store.get("xid")
    ratep = store.get("ratep")
    ratepd = store.get("ratepd")
    alp = store.get("alp")
    alpd = store.get("alpd")
    yi = store.get("yi")
    yid = store.get("yid")
    ratey = store.get("ratey")
    rateyd = store.get("rateyd")
    bet = store.get("bet")
    betd = store.get("betd")
    tip = 0.0
    alphax = 0.0
    betax = 0.0
    if alt > alt_endo:
        flag_exo = 1
        xi = 0.0
        xid = 0.0
        ratep = 0.0
        ratepd = 0.0
        alp = 0.0
        alpd = 0.0
        yi = 0.0
        yid = 0.0
        ratey = 0.0
        rateyd = 0.0
        bet = 0.0
        betd = 0.0
    if not flag_exo:
        ancomx = ancomx_bias
    if maut == 1 and alt < alt_endo:
        tr = ((-2e-7) * pdynmc + 0.22)
        gacp = pow((0.002 * pdynmc), 0.575) * (1 - 0.5)
        ta = 2.2
        tip = dvae * mass / (pdynmc * area * fabs(cnalp) + thrust)
        fspz = -pdynmc * area * cntgt / mass
        gr = gacp * tip * tr / dvae
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
        if fabs(alphax) > alpmax:
            alphax = alpmax * _sign(alphax)
        tiy = dvae * mass / (pdynmc * area * fabs(cybet) + thrust)
        fspy = pdynmc * area * cytgt / mass
        gr = gacp * tiy * tr / dvae
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
        if fabs(betax) > alpmax:
            betax = alpmax * _sign(betax)
    return {
        "flag_exo": flag_exo,
        "tip": tip,
        "xi": xi,
        "xid": xid,
        "ratep": ratep,
        "ratepd": ratepd,
        "alp": alp,
        "alpd": alpd,
        "yi": yi,
        "yid": yid,
        "ratey": ratey,
        "rateyd": rateyd,
        "bet": bet,
        "betd": betd,
        "alphax": alphax,
        "betax": betax,
        "ancomx": store.get("ancomx"),
        "alcomx": store.get("alcomx"),
    }


def _assert_cpp(store, want):
    for name, value in want.items():
        got = store.get(name)
        if name == "flag_exo":
            assert got == value, name
        else:
            assert _approx(got, value), name


def test_name_is_control():
    assert Sam6RocketControl.name == "control"


def test_define_cpp_fields():
    vehicle, _ = _defined()
    store = vehicle.store
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        assert field.module == "control", name
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        if name in INT_FIELDS:
            assert field.type == "int", name
            assert store.get(name) == 0, name
        else:
            assert field.type == "real", name
            assert _approx(store.get(name), 0.0), name
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_maut_1_alt_1000_alphax_finite():
    vehicle, control = _ready(maut=1, alt=ALT, alt_endo=ALT_ENDO)
    want = _cpp_step(vehicle.store, DT)
    control.execute(vehicle, _ctx(DT))
    alphax = vehicle.store.get("alphax")
    assert isfinite(alphax)
    assert isfinite(want["alphax"])
    assert _approx(alphax, want["alphax"])


def test_maut_2_raises():
    vehicle, control = _ready(maut=2)
    with pytest.raises(ValueError):
        control.execute(vehicle, _ctx())
    assert _approx(vehicle.store.get("alphax"), 0.0)


def test_maut_0_ballistic_no_accel_loop():
    vehicle, control = _ready(maut=0, ancomx_bias=ANCOMX_BIAS)
    store = vehicle.store
    store.set("alp", 0.5)
    store.set("xi", 1.0)
    control.execute(vehicle, _ctx())
    assert _approx(store.get("alphax"), 0.0)
    assert _approx(store.get("betax"), 0.0)
    assert _approx(store.get("alp"), 0.5)
    assert _approx(store.get("xi"), 1.0)
    assert store.get("flag_exo") == 0
    assert _approx(store.get("ancomx"), 0.0)


def test_exo_sets_flag_and_zeros_states():
    vehicle, control = _ready(maut=1, alt=40000.0, alt_endo=ALT_ENDO, flag_exo=0)
    store = vehicle.store
    for name in STATES:
        store.set(name, 0.4)
    store.set("alphax", 7.0)
    control.execute(vehicle, _ctx())
    assert store.get("flag_exo") == 1
    for name in STATES:
        assert _approx(store.get(name), 0.0), name
    assert _approx(store.get("alphax"), 0.0)
    assert _approx(store.get("betax"), 0.0)
    assert _approx(store.get("tip"), 0.0)


def test_ascent_applies_ancomx_bias_not_guidance():
    vehicle, control = _ready(
        maut=1,
        flag_exo=0,
        ancomx_bias=ANCOMX_BIAS,
        ancomx=99.0,
        alcomx=0.5,
    )
    ctx = _ctx()
    want = _cpp_step(vehicle.store, ctx.int_step)
    control.execute(vehicle, ctx)
    store = vehicle.store
    assert _approx(store.get("ancomx"), 99.0)
    assert _approx(store.get("alcomx"), 0.5)
    _assert_cpp(store, {k: v for k, v in want.items() if k not in ("ancomx", "alcomx")})
    unbiased = _ready(
        maut=1,
        flag_exo=0,
        ancomx_bias=0.0,
        ancomx=99.0,
        alcomx=0.5,
    )[0].store
    want_guide = _cpp_step(unbiased, ctx.int_step)
    assert not _approx(want["alphax"], want_guide["alphax"])


def test_reentry_skips_bias_when_flag_exo():
    vehicle, control = _ready(
        maut=1,
        flag_exo=1,
        alt=ALT,
        alt_endo=ALT_ENDO,
        ancomx_bias=ANCOMX_BIAS,
        ancomx=1.0,
        alcomx=0.5,
    )
    ctx = _ctx()
    want = _cpp_step(vehicle.store, ctx.int_step)
    control.execute(vehicle, ctx)
    store = vehicle.store
    assert store.get("flag_exo") == 1
    assert _approx(store.get("alphax"), want["alphax"])
    biased = _ready(
        maut=1,
        flag_exo=0,
        alt=ALT,
        ancomx_bias=ANCOMX_BIAS,
        ancomx=1.0,
        alcomx=0.5,
    )[0].store
    want_ascent = _cpp_step(biased, ctx.int_step)
    assert not _approx(store.get("alphax"), want_ascent["alphax"])


def test_pi_pitch_yaw_match_cpp():
    vehicle, control = _ready(
        maut=1,
        flag_exo=0,
        ancomx_bias=ANCOMX_BIAS,
        ancomx=0.0,
        alcomx=0.3,
        cntgt=0.05,
        cytgt=-0.02,
    )
    ctx = _ctx()
    want = _cpp_step(vehicle.store, ctx.int_step)
    control.execute(vehicle, ctx)
    _assert_cpp(vehicle.store, {k: v for k, v in want.items() if k not in ("ancomx", "alcomx")})
    assert isfinite(vehicle.store.get("alphax"))
    assert isfinite(vehicle.store.get("betax"))


def test_stored_slope_second_step():
    vehicle, control = _ready(maut=1, flag_exo=0, ancomx_bias=ANCOMX_BIAS, alcomx=0.2)
    ctx = _ctx()
    control.execute(vehicle, ctx)
    control.execute(vehicle, ctx)
    first = _ready(maut=1, flag_exo=0, ancomx_bias=ANCOMX_BIAS, alcomx=0.2)
    first[1].execute(first[0], ctx)
    want = _cpp_step(first[0].store, ctx.int_step)
    _assert_cpp(vehicle.store, {k: v for k, v in want.items() if k not in ("ancomx", "alcomx")})


def test_alpha_limiter_cadac_sign():
    vehicle, control = _ready(
        maut=1,
        flag_exo=0,
        ancomx_bias=50.0,
        alpmax=1.0,
    )
    vehicle.store.set("alp", 2.0)
    control.execute(vehicle, _ctx())
    alphax = vehicle.store.get("alphax")
    assert _approx(alphax, 1.0)
    assert _approx(alphax, 1.0 * _sign(alphax))

    vehicle, control = _ready(
        maut=1,
        flag_exo=0,
        ancomx_bias=-50.0,
        alpmax=1.0,
    )
    vehicle.store.set("alp", -2.0)
    control.execute(vehicle, _ctx())
    alphax = vehicle.store.get("alphax")
    assert _approx(alphax, -1.0)
    assert _approx(alphax, 1.0 * _sign(alphax))


def test_initialize_and_terminate_are_pass():
    vehicle, control = _ready(maut=1)
    ctx = _ctx()
    assert control.initialize(vehicle, ctx) is None
    control.execute(vehicle, ctx)
    assert control.terminate(vehicle, ctx) is None
    assert isfinite(vehicle.store.get("alphax"))


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.sam6.rocket as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6" not in src
    assert "cadac.eom.flat3" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "Plane5" not in src
    assert "Plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src
    assert "class Sam6Rocket:" not in src
