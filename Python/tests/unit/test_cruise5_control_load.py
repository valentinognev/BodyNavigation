import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import cadtbv
from cadac.vehicles.cruise5.control import Cruise5Control

GACP = 10.0
TA = 0.8
ANPOSLIMX = 3.0
ANNEGLIMX = -1.0
ALPPOSLIMX = 15.0
ALPNEGLIMX = -10.0
MASS = 1000.0
DVBE = 200.0
AREA = 0.929
INT_STEP = 0.05
FSPV = np.array([2.0, 1.0, -12.0])
GRAV = 9.81
PDYNMC = 5000.0
THRUST = 1500.0
CLA = 0.11
ANCOMX = 1.5
PHIMVX = 0.0
ALPHAX = 0.0
RTOL = 1e-12
ATOL = 1e-14


def _expected_load(ancomx, int_step, phimvx, alphax, anposlimx, anneglimx,
                   gacp, ta, alpposlimx, alpneglimx, fspv, grav, mass, dvbe,
                   pdynmc, thrust, area, cla, xi, xid, alp, alpd):
    tbv = cadtbv(phimvx * RAD, alphax * RAD)
    fspb = tbv @ fspv
    if ancomx > anposlimx:
        ancomx = anposlimx
    if ancomx < anneglimx:
        ancomx = anneglimx
    anx = -fspb[2] / grav
    eanx = ancomx - anx
    tip = dvbe * mass / (pdynmc * area * cla / RAD + thrust)
    gr = 0.0
    if ta > 0:
        gr = gacp * tip / dvbe
        gi = gr / ta
        xid_new = gi * eanx
        xi = integrate(xid_new, xid, xi, int_step)
        xid = xid_new
    else:
        xi = 0.0
    qq = gr * eanx + xi
    alpd_new = qq - alp / tip
    alp = integrate(alpd_new, alpd, alp, int_step)
    alpx = alp * DEG
    if alpx > alpposlimx:
        alpx = alpposlimx
    if alpx < alpneglimx:
        alpx = alpneglimx
    return alpx, xi, alp, tip, qq, anx


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ready(ancomx=ANCOMX, ta=TA, alphax=ALPHAX, alp=0.0):
    vehicle = type("V", (), {"store": StateStore()})()
    control = Cruise5Control()
    control.define(vehicle)
    store = vehicle.store
    store.define(Field("FSPV", FSPV, "vec", "out", "forces", ("plot",)))
    store.define(Field("grav", GRAV, "real", "out", "environment"))
    store.define(Field("mass", MASS, "real", "out", "propulsion"))
    store.define(Field("dvbe", DVBE, "real", "out", "newton"))
    store.define(Field("pdynmc", PDYNMC, "real", "out", "environment"))
    store.define(Field("thrust", THRUST, "real", "out", "propulsion"))
    store.define(Field("area", AREA, "real", "data", "aerodynamics"))
    store.define(Field("cla", CLA, "real", "out", "aerodynamics"))
    store.set("ancomx", ancomx)
    store.set("phimvx", PHIMVX)
    store.set("alphax", alphax)
    store.set("gacp", GACP)
    store.set("ta", ta)
    store.set("anposlimx", ANPOSLIMX)
    store.set("anneglimx", ANNEGLIMX)
    store.set("alpposlimx", ALPPOSLIMX)
    store.set("alpneglimx", ALPNEGLIMX)
    store.set("alp", alp)
    return vehicle, control


def test_control_load_one_step_matches_cpp_equations():
    vehicle, control = _ready()
    expected = _expected_load(
        ANCOMX, INT_STEP, PHIMVX, ALPHAX, ANPOSLIMX, ANNEGLIMX,
        GACP, TA, ALPPOSLIMX, ALPNEGLIMX, FSPV, GRAV, MASS, DVBE,
        PDYNMC, THRUST, AREA, CLA, 0.0, 0.0, 0.0, 0.0,
    )
    alpx, xi, alp, tip, qq, anx = expected

    got = control.control_load(vehicle, ANCOMX, INT_STEP)

    store = vehicle.store
    assert _approx(got, alpx)
    assert _approx(store.get("xi"), xi)
    assert _approx(store.get("alp"), alp)
    assert _approx(store.get("tip"), tip)
    assert _approx(store.get("qq"), qq)
    assert _approx(store.get("anx"), anx)
    assert store.get("alphax") == ALPHAX
    assert store.get("ancomx") == ANCOMX


def test_ta_not_positive_zeros_integral():
    vehicle, control = _ready(ta=0.0)
    expected = _expected_load(
        ANCOMX, INT_STEP, PHIMVX, ALPHAX, ANPOSLIMX, ANNEGLIMX,
        GACP, 0.0, ALPPOSLIMX, ALPNEGLIMX, FSPV, GRAV, MASS, DVBE,
        PDYNMC, THRUST, AREA, CLA, 0.0, 0.0, 0.0, 0.0,
    )
    alpx, xi, alp, tip, qq, anx = expected

    got = control.control_load(vehicle, ANCOMX, INT_STEP)

    store = vehicle.store
    assert _approx(got, alpx)
    assert xi == 0.0
    assert store.get("xi") == 0.0
    assert _approx(store.get("qq"), qq)
    assert _approx(store.get("alp"), alp)
    assert _approx(store.get("tip"), tip)
    assert _approx(store.get("anx"), anx)
    assert store.get("alphax") == ALPHAX


def test_control_load_does_not_write_alphax():
    vehicle, control = _ready()
    alphax_before = vehicle.store.get("alphax")

    alpx = control.control_load(vehicle, ANCOMX, INT_STEP)

    assert alpx != alphax_before
    assert vehicle.store.get("alphax") == alphax_before
