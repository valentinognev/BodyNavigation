"""HYPER5 mcontrol 46: lateral + altitude + load (CRUISE5 composition)."""

import math

import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import cadtbv
from cadac.vehicles.round3.hyper5.control import Hyper5Control, _ALLOWED_MCONTROL

ALCOMX = 0.5
ALLIMX = 1.0
PHILIMX = 70.0
TPHI = 1.0
ANPOSLIMX = 2.0
ANNEGLIMX = -2.0
GACP = 10.0
TA = 0.8
ALPPOSLIMX = 6.0
ALPNEGLIMX = -4.0
MASS = 1352.0
DVBE = 254.0
AREA = 11.6986
PDYNMC = 72000.0
THRUST = 0.0
CLA = 0.08
GRAV = 9.81
FSPV = np.array([2.0, 1.0, -12.0])
ALPHAX = -1.5
PHIMVX = 0.0
INT_STEP = 0.05
ALT = 23900.0
ALTCOM = 24000.0
GH = 0.2
GV = 0.3
ALTDLIM = 50.0
VBEG = np.array([1475.0, 0.0, 40.0])
TGV = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
)

RTOL = 1e-12
ATOL = 1e-14


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ctx(int_step=INT_STEP):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _expected_lateral(alcomx, allimx, phimvx, alphax, fspv, grav):
    tbv = cadtbv(phimvx * RAD, alphax * RAD)
    fspb = tbv @ fspv
    anx = -fspb[2] / grav
    if alcomx > allimx:
        alcomx = allimx
    if alcomx < -allimx:
        alcomx = -allimx
    phic = math.atan2(alcomx, anx)
    phicx = phic * DEG
    alx = fspv[1] / grav
    return phicx, alx


def _expected_altitude(
    altcom, phimvx, alt, grav, vbeg, gh, gv, altdlim, anposlimx, anneglimx
):
    ealt = gh * (altcom - alt)
    if ealt > altdlim:
        ealt = altdlim
    if ealt < -altdlim:
        ealt = -altdlim
    altd = -vbeg[2]
    ancomx = (gv * (ealt - altd) / grav + 1) * (1 / math.cos(phimvx * RAD))
    if ancomx > anposlimx:
        ancomx = anposlimx
    if ancomx < anneglimx:
        ancomx = anneglimx
    return ancomx, altd


def _expected_load(
    ancomx,
    int_step,
    phimvx,
    alphax,
    anposlimx,
    anneglimx,
    gacp,
    ta,
    alpposlimx,
    alpneglimx,
    fspv,
    grav,
    mass,
    dvbe,
    pdynmc,
    thrust,
    area,
    cla,
    xi,
    xid,
    alp,
    alpd,
):
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
    return alpx


def _ready(
    mcontrol=46,
    alcomx=ALCOMX,
    ancomx=9.0,
    alphax=ALPHAX,
    phimvx=PHIMVX,
):
    vehicle = _Vehicle()
    control = Hyper5Control()
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
    store.define(Field("tgv", TGV, "mat", "init", "newton"))
    store.define(Field("alt", ALT, "real", "out", "newton", ("scrn", "plot")))
    store.define(Field("vbeg", VBEG, "vec", "state", "newton"))
    store.set("mcontrol", mcontrol)
    store.set("alcomx", alcomx)
    store.set("allimx", ALLIMX)
    store.set("philimx", PHILIMX)
    store.set("tphi", TPHI)
    store.set("anposlimx", ANPOSLIMX)
    store.set("anneglimx", ANNEGLIMX)
    store.set("gacp", GACP)
    store.set("ta", TA)
    store.set("alpposlimx", ALPPOSLIMX)
    store.set("alpneglimx", ALPNEGLIMX)
    store.set("alphax", alphax)
    store.set("phimvx", phimvx)
    store.set("ancomx", ancomx)
    store.set("altcom", ALTCOM)
    store.set("gh", GH)
    store.set("gv", GV)
    store.set("altdlim", ALTDLIM)
    return vehicle, control


def test_hyper5_mcontrol_46():
    assert 46 in _ALLOWED_MCONTROL

    vehicle, control = _ready(mcontrol=46, ancomx=9.0)
    store = vehicle.store

    phicx_lat, expected_alx = _expected_lateral(
        ALCOMX, ALLIMX, PHIMVX, ALPHAX, FSPV, GRAV
    )
    phicx_cmd = phicx_lat
    if phicx_cmd > PHILIMX:
        phicx_cmd = PHILIMX
    if phicx_cmd < -PHILIMX:
        phicx_cmd = -PHILIMX
    phixd_new = (phicx_cmd - 0.0) / TPHI
    phimvx_exp = integrate(phixd_new, 0.0, 0.0, INT_STEP)
    ancomx_exp, altd_exp = _expected_altitude(
        ALTCOM,
        phimvx_exp,
        ALT,
        GRAV,
        VBEG,
        GH,
        GV,
        ALTDLIM,
        ANPOSLIMX,
        ANNEGLIMX,
    )
    # control_load reads store phimvx (pre-update), matching modes 6/16/36/44
    alphax_exp = _expected_load(
        ancomx_exp,
        INT_STEP,
        PHIMVX,
        ALPHAX,
        ANPOSLIMX,
        ANNEGLIMX,
        GACP,
        TA,
        ALPPOSLIMX,
        ALPNEGLIMX,
        FSPV,
        GRAV,
        MASS,
        DVBE,
        PDYNMC,
        THRUST,
        AREA,
        CLA,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    tbv_exp = cadtbv(phimvx_exp * RAD, alphax_exp * RAD)
    tbg_exp = tbv_exp @ TGV.T

    control.execute(vehicle, _ctx())

    assert math.isfinite(store.get("phimvx"))
    assert math.isfinite(store.get("alphax"))
    assert _approx(store.get("phicx"), phicx_lat)
    assert _approx(store.get("phimvx"), phimvx_exp)
    assert _approx(store.get("ancomx"), ancomx_exp)
    assert store.get("ancomx") != 9.0
    assert _approx(store.get("altd"), altd_exp)
    assert _approx(store.get("alphax"), alphax_exp)
    assert store.get("alx") == expected_alx
    np.testing.assert_allclose(store.get("TBV"), tbv_exp, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("TBG"), tbg_exp, rtol=RTOL, atol=ATOL)
