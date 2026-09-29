"""FALCON6 maut pitch digits 2 (pitch rate), 3 (normal accel), 5 (altitude hold)."""

from math import cos, isfinite, sqrt
from types import SimpleNamespace

import pytest

from cadac.constants import AGRAV, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.falcon6.control import Plane6Control

SMALL = 1.0e-7

INT_STEP = 0.001
GMAX = 1.0e12
GMINX = -1.0e12
GRAV = AGRAV
ALTCOM = 1100.0
HBE = 1000.0
GAINALT = 0.3
GAINALTRATE = 0.7
ANLIMPX = 9.0
ANLIMNX = 6.0
VBEL = (180.0, 0.0, 2.0)


def _ctx(int_step=INT_STEP):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _falcon6(**overrides):
    vehicle = SimpleNamespace(store=StateStore(), modules=[])
    ctrl = Plane6Control()
    vehicle.modules.append(ctrl)
    ctrl.define(vehicle)
    store = vehicle.store
    store.set("mroll", 0)
    store.set("wrcl", 15.0)
    store.set("zrcl", 0.7)
    store.set("dalimx", 20.0)
    store.set("delimx", 20.0)
    store.set("drlimx", 20.0)
    store.set("philimx", 70.0)
    store.set("phicomx", 0.0)
    store.set("ancomx", 1.0)
    store.set("gainp", 0.0)
    store.set("gainl", 1.0)
    store.set("waclp", 2.0)
    store.set("zaclp", 0.7)
    store.set("paclp", 5.0)
    store.set("altcom", ALTCOM)
    store.set("gainalt", GAINALT)
    store.set("gainaltrate", GAINALTRATE)
    store.set("anlimpx", ANLIMPX)
    store.set("anlimnx", ANLIMNX)
    store.set("zetlagr", 0.7)
    store.set("facthead", -0.9)
    store.set("rcomx", 0.0)
    store.define(Field("phiblx", 0.0, "real", "diag", "kinematics"))
    store.define(Field("ppx", 0.0, "real", "out", "euler"))
    store.define(Field("dllp", -2.5, "real", "out", "aerodynamics"))
    store.define(Field("dllda", 18.0, "real", "out", "aerodynamics"))
    store.define(Field("dla", 8.0, "real", "out", "aerodynamics"))
    store.define(Field("dlde", -12.0, "real", "out", "aerodynamics"))
    store.define(Field("dma", -2.0, "real", "out", "aerodynamics"))
    store.define(Field("dmq", -1.5, "real", "out", "aerodynamics"))
    store.define(Field("dmde", -8.0, "real", "out", "aerodynamics"))
    store.define(Field("qqx", 4.0, "real", "out", "euler"))
    store.define(Field("dvbe", 180.0, "real", "out", "newton"))
    store.define(Field("pdynmc", 5500.0, "real", "out", "environment"))
    store.define(Field("FSPB", (0.0, 0.0, -AGRAV), "vec", "out", "newton"))
    store.define(Field("gmax", GMAX, "real", "out", "aerodynamics"))
    store.define(Field("gminx", GMINX, "real", "out", "aerodynamics"))
    store.define(Field("grav", GRAV, "real", "out", "environment"))
    store.define(Field("hbe", HBE, "real", "out", "newton"))
    store.define(Field("VBEL", VBEL, "vec", "out", "newton"))
    store.define(Field("dyb", -6.0, "real", "out", "aerodynamics"))
    store.define(Field("dydr", 4.0, "real", "out", "aerodynamics"))
    store.define(Field("dnb", 1.5, "real", "out", "aerodynamics"))
    store.define(Field("dnr", -0.8, "real", "out", "aerodynamics"))
    store.define(Field("dndr", -3.0, "real", "out", "aerodynamics"))
    store.define(Field("rrx", 2.0, "real", "out", "euler"))
    store.define(Field("psivlx", 0.0, "real", "out", "newton"))
    for name, value in overrides.items():
        store.set(name, value)
    ctrl.initialize(vehicle, _ctx())
    return vehicle


def _exec(veh, name):
    module = next(item for item in veh.modules if item.name == name)
    module.execute(veh, _ctx())


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _expected_altitude_ancomx(store):
    gainalt = store.get("gainalt")
    gainaltrate = store.get("gainaltrate")
    grav = store.get("grav")
    phiblx = store.get("phiblx")
    vbel = store.get("VBEL")
    hbe = store.get("hbe")
    altcom = store.get("altcom")
    altrate = -vbel[2]
    eh = gainalt * (altcom - hbe)
    if phiblx == 0:
        phiblx = 1.0e-7
    ancomx = (1.0 / cos(phiblx * RAD)) * (gainaltrate * (eh - altrate) + grav) / AGRAV
    anlimpx = store.get("anlimpx")
    anlimnx = store.get("anlimnx")
    if ancomx > anlimpx:
        ancomx = anlimpx
    if ancomx < -anlimnx:
        ancomx = -anlimnx
    gmax = store.get("gmax")
    gminx = store.get("gminx")
    if ancomx > gmax:
        ancomx = gmax
    if ancomx < gminx:
        ancomx = gminx
    return ancomx


def _expected_pitch_rate_delecx(store, qcomx):
    zetlagr = store.get("zetlagr")
    dla = store.get("dla")
    dlde = store.get("dlde")
    dma = store.get("dma")
    dmq = store.get("dmq")
    dmde = store.get("dmde")
    qqx = store.get("qqx")
    dvbe = store.get("dvbe")
    zrate = dla / dvbe - dma * dlde / (dvbe * dmde)
    aa = dla / dvbe - dmq
    bb = -dma - dmq * dla / dvbe
    dum1 = aa - 2.0 * zetlagr * zetlagr * zrate
    dum2 = aa * aa - 4.0 * zetlagr * zetlagr * bb
    radix = dum1 * dum1 - dum2
    if radix < 0.0:
        radix = 0.0
    if abs(dmde) < SMALL:
        dmde = SMALL * _sign(dmde)
    grate = (-dum1 + sqrt(radix)) / (-dmde)
    return grate * (qqx - qcomx)


def test_falcon6_maut_pitch_digit_2():
    qcomx = 1.5
    veh = _falcon6(maut=32, qcomx=qcomx, alcomx=0.5)
    want = _expected_pitch_rate_delecx(veh.store, qcomx)
    _exec(veh, "control")
    assert isfinite(veh.store.get("delecx"))
    assert veh.store.get("delecx") == pytest.approx(want)
    assert veh.store.get("delecx") != 0.0


def test_falcon6_maut_33_calls_normal_accel():
    veh = _falcon6()
    veh.store.set("maut", 33)
    veh.store.set("ancomx", 1.0)
    _exec(veh, "control")
    assert veh.store.get("delecx") is not None


def test_falcon6_maut_35_altitude_hold_to_normal_accel():
    veh = _falcon6(maut=35, ancomx=1.0, alcomx=0.5)
    want = _expected_altitude_ancomx(veh.store)
    _exec(veh, "control")
    assert isfinite(veh.store.get("delecx"))
    assert veh.store.get("ancomx") == pytest.approx(want)


def test_falcon6_maut_45_altitude_hold_to_normal_accel():
    veh = _falcon6(
        maut=45,
        ancomx=1.0,
        psivlcomx=5.0,
        facthead=-0.9,
        psivlx=0.0,
    )
    # control_heading needs grav/dvbe already in fixture
    want = _expected_altitude_ancomx(veh.store)
    _exec(veh, "control")
    assert isfinite(veh.store.get("delecx"))
    assert veh.store.get("ancomx") == pytest.approx(want)
