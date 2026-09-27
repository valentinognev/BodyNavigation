"""FALCON6 line guidance (mguid 30/33) and accel/heading control (mauty 3/4)."""

from math import atan, cos, exp, sqrt, tan
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import AGRAV, DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr, polar_from_cart
from cadac.vehicles.flat6.falcon6.control import SMALL, Plane6Control
from cadac.vehicles.flat6.falcon6.guidance import Plane6Guidance

RTOL = 1e-12
ATOL = 1e-12

GRAV = AGRAV
LINE_GAIN = 0.15
NL_GAIN_FACT = 1.2
DECREMENT = 8000.0
PSIFLX = 15.0
THTFLX = -3.0
THTVLX = 4.0
DVBE = 200.0
PHILIMX = 45.0
SBEL = (0.0, 0.0, 2000.0)
SWEL = (20000.0, 3000.0, 1000.0)
VBEL = (180.0, 20.0, -5.0)

FLEET_SBEL = (0.0, 0.0, 0.0)
FLEET_SWEL = (500.0, 0.0, 100.0)
FLEET_VBEL = (-80.0, 10.0, 0.0)
FLEET_DVBE = 100.0
FLEET_PHILIMX = 60.0

GAINL = 1.0
ALCOMX = 0.2
WRCL = 3.0
ZRCL = 0.7
DLLP = -1.0
DLLDA = 20.0
PHIBLX = 0.0
PPX = 0.0
DALIMX = 100.0
DELIMX = 100.0
DRLIMX = 100.0
CTRL_PHILIMX = 80.0
FACTHEAD = 0.0
PSIVLCOMX = 11.0
PSIVLX = 10.0
HEAD_DVBE = 120.0
ZETLAGR = 0.7
DLA = 8.0
DLDE = -12.0
DMA = -2.0
DMQ = -1.5
DMDE = -8.0
DYB = -6.0
DYDR = 4.0
DNB = 1.5
DNR = -0.8
DNDR = -3.0
RRX = 2.0
RCOMX = 0.0


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.001,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _cadac_sign(variable):
    if variable < 0.0:
        return -1
    return 1


def _limit(value, lim):
    if abs(value) > lim:
        return lim * _cadac_sign(value)
    return value


def _guidance_vehicle(
    mguid,
    *,
    sbel=SBEL,
    swel=SWEL,
    vbel=VBEL,
    dvbe=DVBE,
    philimx=PHILIMX,
    thtvlx=THTVLX,
):
    vehicle = SimpleNamespace(store=StateStore())
    guid = Plane6Guidance()
    guid.define(vehicle)
    store = vehicle.store
    store.set("mguid", mguid)
    store.set("line_gain", LINE_GAIN)
    store.set("nl_gain_fact", NL_GAIN_FACT)
    store.set("decrement", DECREMENT)
    store.set("swel1", swel[0])
    store.set("swel2", swel[1])
    store.set("swel3", swel[2])
    store.set("psiflx", PSIFLX)
    store.set("thtflx", THTFLX)
    store.define(Field("grav", GRAV, "real", "out", "newton"))
    store.define(Field("SBEL", sbel, "vec", "out", "newton"))
    store.define(Field("VBEL", vbel, "vec", "out", "newton"))
    store.define(Field("dvbe", dvbe, "real", "out", "newton"))
    store.define(Field("thtvlx", thtvlx, "real", "out", "newton"))
    store.define(Field("philimx", philimx, "real", "data", "control"))
    store.define(Field("alcomx", 9.0, "real", "data", "control"))
    store.define(Field("ancomx", 9.0, "real", "data", "control"))
    guid.initialize(vehicle, _ctx())
    return vehicle, guid


def _expected_line_guidance(store):
    """C++ Plane::guidance + guidance_line outputs for mguid 30/33."""
    mguid = store.get("mguid")
    swbl = np.array(
        [
            store.get("swel1") - store.get("SBEL")[0],
            store.get("swel2") - store.get("SBEL")[1],
            store.get("swel3") - store.get("SBEL")[2],
        ],
        dtype=float,
    )
    vbel = np.asarray(store.get("VBEL"), dtype=float)
    grav = store.get("grav")
    dvbe = store.get("dvbe")
    philimx = store.get("philimx")
    thtvlx = store.get("thtvlx")
    dwb = sqrt(float(swbl @ swbl))
    dwbh = sqrt(swbl[0] * swbl[0] + swbl[1] * swbl[1])
    turn_min = dvbe * dvbe / (grav * tan(philimx * RAD))
    wp_flag = 1
    if dwbh < (2.0 * turn_min):
        closing = swbl[0] * vbel[0] + swbl[1] * vbel[1]
        wp_flag = _cadac_sign(closing)
    polar = polar_from_cart(swbl)
    wp_sltrange = float(polar[0])
    tol = mat2tr(float(polar[1]), float(polar[2]))
    tfl = mat2tr(store.get("psiflx") * RAD, store.get("thtflx") * RAD)
    vbeo = tol @ vbel
    vbef = tfl @ vbel
    nl_gain = store.get("nl_gain_fact") * (1.0 - exp(-wp_sltrange / store.get("decrement")))
    line_gain = store.get("line_gain")
    algv2 = line_gain * (-vbeo[1] + nl_gain * vbef[1])
    algv3 = line_gain * (-vbeo[2] + nl_gain * vbef[2]) - grav * cos(thtvlx * RAD)
    mguidl = mguid // 10
    mguidp = mguid % 10
    alcomx = algv2 / grav if mguidl == 3 else 0.0
    ancomx = -algv3 / grav if mguidp == 3 else 0.0
    return {
        "dwb": dwb,
        "dwbh": dwbh,
        "turn_min": turn_min,
        "wp_flag": wp_flag,
        "nl_gain": nl_gain,
        "SWBL": swbl,
        "VBEO": vbeo,
        "VBEF": vbef,
        "alcomx": alcomx,
        "ancomx": ancomx,
    }


def _control_vehicle(maut, *, fspb3=-20.0):
    vehicle = SimpleNamespace(store=StateStore())
    ctrl = Plane6Control()
    ctrl.define(vehicle)
    store = vehicle.store
    store.set("maut", maut)
    store.set("mroll", 0)
    store.set("wrcl", WRCL)
    store.set("zrcl", ZRCL)
    store.set("zetlagr", ZETLAGR)
    store.set("dalimx", DALIMX)
    store.set("delimx", DELIMX)
    store.set("drlimx", DRLIMX)
    store.set("philimx", CTRL_PHILIMX)
    store.set("gainl", GAINL)
    store.set("alcomx", ALCOMX)
    store.set("facthead", FACTHEAD)
    store.set("psivlcomx", PSIVLCOMX)
    store.set("rcomx", RCOMX)
    store.set("phicomx", 99.0)
    store.set("ancomx", 1.5)
    store.define(Field("phiblx", PHIBLX, "real", "diag", "kinematics"))
    store.define(Field("ppx", PPX, "real", "out", "euler"))
    store.define(Field("dllp", DLLP, "real", "out", "aerodynamics"))
    store.define(Field("dllda", DLLDA, "real", "out", "aerodynamics"))
    store.define(Field("dla", DLA, "real", "out", "aerodynamics"))
    store.define(Field("dlde", DLDE, "real", "out", "aerodynamics"))
    store.define(Field("dma", DMA, "real", "out", "aerodynamics"))
    store.define(Field("dmq", DMQ, "real", "out", "aerodynamics"))
    store.define(Field("dmde", DMDE, "real", "out", "aerodynamics"))
    store.define(Field("dyb", DYB, "real", "out", "aerodynamics"))
    store.define(Field("dydr", DYDR, "real", "out", "aerodynamics"))
    store.define(Field("dnb", DNB, "real", "out", "aerodynamics"))
    store.define(Field("dnr", DNR, "real", "out", "aerodynamics"))
    store.define(Field("dndr", DNDR, "real", "out", "aerodynamics"))
    store.define(Field("rrx", RRX, "real", "out", "euler"))
    store.define(Field("dvbe", HEAD_DVBE, "real", "out", "newton"))
    store.define(Field("grav", GRAV, "real", "out", "newton"))
    store.define(Field("psivlx", PSIVLX, "real", "out", "newton"))
    store.define(Field("FSPB", (1.0, -2.0, fspb3), "vec", "out", "newton"))
    ctrl.initialize(vehicle, _ctx())
    return vehicle, ctrl


def _expected_roll(phicomx):
    gkp = (2.0 * ZRCL * WRCL + DLLP) / DLLDA
    gkphi = WRCL * WRCL / DLLDA
    ephi = gkphi * (phicomx - PHIBLX) * RAD
    dpc = ephi - gkp * PPX * RAD
    delacx = _limit(dpc * DEG, DALIMX)
    return delacx, gkp, gkphi


def _expected_lateral_phicomx(fspb3):
    """C++ Plane::control_lateral_accel, then the roll-position limiter."""
    phicomx = -DEG * GAINL * atan(ALCOMX) * _cadac_sign(fspb3)
    return _limit(phicomx, CTRL_PHILIMX)


def _expected_heading():
    """C++ Plane::control_heading gain and bank command."""
    gainpsi = (HEAD_DVBE / GRAV) * ZRCL * WRCL * (1.0 - ZRCL * ZRCL) * (1.0 + FACTHEAD)
    phicomx = gainpsi * (PSIVLCOMX - PSIVLX)
    return gainpsi, _limit(phicomx, CTRL_PHILIMX)


def _expected_yaw():
    """C++ Plane::control_yaw_rate command for rcomx."""
    zrate = -DYB / HEAD_DVBE + DNB * DYDR / (HEAD_DVBE * DNDR)
    aa = -DYB / HEAD_DVBE - DNR
    bb = DNB + DYB * DNR / HEAD_DVBE
    dum1 = aa - 2.0 * ZETLAGR * ZETLAGR * zrate
    dum2 = aa * aa - 4.0 * ZETLAGR * ZETLAGR * bb
    radix = dum1 * dum1 - dum2
    if radix < 0.0:
        radix = 0.0
    dndr = DNDR
    if abs(dndr) < SMALL:
        dndr = SMALL * _cadac_sign(dndr)
    grate = (-dum1 + sqrt(radix)) / (-dndr)
    dum3 = grate * dndr * zrate
    radix = bb + dum3
    if radix < 0.0:
        radix = 0.0
    wnlagr = sqrt(radix)
    delrcx = _limit(grate * (RRX - RCOMX), DRLIMX)
    return delrcx, zrate, grate, wnlagr


def test_mguid_30_writes_lateral_line_command_and_diagnostics():
    vehicle, guid = _guidance_vehicle(30)
    want = _expected_line_guidance(vehicle.store)
    assert want["alcomx"] != 0.0
    assert want["nl_gain"] != 0.0
    assert want["ancomx"] == 0.0
    assert want["wp_flag"] == 1
    guid.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("alcomx") == pytest.approx(want["alcomx"], rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(0.0, abs=ATOL)
    assert store.get("nl_gain") == pytest.approx(want["nl_gain"], rel=RTOL, abs=ATOL)
    assert store.get("dwb") == pytest.approx(want["dwb"], rel=RTOL, abs=ATOL)
    assert store.get("dwbh") == pytest.approx(want["dwbh"], rel=RTOL, abs=ATOL)
    assert store.get("turn_min") == pytest.approx(want["turn_min"], rel=RTOL, abs=ATOL)
    assert store.get("wp_flag") == 1
    np.testing.assert_allclose(store.get("SWBL"), want["SWBL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VBEO"), want["VBEO"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VBEF"), want["VBEF"], rtol=RTOL, atol=ATOL)


def test_mguid_33_writes_normal_and_lateral_line_commands():
    vehicle, guid = _guidance_vehicle(33)
    want = _expected_line_guidance(vehicle.store)
    assert want["ancomx"] != 0.0
    assert want["alcomx"] != 0.0
    guid.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("alcomx") == pytest.approx(want["alcomx"], rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(want["ancomx"], rel=RTOL, abs=ATOL)
    assert store.get("nl_gain") == pytest.approx(want["nl_gain"], rel=RTOL, abs=ATOL)
    assert store.get("dwbh") == pytest.approx(want["dwbh"], rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(store.get("VBEF"), want["VBEF"], rtol=RTOL, atol=ATOL)


def test_mguid_30_fleeting_sets_wp_flag_minus_one():
    vehicle, guid = _guidance_vehicle(
        30,
        sbel=FLEET_SBEL,
        swel=FLEET_SWEL,
        vbel=FLEET_VBEL,
        dvbe=FLEET_DVBE,
        philimx=FLEET_PHILIMX,
    )
    want = _expected_line_guidance(vehicle.store)
    assert want["wp_flag"] == -1
    assert want["dwbh"] < 2.0 * want["turn_min"]
    guid.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("wp_flag") == -1
    assert store.get("dwbh") == pytest.approx(want["dwbh"], rel=RTOL, abs=ATOL)
    assert store.get("turn_min") == pytest.approx(want["turn_min"], rel=RTOL, abs=ATOL)
    assert store.get("alcomx") == pytest.approx(want["alcomx"], rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(store.get("SWBL"), want["SWBL"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("fspb3", (-20.0, 15.0))
def test_maut_30_lateral_accel_commands_bank(fspb3):
    vehicle, ctrl = _control_vehicle(30, fspb3=fspb3)
    phicomx = _expected_lateral_phicomx(fspb3)
    delacx, gkp, gkphi = _expected_roll(phicomx)
    assert phicomx != 0.0
    assert delacx != 0.0
    assert _cadac_sign(phicomx) == -_cadac_sign(fspb3)
    ctrl.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("phicomx") == pytest.approx(phicomx, rel=RTOL, abs=ATOL)
    assert store.get("delacx") == pytest.approx(delacx, rel=RTOL, abs=ATOL)
    assert store.get("delecx") == pytest.approx(0.0, abs=ATOL)
    assert store.get("delrcx") == pytest.approx(0.0, abs=ATOL)
    assert store.get("gkp") == pytest.approx(gkp, rel=RTOL, abs=ATOL)
    assert store.get("gkphi") == pytest.approx(gkphi, rel=RTOL, abs=ATOL)
    assert store.get("gainpsi") == 0.0


def test_maut_40_heading_commands_bank_and_yaw_rate():
    vehicle, ctrl = _control_vehicle(40)
    gainpsi, phicomx = _expected_heading()
    delacx, gkp, gkphi = _expected_roll(phicomx)
    delrcx, zrate, grate, wnlagr = _expected_yaw()
    assert gainpsi != 0.0
    assert phicomx != 0.0
    assert delrcx != 0.0
    ctrl.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("gainpsi") == pytest.approx(gainpsi, rel=RTOL, abs=ATOL)
    assert store.get("phicomx") == pytest.approx(phicomx, rel=RTOL, abs=ATOL)
    assert store.get("delacx") == pytest.approx(delacx, rel=RTOL, abs=ATOL)
    assert store.get("delrcx") == pytest.approx(delrcx, rel=RTOL, abs=ATOL)
    assert store.get("delecx") == pytest.approx(0.0, abs=ATOL)
    assert store.get("gkp") == pytest.approx(gkp, rel=RTOL, abs=ATOL)
    assert store.get("gkphi") == pytest.approx(gkphi, rel=RTOL, abs=ATOL)
    assert store.get("zrate") == pytest.approx(zrate, rel=RTOL, abs=ATOL)
    assert store.get("grate") == pytest.approx(grate, rel=RTOL, abs=ATOL)
    assert store.get("wnlagr") == pytest.approx(wnlagr, rel=RTOL, abs=ATOL)
