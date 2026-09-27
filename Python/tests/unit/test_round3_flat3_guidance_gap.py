"""Round3/Flat3 guidance modes the Task 0 audit still raises."""

from math import cos, exp
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.cli import _resolve_vehicle
from cadac.constants import RAD
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.earth import cadine, cadtei, cadtge
from cadac.math.frames import mat2tr, polar_from_cart, skew
from cadac.tables.lookup import Datadeck

GRAV = 9.81
PHICX = 12.0
ANPOSLIMX = 5.0
ANNEGLIMX = -5.0
ALLIMX = 5.0
CLIP_ANPOS = 0.25
PHILIMX = 70.0
LINE_GAIN = 1.0
NL_GAIN_FACT = 0.6
DECREMENT = 1000.0
PSIFGX = 90.0
THTFGX = 0.0
THTVGX = 0.0
TIME = 0.0
LONX = 14.7
LATX = 35.4
ALT = 7000.0
PSIVGX = 90.0
DVBE = 200.0
WP_LONX = 14.9
WP_LATX = 35.4
WP_ALT = 0.0

SWEL = np.array([5000.0, 2000.0, 0.0])
SBEL = np.array([0.0, 0.0, -3500.0])
VBEL = np.array([200.0, 0.0, 0.0])
PSIFLX = 180.0
THTFLX = 0.0
THTVLX = 0.0

PRONAV_GAIN = 3.5
BIAS = 5.0
CLOSING_SPEED = 240.0
WOEB = np.array([0.02, -0.05, 0.01])
UTBB = np.array([0.95, 0.22, -0.22])
TBG = np.array(
    [
        [0.8, 0.0, 0.6],
        [0.0, 1.0, 0.0],
        [-0.6, 0.0, 0.8],
    ]
)

GAIN_THTVG = 0.5
THTVGCX = 5.0
GAMMA = 0.0
MASS = 100.0
AREA = 1.0
CLA = 2.0
PDYNMC = 5000.0
ALPPOSLIMX = 20.0
ALPNEGLIMX = -15.0
GAIN_PSIVG = 1.5
PSI_CMD = 10.0
PSI_STATE = 2.0
ALPHACX = 3.5
TPHI = 0.5
INT_STEP = 0.01

RTOL = 1e-12
ATOL = 1e-14

_SPECS = {
    "CRUISE5": ("cruise5", "CRUISE3"),
    "HYPER5": (None, "HYPER5"),
    "FALCON5": (None, "PLANE"),
}


def _vehicle(program):
    family, vtype = _SPECS[program]
    cls = _resolve_vehicle(family, vtype)
    deck = Datadeck.from_tables({})
    veh = cls("probe", aero_deck=deck, prop_deck=deck, events=[])
    veh.define()
    return veh


def _execute_module(veh, name):
    ctx = SimpleNamespace(int_step=INT_STEP)
    for module in veh.modules:
        if module.name == name:
            module.execute(veh, ctx)
            return
    raise AssertionError(f"module {name!r} not on vehicle")


def _plant(store, name, value, ftype):
    if name not in store:
        store.define(Field(name, value, ftype, "data", "test"))
    else:
        store.set(name, value)


def _clip(alcomx, ancomx, anposlimx, anneglimx, allimx):
    if ancomx > anposlimx:
        ancomx = anposlimx
    if ancomx < anneglimx:
        ancomx = anneglimx
    if alcomx > allimx:
        alcomx = allimx
    if alcomx < -allimx:
        alcomx = -allimx
    return alcomx, ancomx


def _round3_kinematics():
    tei = cadtei(TIME)
    tge = cadtge(LONX * RAD, LATX * RAD)
    tig = tei.T @ tge.T
    sbii = cadine(LONX * RAD, LATX * RAD, ALT, TIME)
    vbeg = mat2tr(PSIVGX * RAD, THTVGX * RAD).T @ np.array([DVBE, 0.0, 0.0])
    return tig, sbii, vbeg


def _expected_round3_algv(line_gain, thtvgx):
    tig, sbii, vbeg = _round3_kinematics()
    tfg = mat2tr(PSIFGX * RAD, THTFGX * RAD)
    swii = cadine(WP_LONX * RAD, WP_LATX * RAD, WP_ALT, TIME)
    swbg = tig.T @ (swii - sbii)
    polar = polar_from_cart(swbg)
    tog = mat2tr(float(polar[1]), float(polar[2]))
    vbeo = tog @ vbeg
    vbef = tfg @ vbeg
    nl_gain = NL_GAIN_FACT * (1.0 - exp(-float(polar[0]) / DECREMENT))
    algv3 = line_gain * (-vbeo[2] + nl_gain * vbef[2]) - GRAV * cos(thtvgx * RAD)
    return float(algv3)


def _expected_flat_algv(line_gain, thtvlx):
    tfl = mat2tr(PSIFLX * RAD, THTFLX * RAD)
    swbl = SWEL - SBEL
    polar = polar_from_cart(swbl)
    tol = mat2tr(float(polar[1]), float(polar[2]))
    vbeo = tol @ VBEL
    vbef = tfl @ VBEL
    nl_gain = NL_GAIN_FACT * (1.0 - exp(-float(polar[0]) / DECREMENT))
    algv3 = line_gain * (-vbeo[2] + nl_gain * vbef[2]) - GRAV * cos(thtvlx * RAD)
    return float(algv3)


def _expected_pronav(bias):
    grav_g = np.array([0.0, 0.0, GRAV + bias])
    return skew(WOEB) @ UTBB * (PRONAV_GAIN * CLOSING_SPEED) - TBG @ grav_g


def _plant_round3_line(store, line_gain, anposlimx):
    tig, sbii, vbeg = _round3_kinematics()
    _plant(store, "grav", GRAV, "real")
    _plant(store, "time", TIME, "real")
    _plant(store, "tig", tig, "mat")
    _plant(store, "sbii", sbii, "vec")
    _plant(store, "vbeg", vbeg, "vec")
    _plant(store, "thtvgx", THTVGX, "real")
    _plant(store, "philimx", PHILIMX, "real")
    _plant(store, "phicx", PHICX, "real")
    _plant(store, "anposlimx", anposlimx, "real")
    _plant(store, "anneglimx", ANNEGLIMX, "real")
    _plant(store, "allimx", ALLIMX, "real")
    _plant(store, "line_gain", line_gain, "real")
    _plant(store, "nl_gain_fact", NL_GAIN_FACT, "real")
    _plant(store, "decrement", DECREMENT, "real")
    _plant(store, "wp_lonx", WP_LONX, "real")
    _plant(store, "wp_latx", WP_LATX, "real")
    _plant(store, "wp_alt", WP_ALT, "real")
    _plant(store, "psifgx", PSIFGX, "real")
    _plant(store, "thtfgx", THTFGX, "real")
    _plant(store, "alcomx", 4.0, "real")
    _plant(store, "ancomx", 4.0, "real")


def _plant_falcon_line(store, line_gain, anposlimx):
    _plant(store, "grav", GRAV, "real")
    _plant(store, "SBEL", SBEL, "vec")
    _plant(store, "VBEL", VBEL, "vec")
    _plant(store, "thtvlx", THTVLX, "real")
    _plant(store, "philimx", PHILIMX, "real")
    _plant(store, "phicx", PHICX, "real")
    _plant(store, "anposlimx", anposlimx, "real")
    _plant(store, "anneglimx", ANNEGLIMX, "real")
    _plant(store, "allimx", ALLIMX, "real")
    _plant(store, "line_gain", line_gain, "real")
    _plant(store, "nl_gain_fact", NL_GAIN_FACT, "real")
    _plant(store, "decrement", DECREMENT, "real")
    _plant(store, "swel1", float(SWEL[0]), "real")
    _plant(store, "swel2", float(SWEL[1]), "real")
    _plant(store, "swel3", float(SWEL[2]), "real")
    _plant(store, "psiflx", PSIFLX, "real")
    _plant(store, "thtflx", THTFLX, "real")
    _plant(store, "alcomx", 4.0, "real")
    _plant(store, "ancomx", 4.0, "real")


def _plant_pronav(store, bias):
    _plant(store, "grav", GRAV, "real")
    _plant(store, "phicx", PHICX, "real")
    _plant(store, "anposlimx", ANPOSLIMX, "real")
    _plant(store, "anneglimx", ANNEGLIMX, "real")
    _plant(store, "allimx", ALLIMX, "real")
    _plant(store, "pronav_gain", PRONAV_GAIN, "real")
    _plant(store, "TBG", TBG, "mat")
    _plant(store, "WOEB", WOEB, "vec")
    _plant(store, "UTBB", UTBB, "vec")
    _plant(store, "closing_speed", CLOSING_SPEED, "real")
    _plant(store, "alcomx", 4.0, "real")
    _plant(store, "ancomx", 4.0, "real")
    if bias is not None:
        _plant(store, "bias", bias, "real")


def _heading_cmd_name(program):
    if program == "FALCON5":
        return "psivlcx"
    return "psivgcx"


def _heading_state_name(program):
    if program == "FALCON5":
        return "psivlx"
    return "psivgx"


def _gamma_state_name(program):
    if program == "FALCON5":
        return "thtvl"
    return "thtvg"


def _expected_heading(gain, cmd, state):
    if abs(cmd) <= 135.0:
        comp = state
    elif state * cmd >= 0.0:
        comp = state
    else:
        sign = 1.0 if state >= 0.0 else -1.0
        comp = 360.0 - state * sign
    return gain * (cmd - comp)


def _expected_bank(phicx):
    limited = phicx
    if limited > PHILIMX:
        limited = PHILIMX
    if limited < -PHILIMX:
        limited = -PHILIMX
    phixd_new = (limited - 0.0) / TPHI
    return integrate(phixd_new, 0.0, 0.0, INT_STEP)


def _expected_flightpath(phimvx):
    avx = GAIN_THTVG * (THTVGCX * RAD - GAMMA)
    anx = avx / cos(phimvx * RAD)
    alphax = (anx * MASS * GRAV) / (PDYNMC * AREA * CLA)
    if alphax > ALPPOSLIMX:
        alphax = ALPPOSLIMX
    if alphax < ALPNEGLIMX:
        alphax = ALPNEGLIMX
    return alphax


def _plant_control(store, program):
    _plant(store, "gain_thtvg", GAIN_THTVG, "real")
    _plant(store, "thtvgcx", THTVGCX, "real")
    _plant(store, _gamma_state_name(program), GAMMA, "real")
    _plant(store, "grav", GRAV, "real")
    _plant(store, "mass", MASS, "real")
    _plant(store, "area", AREA, "real")
    _plant(store, "cla", CLA, "real")
    _plant(store, "pdynmc", PDYNMC, "real")
    _plant(store, "alpposlimx", ALPPOSLIMX, "real")
    _plant(store, "alpneglimx", ALPNEGLIMX, "real")
    _plant(store, "gain_psivg", GAIN_PSIVG, "real")
    _plant(store, _heading_cmd_name(program), PSI_CMD, "real")
    _plant(store, _heading_state_name(program), PSI_STATE, "real")
    _plant(store, "alphacx", ALPHACX, "real")
    _plant(store, "philimx", PHILIMX, "real")
    _plant(store, "tphi", TPHI, "real")
    _plant(store, "phix", 0.0, "real")
    _plant(store, "phixd", 0.0, "real")
    _plant(store, "tgv", np.eye(3), "mat")


@pytest.mark.parametrize("program", ["CRUISE5", "HYPER5"])
def test_pitch_line_guidance(program):
    veh = _vehicle(program)
    _plant_round3_line(veh.store, LINE_GAIN, ANPOSLIMX)
    veh.store.set("mguidance", 3)
    algv3 = _expected_round3_algv(LINE_GAIN, THTVGX)
    _, want_an = _clip(0.0, -algv3 / GRAV, ANPOSLIMX, ANNEGLIMX, ALLIMX)
    assert want_an != 0.0

    _execute_module(veh, "guidance")

    assert veh.store.get("alcomx") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert veh.store.get("ancomx") == pytest.approx(want_an, rel=RTOL, abs=ATOL)
    assert veh.store.get("phicx") == PHICX


def test_falcon5_pitch_line_guidance():
    veh = _vehicle("FALCON5")
    _plant_falcon_line(veh.store, LINE_GAIN, ANPOSLIMX)
    veh.store.set("mguidance", 3)
    algv3 = _expected_flat_algv(LINE_GAIN, THTVLX)
    _, want_an = _clip(0.0, -algv3 / GRAV, ANPOSLIMX, ANNEGLIMX, ALLIMX)
    assert want_an != 0.0

    _execute_module(veh, "guidance")

    assert veh.store.get("alcomx") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert veh.store.get("ancomx") == pytest.approx(want_an, rel=RTOL, abs=ATOL)
    assert veh.store.get("phicx") == PHICX


@pytest.mark.parametrize("program", ["CRUISE5", "HYPER5", "FALCON5"])
def test_pitch_line_clips_ancomx(program):
    veh = _vehicle(program)
    if program == "FALCON5":
        _plant_falcon_line(veh.store, 0.0, CLIP_ANPOS)
        algv3 = _expected_flat_algv(0.0, THTVLX)
    else:
        _plant_round3_line(veh.store, 0.0, CLIP_ANPOS)
        algv3 = _expected_round3_algv(0.0, THTVGX)
    raw = -algv3 / GRAV
    assert raw > CLIP_ANPOS
    veh.store.set("mguidance", 3)

    _execute_module(veh, "guidance")

    assert veh.store.get("ancomx") == pytest.approx(CLIP_ANPOS, rel=RTOL, abs=ATOL)
    assert veh.store.get("alcomx") == pytest.approx(0.0, rel=RTOL, abs=ATOL)


@pytest.mark.parametrize("program", ["CRUISE5", "HYPER5"])
@pytest.mark.parametrize("mguidance", [6, 60])
def test_pronav_pitch_and_lateral(program, mguidance):
    bias = BIAS if program == "HYPER5" else None
    veh = _vehicle(program)
    _plant_pronav(veh.store, bias)
    veh.store.set("mguidance", mguidance)
    apnb = _expected_pronav(0.0 if bias is None else bias)
    if mguidance == 60:
        want_al, want_an = _clip(
            float(apnb[1] / GRAV), 0.0, ANPOSLIMX, ANNEGLIMX, ALLIMX
        )
    else:
        want_al, want_an = _clip(
            0.0, float(-apnb[2] / GRAV), ANPOSLIMX, ANNEGLIMX, ALLIMX
        )
    assert want_al != 0.0 or want_an != 0.0

    _execute_module(veh, "guidance")

    assert veh.store.get("alcomx") == pytest.approx(want_al, rel=RTOL, abs=ATOL)
    assert veh.store.get("ancomx") == pytest.approx(want_an, rel=RTOL, abs=ATOL)
    assert veh.store.get("phicx") == PHICX


@pytest.mark.parametrize("mguidance", [6, 60])
def test_falcon5_pronav_modes_stay_unknown(mguidance):
    veh = _vehicle("FALCON5")
    veh.store.set("mguidance", mguidance)
    with pytest.raises(ValueError, match="unknown mguidance"):
        _execute_module(veh, "guidance")


@pytest.mark.parametrize("program", ["CRUISE5", "HYPER5", "FALCON5"])
def test_mcontrol_1_flightpath(program):
    veh = _vehicle(program)
    _plant_control(veh.store, program)
    veh.store.set("mcontrol", 1)
    want_alpha = _expected_flightpath(0.0)
    assert want_alpha != 0.0

    _execute_module(veh, "control")

    assert veh.store.get("phimvx") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert veh.store.get("alphax") == pytest.approx(want_alpha, rel=RTOL, abs=ATOL)


@pytest.mark.parametrize("program", ["CRUISE5", "HYPER5", "FALCON5"])
def test_mcontrol_10_heading_and_alpha(program):
    veh = _vehicle(program)
    _plant_control(veh.store, program)
    veh.store.set("mcontrol", 10)
    phicx = _expected_heading(GAIN_PSIVG, PSI_CMD, PSI_STATE)
    want_phi = _expected_bank(phicx)

    _execute_module(veh, "control")

    assert veh.store.get("phicx") == pytest.approx(phicx, rel=RTOL, abs=ATOL)
    assert veh.store.get("phimvx") == pytest.approx(want_phi, rel=RTOL, abs=ATOL)
    assert veh.store.get("alphax") == pytest.approx(ALPHACX, rel=RTOL, abs=ATOL)


@pytest.mark.parametrize("program", ["CRUISE5", "HYPER5", "FALCON5"])
def test_mcontrol_11_heading_and_flightpath(program):
    veh = _vehicle(program)
    _plant_control(veh.store, program)
    veh.store.set("mcontrol", 11)
    phicx = _expected_heading(GAIN_PSIVG, PSI_CMD, PSI_STATE)
    want_phi = _expected_bank(phicx)
    want_alpha = _expected_flightpath(want_phi)

    _execute_module(veh, "control")

    assert veh.store.get("phicx") == pytest.approx(phicx, rel=RTOL, abs=ATOL)
    assert veh.store.get("phimvx") == pytest.approx(want_phi, rel=RTOL, abs=ATOL)
    assert veh.store.get("alphax") == pytest.approx(want_alpha, rel=RTOL, abs=ATOL)
