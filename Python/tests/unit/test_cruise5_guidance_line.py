from math import cos, exp, hypot, sin, sqrt, tan

import numpy as np
import pytest

from cadac.constants import RAD
from cadac.kernel.state import Field, StateStore
from cadac.math.earth import cadine, cadtei, cadtge
from cadac.math.frames import mat2tr, polar_from_cart
from cadac.vehicles.round3.cruise5.guidance import Cruise5Guidance

RTOL = 1e-12
ATOL = 1e-14

WP_LONX = 14.9
WP_LATX = 35.4
WP_ALT = 0.0
PSIFGX = 90.0
THTFGX = 0.0
LINE_GAIN = 1.0
NL_GAIN_FACT = 0.6
DECREMENT = 1000.0
LONX = 14.7
LATX = 35.4
ALT = 7000.0
PSIVGX = 90.0
THTVGX = 0.0
DVBE = 200.0
PHILIMX = 70.0
TIME = 0.0
GRAV = 9.81

# C++ Cruise::def_guidance roles/outputs. "dia" maps to Python "diag".
DEF_GUIDANCE = {
    "mguidance": ("int", "data", ("scrn",), 0),
    "pronav_gain": ("real", "data", (), 0.0),
    "line_gain": ("real", "data", (), 0.0),
    "nl_gain_fact": ("real", "data", (), 1.0),
    "decrement": ("real", "data", (), 0.0),
    "wp_lonx": ("real", "data", (), 0.0),
    "wp_latx": ("real", "data", (), 0.0),
    "wp_alt": ("real", "data", (), 0.0),
    "psifgx": ("real", "data", (), 0.0),
    "thtfgx": ("real", "data", (), 0.0),
    "point_gain": ("real", "data", (), 0.0),
    "wp_sltrange": ("real", "diag", (), 999999.0),
    "nl_gain": ("real", "diag", (), 0.0),
    "VBEO": ("vec", "diag", (), None),
    "VBEF": ("vec", "diag", (), None),
    "wp_grdrange": ("real", "diag", ("scrn", "plot"), 999999.0),
    "SWBG": ("vec", "out", (), None),
    "rad_min": ("real", "diag", (), 0.0),
    "rad_geometric": ("real", "diag", (), 0.0),
    "wp_flag": ("int", "diag", ("plot",), 0),
}


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _expected_line(wp_lonx, wp_latx, wp_alt, psifgx, thtfgx, line_gain,
                   nl_gain_fact, decrement, time, grav, tig, thtvgx, vbeg,
                   sbii, philimx):
    tfg = mat2tr(psifgx * RAD, thtfgx * RAD)
    swii = cadine(wp_lonx * RAD, wp_latx * RAD, wp_alt, time)
    swbg = tig.T @ (swii - sbii)
    polar = polar_from_cart(swbg)
    wp_sltrange = float(polar[0])
    tog = mat2tr(float(polar[1]), float(polar[2]))
    wp_grdrange = hypot(float(swbg[0]), float(swbg[1]))
    vbeo = tog @ vbeg
    vbef = tfg @ vbeg
    nl_gain = nl_gain_fact * (1 - exp(-wp_sltrange / decrement))
    algv = np.array([
        grav * sin(thtvgx * RAD),
        line_gain * (-vbeo[1] + nl_gain * vbef[1]),
        line_gain * (-vbeo[2] + nl_gain * vbef[2]) - grav * cos(thtvgx * RAD),
    ])
    dvbe = sqrt(float(vbeg @ vbeg))
    rad_min = dvbe * dvbe / (grav * tan(philimx * RAD))
    if wp_grdrange < 2 * rad_min:
        sh = np.array([swbg[0], swbg[1], 0.0])
        vh = np.array([vbeg[0], vbeg[1], 0.0])
        wp_flag = _sign(float(vh @ sh))
    else:
        wp_flag = 0
    return algv, wp_flag, wp_grdrange, nl_gain


def _round3_state(lonx, latx, alt, psivgx, thtvgx, dvbe, time):
    tei = cadtei(time)
    tge = cadtge(lonx * RAD, latx * RAD)
    tig = tei.T @ tge.T
    sbii = cadine(lonx * RAD, latx * RAD, alt, time)
    vbeg = mat2tr(psivgx * RAD, thtvgx * RAD).T @ np.array([dvbe, 0.0, 0.0])
    return tig, sbii, vbeg


def _ready():
    vehicle = type("V", (), {"store": StateStore()})()
    guidance = Cruise5Guidance()
    guidance.define(vehicle)
    store = vehicle.store
    tig, sbii, vbeg = _round3_state(
        LONX, LATX, ALT, PSIVGX, THTVGX, DVBE, TIME
    )
    store.define(Field("time", TIME, "real", "exec", "environment"))
    store.define(Field("grav", GRAV, "real", "out", "environment"))
    store.define(Field("tig", tig, "mat", "init/out", "newton"))
    store.define(Field("thtvgx", THTVGX, "real", "init/out", "newton"))
    store.define(Field("vbeg", vbeg, "vec", "state", "newton"))
    store.define(Field("sbii", sbii, "vec", "state", "newton"))
    store.define(Field("philimx", PHILIMX, "real", "data", "control"))
    store.set("wp_lonx", WP_LONX)
    store.set("wp_latx", WP_LATX)
    store.set("wp_alt", WP_ALT)
    store.set("psifgx", PSIFGX)
    store.set("thtfgx", THTFGX)
    store.set("line_gain", LINE_GAIN)
    store.set("nl_gain_fact", NL_GAIN_FACT)
    store.set("decrement", DECREMENT)
    return vehicle, guidance


def test_name_is_guidance():
    assert Cruise5Guidance().name == "guidance"


def test_define_registers_def_guidance_fields():
    vehicle = type("V", (), {"store": StateStore()})()
    Cruise5Guidance().define(vehicle)
    store = vehicle.store
    zeros = np.zeros(3)
    for name, (ftype, role, outputs, default) in DEF_GUIDANCE.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "guidance"
        assert field.outputs == outputs
        if ftype == "vec":
            np.testing.assert_array_equal(store.get(name), zeros)
        else:
            assert store.get(name) == default
    assert type(store.get("mguidance")) is int
    assert type(store.get("wp_flag")) is int


def test_guidance_line_input1_waypoint1_matches_replica():
    vehicle, guidance = _ready()
    store = vehicle.store
    expected_algv, expected_flag, expected_range, _nl = _expected_line(
        store.get("wp_lonx"),
        store.get("wp_latx"),
        store.get("wp_alt"),
        store.get("psifgx"),
        store.get("thtfgx"),
        store.get("line_gain"),
        store.get("nl_gain_fact"),
        store.get("decrement"),
        store.get("time"),
        store.get("grav"),
        store.get("tig"),
        store.get("thtvgx"),
        store.get("vbeg"),
        store.get("sbii"),
        store.get("philimx"),
    )
    algv = guidance.guidance_line(vehicle)
    assert np.all(np.isfinite(algv))
    assert algv.shape == (3,)
    assert store.get("wp_grdrange") > 0
    assert store.get("wp_flag") in (-1, 0, 1)
    np.testing.assert_allclose(algv, expected_algv, rtol=RTOL, atol=ATOL)
    assert store.get("wp_flag") == expected_flag
    assert store.get("wp_grdrange") == pytest.approx(
        expected_range, rel=RTOL, abs=ATOL
    )
