import math

import numpy as np
import pytest

from cadac.constants import RAD, REARTH
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.earth import cadine, cadtei, cadtge
from cadac.math.frames import mat2tr, polar_from_cart
from cadac.vehicles.hyper5.guidance import Hyper5Guidance

# Demo 4.6 Point guidance against WSMR coordinates
LONX = -106.3
LATX = 32.0
ALT = 20000.0
PSIVGX = 0.0
THTVGX = 0.0
DVBE = 1181.0
WP_LONX = -106.28
WP_LATX = 33.4
WP_ALT = 1200.0
POINT_GAIN = 0.04
PHILIMX = 70.0
GRAV = 9.81
TIME = 0.0

RTOL = 1e-12
ATOL = 1e-14

POINT_FIELDS = {
    "mguidance": ("int", "data", ("scrn",)),
    "wp_lonx": ("real", "data", ()),
    "wp_latx": ("real", "data", ()),
    "wp_alt": ("real", "data", ()),
    "point_gain": ("real", "data", ()),
    "wp_sltrange": ("real", "diag", ("scrn", "plot")),
    "VBEO": ("vec", "diag", ()),
    "wp_grdrange": ("real", "diag", ("scrn", "plot")),
    "SWBG": ("vec", "out", ()),
    "rad_min": ("real", "diag", ()),
    "wp_flag": ("int", "diag", ()),
}

NOT_DEFINED = (
    "pronav_gain",
    "line_gain",
    "nl_gain_fact",
    "decrement",
    "psifgx",
    "thtfgx",
    "nl_gain",
    "VBEF",
    "bias",
    "philimx",
    "write",
    "SWBL",
    "swel1",
    "time",
    "grav",
    "tig",
    "thtvgx",
    "vbeg",
    "sbii",
    "TIG",
    "VBEG",
    "SBII",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.05,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _cadac_sign(variable):
    if variable < 0:
        return -1
    return 1


def _round3_kinematics(lonx, latx, alt, psivgx, thtvgx, dvbe, time):
    sbig = np.array([0.0, 0.0, -(alt + REARTH)])
    tge = cadtge(lonx * RAD, latx * RAD)
    teg = tge.T
    sbie = teg @ sbig
    tei = cadtei(time)
    sbii = tei.T @ sbie
    psivg = psivgx * RAD
    thtvg = thtvgx * RAD
    vbeg = np.array(
        [
            dvbe * np.cos(thtvg) * np.cos(psivg),
            dvbe * np.cos(thtvg) * np.sin(psivg),
            dvbe * (-np.sin(thtvg)),
        ]
    )
    tig = tei.T @ teg
    return tig, sbii, vbeg


def _expected_point(
    wp_lonx, wp_latx, wp_alt, point_gain, time, grav, tig, thtvgx, vbeg, sbii, philimx
):
    swii = cadine(wp_lonx * RAD, wp_latx * RAD, wp_alt, time)
    swbg = tig.T @ (swii - sbii)
    polar = polar_from_cart(swbg)
    wp_sltrange = polar[0]
    psiog = polar[1]
    thtog = polar[2]
    tog = mat2tr(psiog, thtog)
    wp_grdrange = math.hypot(float(swbg[0]), float(swbg[1]))
    vbeo = tog @ vbeg
    apgv1 = grav * math.sin(thtvgx * RAD)
    apgv2 = point_gain * (-vbeo[1])
    apgv3 = point_gain * (-vbeo[2]) - grav * math.cos(thtvgx * RAD)
    apgv = np.array([apgv1, apgv2, apgv3])
    dvbe = math.sqrt(float(vbeg[0] ** 2 + vbeg[1] ** 2 + vbeg[2] ** 2))
    rad_min = dvbe * dvbe / (grav * math.tan(philimx * RAD))
    if wp_grdrange < 2 * rad_min:
        sh = np.array([swbg[0], swbg[1], 0.0])
        vh = np.array([vbeg[0], vbeg[1], 0.0])
        wp_flag = _cadac_sign(float(vh @ sh))
    else:
        wp_flag = 0
    return apgv, {
        "wp_sltrange": wp_sltrange,
        "VBEO": vbeo,
        "wp_grdrange": wp_grdrange,
        "SWBG": swbg,
        "rad_min": rad_min,
        "wp_flag": wp_flag,
    }


def _ready(
    wp_lonx=WP_LONX,
    wp_latx=WP_LATX,
    wp_alt=WP_ALT,
    point_gain=POINT_GAIN,
    lonx=LONX,
    latx=LATX,
    alt=ALT,
    psivgx=PSIVGX,
    thtvgx=THTVGX,
    dvbe=DVBE,
    time=TIME,
    grav=GRAV,
    philimx=PHILIMX,
    tig=None,
    sbii=None,
    vbeg=None,
    mguidance=44,
    wp_flag=0,
):
    vehicle = _Vehicle()
    guidance = Hyper5Guidance()
    guidance.define(vehicle)
    store = vehicle.store
    if tig is None or sbii is None or vbeg is None:
        tig, sbii, vbeg = _round3_kinematics(
            lonx, latx, alt, psivgx, thtvgx, dvbe, time
        )
    store.define(Field("time", time, "real", "exec", "environment"))
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("tig", tig, "mat", "init/out", "newton"))
    store.define(Field("thtvgx", thtvgx, "real", "init/out", "newton"))
    store.define(Field("vbeg", vbeg, "vec", "state", "newton"))
    store.define(Field("sbii", sbii, "vec", "state", "newton"))
    store.define(Field("philimx", philimx, "real", "data", "control"))
    store.set("mguidance", mguidance)
    store.set("wp_lonx", wp_lonx)
    store.set("wp_latx", wp_latx)
    store.set("wp_alt", wp_alt)
    store.set("point_gain", point_gain)
    store.set("wp_flag", wp_flag)
    return vehicle, guidance


def test_name_is_guidance():
    assert Hyper5Guidance().name == "guidance"


def test_define_registers_point_and_dispatcher_fields():
    vehicle = _Vehicle()
    Hyper5Guidance().define(vehicle)
    store = vehicle.store
    for name, (ftype, role, outputs) in POINT_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "guidance"
        assert field.outputs == outputs
    assert store.get("mguidance") == 0
    assert type(store.get("mguidance")) is int
    assert store.get("wp_lonx") == 0.0
    assert store.get("wp_latx") == 0.0
    assert store.get("wp_alt") == 0.0
    assert store.get("point_gain") == 0.0
    assert store.get("wp_sltrange") == 999999.0
    np.testing.assert_array_equal(store.get("VBEO"), np.zeros(3))
    assert store.get("wp_grdrange") == 999999.0
    np.testing.assert_array_equal(store.get("SWBG"), np.zeros(3))
    assert store.get("rad_min") == 0.0
    assert store.get("wp_flag") == 0
    assert type(store.get("wp_flag")) is int


def test_define_does_not_register_pronav_line_arc_philimx_or_round3():
    vehicle = _Vehicle()
    Hyper5Guidance().define(vehicle)
    store = vehicle.store
    for name in NOT_DEFINED:
        with pytest.raises(KeyError):
            store.get(name)


def test_execute_is_pass():
    vehicle, guidance = _ready()
    store = vehicle.store
    store.set("wp_sltrange", 1.0)
    store.set("wp_grdrange", 2.0)
    store.set("rad_min", 3.0)
    store.set("wp_flag", 5)
    store.set("VBEO", np.array([1.0, 2.0, 3.0]))
    store.set("SWBG", np.array([4.0, 5.0, 6.0]))
    guidance.execute(vehicle, _ctx())
    assert store.get("wp_sltrange") == 1.0
    assert store.get("wp_grdrange") == 2.0
    assert store.get("rad_min") == 3.0
    assert store.get("wp_flag") == 5
    np.testing.assert_array_equal(store.get("VBEO"), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(store.get("SWBG"), np.array([4.0, 5.0, 6.0]))
    assert store.get("mguidance") == 44


def test_guidance_point_demo_46_waypoint_offset_finite_apgv():
    vehicle, guidance = _ready()
    store = vehicle.store
    assert store.get("wp_lonx") != LONX
    assert store.get("wp_latx") != LATX
    assert store.get("wp_alt") != ALT
    apgv = guidance.guidance_point(vehicle)
    assert np.all(np.isfinite(apgv))
    assert apgv.shape == (3,)
    assert not np.allclose(apgv, 0.0)


def test_wp_flag_zero_outside_two_rad_min():
    vehicle, guidance = _ready()
    store = vehicle.store
    _, expected = _expected_point(
        store.get("wp_lonx"),
        store.get("wp_latx"),
        store.get("wp_alt"),
        store.get("point_gain"),
        store.get("time"),
        store.get("grav"),
        store.get("tig"),
        store.get("thtvgx"),
        store.get("vbeg"),
        store.get("sbii"),
        store.get("philimx"),
    )
    assert expected["wp_grdrange"] >= 2 * expected["rad_min"]
    assert expected["wp_flag"] == 0
    guidance.guidance_point(vehicle)
    assert store.get("wp_flag") == 0


def test_wp_flag_plus_one_when_closing_inside_two_rad_min():
    vehicle, guidance = _ready(wp_lonx=LONX, wp_latx=LATX + 0.05, wp_alt=ALT)
    store = vehicle.store
    _, expected = _expected_point(
        store.get("wp_lonx"),
        store.get("wp_latx"),
        store.get("wp_alt"),
        store.get("point_gain"),
        store.get("time"),
        store.get("grav"),
        store.get("tig"),
        store.get("thtvgx"),
        store.get("vbeg"),
        store.get("sbii"),
        store.get("philimx"),
    )
    assert expected["wp_grdrange"] < 2 * expected["rad_min"]
    assert expected["wp_flag"] == 1
    guidance.guidance_point(vehicle)
    assert store.get("wp_flag") == 1


def test_wp_flag_minus_one_when_fleeting_inside_two_rad_min():
    vehicle, guidance = _ready(wp_lonx=LONX, wp_latx=LATX - 0.05, wp_alt=ALT)
    store = vehicle.store
    _, expected = _expected_point(
        store.get("wp_lonx"),
        store.get("wp_latx"),
        store.get("wp_alt"),
        store.get("point_gain"),
        store.get("time"),
        store.get("grav"),
        store.get("tig"),
        store.get("thtvgx"),
        store.get("vbeg"),
        store.get("sbii"),
        store.get("philimx"),
    )
    assert expected["wp_grdrange"] < 2 * expected["rad_min"]
    assert expected["wp_flag"] == -1
    guidance.guidance_point(vehicle)
    assert store.get("wp_flag") == -1


def test_cadac_sign_zero_horizontal_dot_is_plus_one():
    tig = np.eye(3)
    sbii = cadine(0.0, 0.0, 0.0, 0.0)
    vbeg = np.array([0.0, DVBE, 0.0])
    vehicle, guidance = _ready(
        wp_lonx=0.0,
        wp_latx=0.0,
        wp_alt=100.0,
        tig=tig,
        sbii=sbii,
        vbeg=vbeg,
        thtvgx=0.0,
        time=0.0,
    )
    store = vehicle.store
    _, expected = _expected_point(
        store.get("wp_lonx"),
        store.get("wp_latx"),
        store.get("wp_alt"),
        store.get("point_gain"),
        store.get("time"),
        store.get("grav"),
        store.get("tig"),
        store.get("thtvgx"),
        store.get("vbeg"),
        store.get("sbii"),
        store.get("philimx"),
    )
    assert expected["wp_grdrange"] < 2 * expected["rad_min"]
    sh = np.array([expected["SWBG"][0], expected["SWBG"][1], 0.0])
    vh = np.array([vbeg[0], vbeg[1], 0.0])
    assert abs(float(vh @ sh)) < 1e-8
    assert expected["wp_flag"] == 1
    guidance.guidance_point(vehicle)
    assert store.get("wp_flag") == 1


def test_guidance_point_one_step_matches_cpp_equations():
    vehicle, guidance = _ready()
    store = vehicle.store
    expected_apgv, expected = _expected_point(
        store.get("wp_lonx"),
        store.get("wp_latx"),
        store.get("wp_alt"),
        store.get("point_gain"),
        store.get("time"),
        store.get("grav"),
        store.get("tig"),
        store.get("thtvgx"),
        store.get("vbeg"),
        store.get("sbii"),
        store.get("philimx"),
    )
    got = guidance.guidance_point(vehicle)
    np.testing.assert_allclose(got, expected_apgv, rtol=RTOL, atol=ATOL)
    assert _approx(store.get("wp_sltrange"), expected["wp_sltrange"])
    np.testing.assert_allclose(store.get("VBEO"), expected["VBEO"], rtol=RTOL, atol=ATOL)
    assert _approx(store.get("wp_grdrange"), expected["wp_grdrange"])
    np.testing.assert_allclose(store.get("SWBG"), expected["SWBG"], rtol=RTOL, atol=ATOL)
    assert _approx(store.get("rad_min"), expected["rad_min"])
    assert store.get("wp_flag") == expected["wp_flag"]


def test_guidance_point_does_not_write_uppercase_round3_aliases():
    vehicle, guidance = _ready()
    guidance.guidance_point(vehicle)
    store = vehicle.store
    for name in ("TIG", "VBEG", "SBII"):
        with pytest.raises(KeyError):
            store.get(name)
