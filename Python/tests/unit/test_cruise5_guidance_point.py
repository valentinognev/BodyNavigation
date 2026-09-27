from math import cos, hypot, sin, sqrt, tan

import numpy as np
import pytest

from cadac.constants import RAD
from cadac.kernel.state import Field, StateStore
from cadac.math.earth import cadine, cadtei, cadtge
from cadac.math.frames import mat2tr, polar_from_cart
from cadac.vehicles.round3.cruise5.guidance import Cruise5Guidance

RTOL = 1e-12
ATOL = 1e-14

POINT_GAIN = 1.0
WP_LONX = 15.4
WP_LATX = 35.3
WP_ALT = 100.0
LONX = 14.7
LATX = 35.4
ALT = 7000.0
PSIVGX = 90.0
THTVGX = 0.0
DVBE = 200.0
PHILIMX = 70.0
TIME = 0.0
GRAV = 9.81


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _expected_point(wp_lonx, wp_latx, wp_alt, point_gain, time, grav, tig,
                    thtvgx, vbeg, sbii, philimx):
    swii = cadine(wp_lonx * RAD, wp_latx * RAD, wp_alt, time)
    swbg = tig.T @ (swii - sbii)
    polar = polar_from_cart(swbg)
    tog = mat2tr(float(polar[1]), float(polar[2]))
    vbeo = tog @ vbeg
    apgv = np.array([
        grav * sin(thtvgx * RAD),
        point_gain * (-vbeo[1]),
        point_gain * (-vbeo[2]) - grav * cos(thtvgx * RAD),
    ])
    wp_grdrange = hypot(float(swbg[0]), float(swbg[1]))
    dvbe = sqrt(float(vbeg @ vbeg))
    rad_min = dvbe * dvbe / (grav * tan(philimx * RAD))
    if wp_grdrange < 2 * rad_min:
        sh = np.array([swbg[0], swbg[1], 0.0])
        vh = np.array([vbeg[0], vbeg[1], 0.0])
        wp_flag = _sign(float(vh @ sh))
    else:
        wp_flag = 0
    return apgv, wp_flag, wp_grdrange


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
    store.set("point_gain", POINT_GAIN)
    return vehicle, guidance


def test_guidance_point_event3_tank_matches_replica():
    vehicle, guidance = _ready()
    store = vehicle.store
    expected_apgv, expected_flag, expected_range = _expected_point(
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
    swii = cadine(WP_LONX * RAD, WP_LATX * RAD, WP_ALT, TIME)
    swbg = store.get("tig").T @ (swii - store.get("sbii"))
    polar = polar_from_cart(swbg)
    tog = mat2tr(float(polar[1]), float(polar[2]))
    vbeo = tog @ store.get("vbeg")
    dvbe = sqrt(float(store.get("vbeg") @ store.get("vbeg")))
    rad_min = dvbe * dvbe / (GRAV * tan(PHILIMX * RAD))

    apgv = guidance.guidance_point(vehicle)
    assert np.all(np.isfinite(apgv))
    assert apgv.shape == (3,)
    assert store.get("wp_grdrange") > 0
    assert store.get("wp_flag") in (-1, 0, 1)
    np.testing.assert_allclose(apgv, expected_apgv, rtol=RTOL, atol=ATOL)
    assert store.get("wp_flag") == expected_flag
    assert store.get("wp_grdrange") == pytest.approx(
        expected_range, rel=RTOL, abs=ATOL
    )
    assert store.get("wp_sltrange") == pytest.approx(
        float(polar[0]), rel=RTOL, abs=ATOL
    )
    np.testing.assert_allclose(store.get("VBEO"), vbeo, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SWBG"), swbg, rtol=RTOL, atol=ATOL)
    assert store.get("rad_min") == pytest.approx(rad_min, rel=RTOL, abs=ATOL)


def test_guidance_point_does_not_write_loa_fields():
    vehicle, guidance = _ready()
    store = vehicle.store
    store.set("nl_gain", 3.14)
    store.set("VBEF", np.array([1.0, 2.0, 3.0]))
    guidance.guidance_point(vehicle)
    assert store.get("nl_gain") == 3.14
    np.testing.assert_array_equal(store.get("VBEF"), np.array([1.0, 2.0, 3.0]))
