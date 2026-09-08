import numpy as np
from cadac.constants import PI, RAD, REARTH
from cadac.math.earth import cadtei, cadtge, cadsph


def test_cadtei_t0_identity():
    T = cadtei(0)
    np.testing.assert_allclose(T, np.eye(3), atol=1e-14)


def test_cadsph_equator_prime_meridian():
    lon, lat, alt = cadsph([REARTH, 0, 0])
    np.testing.assert_allclose([lon, lat, alt], [0.0, 0.0, 0.0], atol=1e-14)


def test_cadsph_quadrant_iii_uses_180_rad_not_pi():
    # HYPER5 Demo 4.7 lonx=-106.28 is QIII (x<0,y<0). C++ cadsph uses
    # 180*RAD / 360*RAD, not PI=3.1415927 / 2*PI (delta ~4.64e-8 rad).
    lonx, latx, alt = -106.28, 33.35, 2400.0
    lon = lonx * RAD
    lat = latx * RAD
    radius = alt + REARTH
    sbie = np.array(
        [
            radius * np.cos(lat) * np.cos(lon),
            radius * np.cos(lat) * np.sin(lon),
            radius * np.sin(lat),
        ]
    )
    assert sbie[0] < 0.0 and sbie[1] < 0.0
    got_lon, _, _ = cadsph(sbie)
    x, y, _ = sbie
    dum4 = np.arcsin(y / np.sqrt(x * x + y * y))
    alamda = 180 * RAD - dum4
    want_lon = alamda
    if want_lon > 180 * RAD:
        want_lon = -(360 * RAD - want_lon)
    np.testing.assert_allclose(got_lon, want_lon, rtol=0.0, atol=1e-15)
    pi_lon = PI - dum4
    if pi_lon > PI:
        pi_lon = -(2 * PI - pi_lon)
    assert abs(got_lon - pi_lon) > 1e-10


def test_cadtge_lon0_lat0():
    T = cadtge(0.0, 0.0)
    expected = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(T, expected, atol=1e-14)
