import numpy as np
from cadac.constants import REARTH
from cadac.math.earth import cadtei, cadtge, cadsph


def test_cadtei_t0_identity():
    T = cadtei(0)
    np.testing.assert_allclose(T, np.eye(3), atol=1e-14)


def test_cadsph_equator_prime_meridian():
    lon, lat, alt = cadsph([REARTH, 0, 0])
    np.testing.assert_allclose([lon, lat, alt], [0.0, 0.0, 0.0], atol=1e-14)


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
