import numpy as np
from cadac.constants import REARTH, WEII3
from cadac.math.earth import cadine


def test_cadine_equator_t0():
    sbii = cadine(0.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(sbii, [REARTH, 0.0, 0.0], atol=1e-9)

def test_cadine_celestial_longitude():
    lon, lat, alt, time = 0.1, 0.2, 1000.0, 10.0
    rad = alt + REARTH
    cel = lon + WEII3 * time
    expected = np.array([
        rad * np.cos(lat) * np.cos(cel),
        rad * np.cos(lat) * np.sin(cel),
        rad * np.sin(lat),
    ])
    np.testing.assert_allclose(cadine(lon, lat, alt, time), expected, rtol=1e-12, atol=1e-14)
