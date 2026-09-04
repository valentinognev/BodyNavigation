from cadac.constants import G, EARTH_MASS, REARTH
from cadac.env.gravity import gravity


def test_at_altitude():
    alt = 3000.0
    assert gravity(alt) == G * EARTH_MASS / (REARTH + alt) ** 2
