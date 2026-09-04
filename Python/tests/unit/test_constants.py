from cadac.constants import REARTH, WEII3, RAD, DEG, AGRAV, G, EARTH_MASS, R, PI, EPS

def test_cadac_constants():
    assert REARTH == 6370987.308
    assert WEII3 == 7.292115e-5
    assert RAD == 0.0174532925199432
    assert DEG == 57.2957795130823
    assert AGRAV == 9.80675445
    assert G == 6.673e-11
    assert EARTH_MASS == 5.973e24
    assert R == 287.053
    assert PI == 3.1415927
    assert EPS == 1e-10
