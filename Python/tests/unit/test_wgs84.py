import numpy as np
from cadac.math.wgs84 import cad_geo84_in, cad_grav84, cad_in_geo84, cad_tdi84, cad_tgi84


def test_roundtrip_geo84_equator():
    sbii = cad_in_geo84(0.0, 0.0, 0.0, 0.0)
    lon, lat, alt = cad_geo84_in(sbii, 0.0)
    np.testing.assert_allclose([lon, lat, alt], [0.0, 0.0, 0.0], atol=1e-6)


def test_grav84_finite_at_surface():
    sbii = cad_in_geo84(0.0, 0.0, 0.0, 0.0)
    g = cad_grav84(sbii, 0.0)
    assert g.shape == (3,)
    assert np.linalg.norm(g) > 9.0


def test_tdi84_equator_assign_loc():
    # C++ cad_tdi84(0,0,0,0): assign_loc(0, 2, tdi13) with tdi13 = cos(lat) = 1
    tdi = cad_tdi84(0.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(tdi[0, 2], 1.0, rtol=1e-12)


def test_tgi84_equator_assign_loc():
    # C++ cad_tgi84(0,0,0,0): TGD identity (dd=0), TGI = TGD*TDI, TDI.assign_loc(2, 0, -1)
    tgi = cad_tgi84(0.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(tgi[2, 0], -1.0, rtol=1e-12)
    assert tgi.shape == (3, 3)
