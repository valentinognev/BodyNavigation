import numpy as np
from cadac.math.frames import polar_from_cart, mat2tr

def test_polar_east():
    d, az, el = polar_from_cart(np.array([0.0, 250.0, 0.0]))
    np.testing.assert_allclose([d, az, el], [250.0, np.arctan2(250.0, 0.0), 0.0], atol=1e-14)

def test_mat2tr_level_north():
    T = mat2tr(0.0, 0.0)
    np.testing.assert_allclose(T, np.eye(3), atol=1e-14)
