import numpy as np
from cadac.kernel.integrate import integrate

def test_scalar():
    assert integrate(3.0, 1.0, 10.0, 0.5) == 11.0

def test_vec():
    y = integrate(np.array([2.0, 0.0]), np.array([0.0, 2.0]), np.array([0.0, 0.0]), 1.0)
    np.testing.assert_allclose(y, [1.0, 1.0])
