import numpy as np
from cadac.math.frames import mat3tr


def _expected_mat3tr(psi, tht, phi):
    amat = np.zeros((3, 3))
    spsi = np.sin(psi)
    cpsi = np.cos(psi)
    stht = np.sin(tht)
    ctht = np.cos(tht)
    sphi = np.sin(phi)
    cphi = np.cos(phi)
    amat[0, 0] = cpsi * ctht
    amat[1, 0] = cpsi * stht * sphi - spsi * cphi
    amat[2, 0] = cpsi * stht * cphi + spsi * sphi
    amat[0, 1] = spsi * ctht
    amat[1, 1] = spsi * stht * sphi + cpsi * cphi
    amat[2, 1] = spsi * stht * cphi - cpsi * sphi
    amat[0, 2] = -stht
    amat[1, 2] = ctht * sphi
    amat[2, 2] = ctht * cphi
    return amat


def test_zero_angles_identity():
    np.testing.assert_allclose(mat3tr(0.0, 0.0, 0.0), np.eye(3), atol=1e-14)


def test_roll_element_1_2_is_ctht_sphi():
    assert mat3tr(0, 0, 0.1)[1, 2] == np.cos(0) * np.sin(0.1)


def test_all_nine_elements_match_cpp_assign_loc():
    psi, tht, phi = 0.4, -0.2, 0.1
    np.testing.assert_array_equal(mat3tr(psi, tht, phi), _expected_mat3tr(psi, tht, phi))
