import numpy as np
from cadac.math.frames import (
    angle,
    cadac_sign,
    cart_from_pol,
    hypot3,
    mat2tr,
    matvec3,
    polar_from_cart,
    quat_to_dcm,
    skew,
)

_RTOL = 1e-12
_ATOL = 1e-14

def test_polar_east():
    d, az, el = polar_from_cart(np.array([0.0, 250.0, 0.0]))
    np.testing.assert_allclose([d, az, el], [250.0, np.arctan2(250.0, 0.0), 0.0], atol=1e-14)

def test_mat2tr_level_north():
    T = mat2tr(0.0, 0.0)
    np.testing.assert_allclose(T, np.eye(3), atol=1e-14)


def test_cart_from_pol_unit_x():
    np.testing.assert_allclose(
        cart_from_pol(1.0, 0.0, 0.0), [1.0, 0.0, 0.0], rtol=_RTOL, atol=_ATOL
    )


def test_angle_orthogonal_xy():
    theta = angle(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(theta, np.pi / 2, rtol=_RTOL, atol=_ATOL)


def test_cadac_sign_matches_cpp():
    assert cadac_sign(-1.0) == -1
    assert cadac_sign(0.0) == 1
    assert cadac_sign(2.5) == 1


def test_skew_cross_product_matrix():
    k = skew(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(
        k,
        [[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]],
        rtol=1e-12,
        atol=1e-14,
    )


def test_hypot3_zero_and_3_4_12():
    assert hypot3(np.array([0.0, 0.0, 0.0])) == 0.0
    np.testing.assert_allclose(hypot3(np.array([3.0, 4.0, 12.0])), 13.0, atol=1e-14)


def test_quat_to_dcm_identity():
    dcm = quat_to_dcm(1.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(dcm, np.eye(3), atol=1e-14)


def test_quat_to_dcm_equal_components():
    dcm = quat_to_dcm(0.5, 0.5, 0.5, 0.5)
    np.testing.assert_allclose(
        dcm,
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        rtol=1e-12,
        atol=1e-14,
    )


def test_matvec3_three_dots():
    mat = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=float,
    )
    vec = np.array([1.0, 0.0, -1.0], dtype=float)
    np.testing.assert_allclose(
        matvec3(mat, vec),
        [-2.0, -2.0, -2.0],
        rtol=1e-12,
        atol=1e-14,
    )
