import numpy as np

from cadac.constants import PI


def polar_from_cart(v):
    v1 = v[0]
    v2 = v[1]
    v3 = v[2]
    d = np.sqrt(v1 * v1 + v2 * v2 + v3 * v3)
    azimuth = np.arctan2(v2, v1)
    denom = np.sqrt(v1 * v1 + v2 * v2)
    if denom > 0.0:
        elevation = np.arctan2(-v3, denom)
    else:
        elevation = 0.0
        if v3 > 0:
            elevation = -PI / 2.0
        if v3 < 0:
            elevation = PI / 2.0
        if v3 == 0:
            elevation = 0.0
    return np.array([d, azimuth, elevation])


def mat2tr(psivg, thtvg):
    amat = np.zeros((3, 3))
    amat[0, 2] = -np.sin(thtvg)
    amat[1, 0] = -np.sin(psivg)
    amat[1, 1] = np.cos(psivg)
    amat[2, 2] = np.cos(thtvg)
    amat[0, 0] = amat[2, 2] * amat[1, 1]
    amat[0, 1] = -amat[2, 2] * amat[1, 0]
    amat[2, 0] = -amat[0, 2] * amat[1, 1]
    amat[2, 1] = amat[0, 2] * amat[1, 0]
    amat[1, 2] = 0.0
    return amat
