import numpy as np

from cadac.constants import EPS, PI


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


def cadac_matmul(a, b):
    """C++ `Matrix::operator*` (row-major ijk, no BLAS/FMA)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    squeeze = False
    if b.ndim == 1:
        b = b.reshape(-1, 1)
        squeeze = True
    nrow, nmid = a.shape
    ncol = b.shape[1]
    result = np.zeros((nrow, ncol), dtype=float)
    for i in range(nrow * ncol):
        r = i // ncol
        c = i % ncol
        acc = 0.0
        for k in range(nmid):
            acc += a[r, k] * b[k, c]
        result[r, c] = acc
    if squeeze:
        return result.reshape(nrow)
    return result


def _cadac_sub_matrix(amat, row, col):
    """C++ `Matrix::sub_matrix` (1-based row/col omitted)."""
    amat = np.asarray(amat, dtype=float)
    n = amat.shape[0]
    result = np.zeros((n - 1, n - 1), dtype=float)
    skip_start = (row - 1) * n
    skip_end = skip_start + n
    j = 0
    num_elem = n * n
    for i in range(num_elem):
        if i < skip_start or i >= skip_end:
            offset_col = (col - 1) + (i // n) * n
            if i != offset_col:
                result.flat[j] = amat.flat[i]
                j += 1
    return result


def cadac_determinant(amat):
    """C++ `Matrix::determinant` (first-row cofactor expansion)."""
    amat = np.asarray(amat, dtype=float)
    n = amat.shape[0]
    if n == 1:
        return float(amat.flat[0])
    if n == 2:
        return float(amat[0, 0] * amat[1, 1] - amat[0, 1] * amat[1, 0])
    result = 0.0
    for j in range(n):
        cof = cadac_determinant(_cadac_sub_matrix(amat, 1, j + 1))
        if (j % 2) == 0:
            result += cof * amat.flat[j]
        else:
            result += (-1.0) * cof * amat.flat[j]
    return result


def cadac_adjoint(amat):
    """C++ `Matrix::adjoint` (cofactors then trans)."""
    amat = np.asarray(amat, dtype=float)
    n = amat.shape[0]
    result = np.zeros((n, n), dtype=float)
    for i in range(n * n):
        row = i // n + 1
        col = i % n + 1
        det = cadac_determinant(_cadac_sub_matrix(amat, row, col))
        if ((row + col) % 2) == 0:
            result.flat[i] = det
        else:
            result.flat[i] = -1.0 * det
    return result.T.copy()


def cadac_inverse(amat):
    """C++ `Matrix::inverse` = (1/det)*adjoint (not LAPACK)."""
    amat = np.asarray(amat, dtype=float)
    d = cadac_determinant(amat)
    if d == 0.0:
        raise ValueError("singular! 'Matrix::inverse()'")
    d = 1.0 / d
    return cadac_adjoint(amat) * d


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


def mat3tr(psi, tht, phi):
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


def cadtbv(phi, alpha):
    amat = np.zeros((3, 3))
    salpha = np.sin(alpha)
    calpha = np.cos(alpha)
    sphi = np.sin(phi)
    cphi = np.cos(phi)
    amat[0, 0] = calpha
    amat[0, 1] = sphi * salpha
    amat[0, 2] = -cphi * salpha
    amat[1, 1] = cphi
    amat[1, 2] = sphi
    amat[2, 0] = salpha
    amat[2, 1] = -sphi * calpha
    amat[2, 2] = cphi * calpha
    return amat


def cart_from_pol(magnitude, azimuth, elevation):
    vec = np.zeros(3)
    vec[0] = magnitude * (np.cos(elevation) * np.cos(azimuth))
    vec[1] = magnitude * (np.cos(elevation) * np.sin(azimuth))
    vec[2] = magnitude * (np.sin(elevation) * (-1.0))
    return vec


def angle(vec1, vec2):
    scalar = vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]
    abs1 = np.sqrt(vec1[0] * vec1[0] + vec1[1] * vec1[1] + vec1[2] * vec1[2])
    abs2 = np.sqrt(vec2[0] * vec2[0] + vec2[1] * vec2[1] + vec2[2] * vec2[2])
    dum = abs1 * abs2
    if abs1 * abs2 > EPS:
        argument = scalar / dum
    else:
        argument = 1.0
    if argument > 1.0:
        argument = 1.0
    if argument < -1.0:
        argument = -1.0
    return np.arccos(argument)
