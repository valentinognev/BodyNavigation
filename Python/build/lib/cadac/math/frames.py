import math

import numpy as np

from cadac.constants import EPS, PI


def cadac_sign(variable: float) -> int:
    if variable < 0.0:
        return -1
    return 1


def skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def hypot3(vec: np.ndarray) -> float:
    x, y, z = vec
    return float(np.sqrt(x * x + y * y + z * z))


def quat_to_dcm(q0: float, q1: float, q2: float, q3: float) -> np.ndarray:
    tbl = np.zeros((3, 3))
    tbl[0, 0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    tbl[0, 1] = 2.0 * (q1 * q2 + q0 * q3)
    tbl[0, 2] = 2.0 * (q1 * q3 - q0 * q2)
    tbl[1, 0] = 2.0 * (q1 * q2 - q0 * q3)
    tbl[1, 1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
    tbl[1, 2] = 2.0 * (q2 * q3 + q0 * q1)
    tbl[2, 0] = 2.0 * (q1 * q3 + q0 * q2)
    tbl[2, 1] = 2.0 * (q2 * q3 - q0 * q1)
    tbl[2, 2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3
    return tbl


def matvec3(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return np.array(
        [np.dot(mat[0], vec), np.dot(mat[1], vec), np.dot(mat[2], vec)],
        dtype=float,
    )


def incidence_angles(
    vbab: np.ndarray, dvba: float
) -> tuple[float, float, float, float]:
    """Aerodynamic incidence (alpha, beta, alpp, phip) from body airspeed.

    CADAC ``Flat6::kinematics`` branches: ``fabs(dum)>1`` clamp,
    ``vbab2==0 and vbab3==0``, ``fabs(vbab2)<EPS`` then ``vbab3>0`` → 0,
    ``vbab3<0`` → ``PI``, else ``atan2(vbab2, vbab3)``.
    """
    vbab1 = float(vbab[0])
    vbab2 = float(vbab[1])
    vbab3 = float(vbab[2])
    alpha = math.atan2(vbab3, vbab1)
    beta = math.asin(vbab2 / dvba)
    dum = vbab1 / dvba
    if math.fabs(dum) > 1.0:
        dum = 1.0 * cadac_sign(dum)
    alpp = math.acos(dum)
    if vbab2 == 0.0 and vbab3 == 0.0:
        phip = 0.0
    elif math.fabs(vbab2) < EPS:
        if vbab3 > 0.0:
            phip = 0.0
        elif vbab3 < 0.0:
            phip = PI
        else:
            phip = 0.0
    else:
        phip = math.atan2(vbab2, vbab3)
    return alpha, beta, alpp, phip


def polar_from_cart(v: np.ndarray) -> np.ndarray:
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


def cadac_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """C++ `Matrix::operator*` (row-major ijk, no BLAS/FMA). 3×3 unrolled, same ijk as C++ `Matrix::operator*`."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    squeeze = False
    if b.ndim == 1:
        b = b.reshape(-1, 1)
        squeeze = True
    if a.shape == (3, 3) and b.shape == (3, 3):
        result = np.zeros((3, 3), dtype=float)
        result[0, 0] = a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] + a[0, 2] * b[2, 0]
        result[0, 1] = a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1] + a[0, 2] * b[2, 1]
        result[0, 2] = a[0, 0] * b[0, 2] + a[0, 1] * b[1, 2] + a[0, 2] * b[2, 2]
        result[1, 0] = a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0] + a[1, 2] * b[2, 0]
        result[1, 1] = a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1] + a[1, 2] * b[2, 1]
        result[1, 2] = a[1, 0] * b[0, 2] + a[1, 1] * b[1, 2] + a[1, 2] * b[2, 2]
        result[2, 0] = a[2, 0] * b[0, 0] + a[2, 1] * b[1, 0] + a[2, 2] * b[2, 0]
        result[2, 1] = a[2, 0] * b[0, 1] + a[2, 1] * b[1, 1] + a[2, 2] * b[2, 1]
        result[2, 2] = a[2, 0] * b[0, 2] + a[2, 1] * b[1, 2] + a[2, 2] * b[2, 2]
        return result
    if a.shape == (3, 3) and b.shape == (3, 1):
        result = np.zeros((3, 1), dtype=float)
        result[0, 0] = a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] + a[0, 2] * b[2, 0]
        result[1, 0] = a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0] + a[1, 2] * b[2, 0]
        result[2, 0] = a[2, 0] * b[0, 0] + a[2, 1] * b[1, 0] + a[2, 2] * b[2, 0]
        if squeeze:
            return result.reshape(3)
        return result
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


def _cadac_sub_matrix(amat: np.ndarray, row: int, col: int) -> np.ndarray:
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


def cadac_determinant(amat: np.ndarray) -> float:
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


def cadac_adjoint(amat: np.ndarray) -> np.ndarray:
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


def cadac_inverse(amat: np.ndarray) -> np.ndarray:
    """C++ `Matrix::inverse` = (1/det)*adjoint (not LAPACK)."""
    amat = np.asarray(amat, dtype=float)
    d = cadac_determinant(amat)
    if d == 0.0:
        raise ValueError("singular! 'Matrix::inverse()'")
    d = 1.0 / d
    return cadac_adjoint(amat) * d


def mat2tr(psivg: float, thtvg: float) -> np.ndarray:
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


def mat3tr(psi: float, tht: float, phi: float) -> np.ndarray:
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


def cadtbv(phi: float, alpha: float) -> np.ndarray:
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


def cart_from_pol(magnitude: float, azimuth: float, elevation: float) -> np.ndarray:
    vec = np.zeros(3)
    vec[0] = magnitude * (np.cos(elevation) * np.cos(azimuth))
    vec[1] = magnitude * (np.cos(elevation) * np.sin(azimuth))
    vec[2] = magnitude * (np.sin(elevation) * (-1.0))
    return vec


def angle(vec1: np.ndarray, vec2: np.ndarray) -> float:
    scalar = vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]
    abs1 = np.sqrt(vec1[0] * vec1[0] + vec1[1] * vec1[1] + vec1[2] * vec1[2])
    abs2 = np.sqrt(vec2[0] * vec2[0] + vec2[1] * vec2[1] + vec2[2] * vec2[2])
    dum = abs1 * abs2
    if abs1 * abs2 > EPS:
        argument = scalar / dum
    else:
        argument = 1.0
    if argument > 1.0:  # noqa: PLR1730
        argument = 1.0
    if argument < -1.0:  # noqa: PLR1730
        argument = -1.0
    return np.arccos(argument)
