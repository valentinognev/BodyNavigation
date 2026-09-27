import numpy as np

from cadac.constants import RAD, REARTH, WEII3


def cadtei(sim_time: float) -> np.ndarray:
    tei = np.eye(3)
    xi = WEII3 * sim_time
    sxi = np.sin(xi)
    cxi = np.cos(xi)
    tei[0, 0] = cxi
    tei[0, 1] = sxi
    tei[1, 0] = -sxi
    tei[1, 1] = cxi
    return tei


def cadtge(lon_rad: float, lat_rad: float) -> np.ndarray:
    amat = np.zeros((3, 3))
    clon = np.cos(lon_rad)
    slon = np.sin(lon_rad)
    clat = np.cos(lat_rad)
    slat = np.sin(lat_rad)
    amat[0, 0] = -slat * clon
    amat[0, 1] = -slat * slon
    amat[0, 2] = clat
    amat[1, 0] = -slon
    amat[1, 1] = clon
    amat[1, 2] = 0.0
    amat[2, 0] = -clat * clon
    amat[2, 1] = -clat * slon
    amat[2, 2] = -slat
    return amat


def cadine(lon_rad: float, lat_rad: float, alt_m: float, time: float) -> np.ndarray:
    rad = alt_m + REARTH
    cel_lon = lon_rad + WEII3 * time
    clat = np.cos(lat_rad)
    slat = np.sin(lat_rad)
    clon = np.cos(cel_lon)
    slon = np.sin(cel_lon)
    return np.array([rad * clat * clon, rad * clat * slon, rad * slat])


def cadsph(sbie: np.ndarray) -> tuple[float, float, float]:
    x = sbie[0]
    y = sbie[1]
    z = sbie[2]
    dbi = np.sqrt(x * x + y * y + z * z)
    lat = np.arcsin(z / dbi)
    alt = dbi - REARTH
    dum4 = np.arcsin(y / np.sqrt(x * x + y * y))
    alamda = 0.0
    if (x >= 0) and (y >= 0):
        alamda = dum4
    if (x < 0) and (y >= 0):
        alamda = 180 * RAD - dum4
    if (x < 0) and (y < 0):
        alamda = 180 * RAD - dum4
    if (x >= 0) and (y < 0):
        alamda = 360 * RAD + dum4
    lon = alamda
    if lon > 180 * RAD:
        lon = -(360 * RAD - lon)
    return lon, lat, alt
