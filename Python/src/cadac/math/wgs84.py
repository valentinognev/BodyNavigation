import numpy as np

from cadac.constants import RAD, REARTH, WEII3

GM = 3.9860044e14
C20 = -4.8416685e-4
FLATTENING = 3.33528106e-3
SMAJOR_AXIS = 6378137
GW_CLONG = 0
SMALL = 1e-7


def cad_geo84_in(sbii, time):
    count = 0
    alamda = 0.0
    sbii1 = sbii[0]
    sbii2 = sbii[1]
    sbii3 = sbii[2]
    dbi = np.sqrt(sbii1 * sbii1 + sbii2 * sbii2 + sbii3 * sbii3)
    latg = np.arcsin(sbii3 / dbi)
    lat = latg
    while True:
        lat0 = lat
        r0 = SMAJOR_AXIS * (
            1.0
            - FLATTENING * (1.0 - np.cos(2.0 * lat0)) / 2.0
            + 5.0 * FLATTENING**2 * (1.0 - np.cos(4.0 * lat0)) / 16.0
        )
        alt = dbi - r0
        dd = FLATTENING * np.sin(2.0 * lat0) * (1.0 - FLATTENING / 2.0 - alt / r0)
        lat = latg + dd
        count += 1
        if count > 100:
            raise ValueError(
                "Geodetic latitude does not converge, 'cad_geo84_in()'"
            )
        if abs(lat - lat0) <= SMALL:
            break
    dum4 = np.arcsin(sbii2 / np.sqrt(sbii1 * sbii1 + sbii2 * sbii2))
    if (sbii1 >= 0.0) and (sbii2 >= 0.0):
        alamda = dum4
    if (sbii1 < 0.0) and (sbii2 >= 0.0):
        alamda = (180.0 * RAD) - dum4
    if (sbii1 < 0.0) and (sbii2 < 0.0):
        alamda = (180.0 * RAD) - dum4
    if (sbii1 > 0.0) and (sbii2 < 0.0):
        alamda = (360.0 * RAD) + dum4
    lon = alamda - WEII3 * time - GW_CLONG
    if lon > (180.0 * RAD):
        lon = -((360.0 * RAD) - lon)
    return lon, lat, alt


def cad_geoc_in(sbii, time):
    lon_cel = 0.0
    sbii1 = sbii[0]
    sbii2 = sbii[1]
    sbii3 = sbii[2]
    dbi = np.sqrt(sbii1 * sbii1 + sbii2 * sbii2 + sbii3 * sbii3)
    latc = np.arcsin(sbii3 / dbi)
    altc = dbi - REARTH
    dum4 = np.arcsin(sbii2 / np.sqrt(sbii1 * sbii1 + sbii2 * sbii2))
    if (sbii1 >= 0.0) and (sbii2 >= 0.0):
        lon_cel = dum4
    if (sbii1 < 0.0) and (sbii2 >= 0.0):
        lon_cel = (180.0 * RAD) - dum4
    if (sbii1 < 0.0) and (sbii2 < 0.0):
        lon_cel = (180.0 * RAD) - dum4
    if (sbii1 > 0.0) and (sbii2 < 0.0):
        lon_cel = (360.0 * RAD) + dum4
    lonc = lon_cel - WEII3 * time - GW_CLONG
    if lonc > (180.0 * RAD):
        lonc = -((360.0 * RAD) - lonc)
    return lonc, latc, altc


def cad_grav84(sbii, time):
    lonc, latc, altc = cad_geoc_in(sbii, time)
    dbi = np.sqrt(sbii[0] * sbii[0] + sbii[1] * sbii[1] + sbii[2] * sbii[2])
    dum1 = GM / (dbi * dbi)
    dum2 = 3 * np.sqrt(5.0)
    dum3 = (SMAJOR_AXIS / dbi) ** 2
    gravg1 = -dum1 * dum2 * C20 * dum3 * np.sin(latc) * np.cos(latc)
    gravg2 = 0
    gravg3 = dum1 * (
        1.0 + dum2 / 2.0 * C20 * dum3 * (3.0 * np.sin(latc) ** 2 - 1.0)
    )
    gravg = np.zeros(3)
    gravg[0] = gravg1
    gravg[1] = gravg2
    gravg[2] = gravg3
    return gravg


def cad_in_geo84(lon, lat, alt, time):
    r0 = SMAJOR_AXIS * (
        1.0
        - FLATTENING * (1.0 - np.cos(2.0 * lat)) / 2.0
        + 5.0 * FLATTENING**2 * (1.0 - np.cos(4.0 * lat)) / 16.0
    )
    dd = FLATTENING * np.sin(2.0 * lat) * (1.0 - FLATTENING / 2.0 - alt / r0)
    dbi = r0 + alt
    sbid1 = -dbi * np.sin(dd)
    sbid3 = -dbi * np.cos(dd)
    lon_cel = GW_CLONG + WEII3 * time + lon
    slat = np.sin(lat)
    clat = np.cos(lat)
    slon = np.sin(lon_cel)
    clon = np.cos(lon_cel)
    sbii1 = -slat * clon * sbid1 - clat * clon * sbid3
    sbii2 = -slat * slon * sbid1 - clat * slon * sbid3
    sbii3 = clat * sbid1 - slat * sbid3
    return np.array([sbii1, sbii2, sbii3])


def cad_tdi84(lon, lat, alt, time):
    tdi = np.zeros((3, 3))
    lon_cel = GW_CLONG + WEII3 * time + lon
    tdi13 = np.cos(lat)
    tdi33 = -np.sin(lat)
    tdi22 = np.cos(lon_cel)
    tdi21 = -np.sin(lon_cel)
    tdi[0, 2] = tdi13
    tdi[2, 2] = tdi33
    tdi[1, 1] = tdi22
    tdi[1, 0] = tdi21
    tdi[0, 0] = tdi33 * tdi22
    tdi[0, 1] = -tdi33 * tdi21
    tdi[2, 0] = -tdi13 * tdi22
    tdi[2, 1] = tdi13 * tdi21
    return tdi


def cad_tgi84(lon, lat, alt, time):
    tdi = np.zeros((3, 3))
    tgd = np.zeros((3, 3))
    lon_cel = GW_CLONG + WEII3 * time + lon
    tdi13 = np.cos(lat)
    tdi33 = -np.sin(lat)
    tdi22 = np.cos(lon_cel)
    tdi21 = -np.sin(lon_cel)
    tdi[0, 2] = tdi13
    tdi[2, 2] = tdi33
    tdi[1, 1] = tdi22
    tdi[1, 0] = tdi21
    tdi[0, 0] = tdi33 * tdi22
    tdi[0, 1] = -tdi33 * tdi21
    tdi[2, 0] = -tdi13 * tdi22
    tdi[2, 1] = tdi13 * tdi21
    r0 = SMAJOR_AXIS * (
        1.0
        - FLATTENING * (1.0 - np.cos(2.0 * lat)) / 2.0
        + 5.0 * FLATTENING**2 * (1.0 - np.cos(4.0 * lat)) / 16.0
    )
    dd = FLATTENING * np.sin(2.0 * lat) * (1.0 - FLATTENING / 2.0 - alt / r0)
    tgd[0, 0] = np.cos(dd)
    tgd[2, 2] = np.cos(dd)
    tgd[1, 1] = 1
    tgd[2, 0] = np.sin(dd)
    tgd[0, 2] = -np.sin(dd)
    return tgd @ tdi
