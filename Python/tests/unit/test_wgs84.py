import numpy as np

from cadac.constants import REARTH
from cadac.math.wgs84 import (
    GM,
    SMALL,
    cad_geo84_in,
    cad_grav84,
    cad_in_geo84,
    cad_kepler,
    cad_tdi84,
    cad_tgi84,
)

_RTOL = 1e-12
_ATOL = 1e-14


def _morth_fk_gk(sbii, vbii, tgo):
    """Replica of ROCKET6 C++ cad_kepler fk/gk (Morth)."""
    sqrt_gm = np.sqrt(GM)
    ro = np.linalg.norm(sbii)
    vo = np.linalg.norm(vbii)
    rvo = float(sbii[0] * vbii[0] + sbii[1] * vbii[1] + sbii[2] * vbii[2])
    a1 = vo * vo / GM
    sa = ro / (2 - ro * a1)
    smua = sqrt_gm * np.sqrt(sa)
    mdot = smua / (sa * sa)
    dm = mdot * tgo
    de = dm
    a11 = rvo / smua
    a21 = (sa - ro) / sa
    count20 = 0
    while True:
        cde = 1 - np.cos(de)
        sde = np.sin(de)
        dmn = de + a11 * cde - a21 * sde
        dmerr = dm - dmn
        adm = abs(dmerr) / mdot
        dmde = 1 + a11 * sde - a21 * (1 - cde)
        de = de + dmerr / dmde
        count20 += 1
        if count20 > 20:
            raise AssertionError("Morth replica did not converge")
        if not (adm > SMALL):
            break
    fk = (ro - sa * cde) / ro
    gk = (dm + sde - de) / mdot
    spii = sbii * fk + vbii * gk
    rp = np.linalg.norm(spii)
    fdk = -smua * sde / ro
    gdk = rp - sa * cde
    vpii = sbii * (fdk / rp) + vbii * (gdk / rp)
    return spii, vpii


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


def test_cad_kepler_circular_equatorial_matches_fk_gk():
    r = REARTH + 400e3
    sbii = np.array([r, 0.0, 0.0])
    v = np.sqrt(GM / r)
    vbii = np.array([0.0, v, 0.0])
    tgo = 1.0
    spii, vpii, flag = cad_kepler(sbii, vbii, tgo)
    assert flag == 0
    np.testing.assert_allclose(np.linalg.norm(spii), r, rtol=_RTOL)
    exp_spii, exp_vpii = _morth_fk_gk(sbii, vbii, tgo)
    np.testing.assert_allclose(spii, exp_spii, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(vpii, exp_vpii, rtol=_RTOL, atol=_ATOL)


def test_cad_kepler_flag_one_returns_input_copies():
    r = REARTH + 400e3
    sbii = np.array([r, 0.0, 0.0])
    vbii = np.array([0.0, 2.0 * np.sqrt(2.0 * GM / r), 0.0])
    tgo = 1.0
    spii, vpii, flag = cad_kepler(sbii, vbii, tgo)
    assert flag == 1
    np.testing.assert_array_equal(spii, sbii)
    np.testing.assert_array_equal(vpii, vbii)
    spii[0] = 0.0
    vpii[0] = 0.0
    assert sbii[0] == r
    assert vbii[0] == 0.0
