"""Task 68: SRAAM5/SRAAM6 G4SHAZ (MTERM>0) — AFATL-TR-86-32 YSS/ZSS/DYRB."""

from math import copysign

import numpy as np
import pytest

from cadac.constants import DEG
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import cadac_matmul, mat2tr, mat3tr, polar_from_cart
from cadac.vehicles.flat5.sraam5.intercept import Sraam5Intercept, g4shaz as g4shaz5
from cadac.vehicles.flat6.sraam6.intercept import Sraam6Intercept, g4shaz as g4shaz6

RTOL = 1e-12
ATOL = 1e-12
# Fortran G4SHAZ PARAMETER(PI=3.14159)
_FTN_PI = 3.14159

SBTL = np.array([2.0, -3.0, 1.5], dtype=float)
VBT1L = np.array([400.0, 50.0, -20.0], dtype=float)
VT1EL = np.array([240.0, 0.0, 0.0], dtype=float)
VBEL = np.array([600.0, 40.0, -30.0], dtype=float)
TT1L = mat3tr(0.1, -0.05, 0.2)


def _fortran_yss_zss_dyrb(sbtl, vbt1l, vt1el, vbel, tt1l):
    """Replica of MODULE.FOR G4SHAZ miss-plane formulas (YSS/ZSS/DYRB)."""
    _dvt1e, psiul, thtul = polar_from_cart(np.asarray(vt1el, dtype=float))
    tul = mat2tr(float(psiul), float(thtul))
    vbt1t1 = cadac_matmul(np.asarray(tt1l, dtype=float), np.asarray(vbt1l, dtype=float))
    _dv, psiyt1, thtyt1 = polar_from_cart(vbt1t1)
    vbt1u = cadac_matmul(tul, np.asarray(vbt1l, dtype=float))
    _dv2, psizu, thtzu = polar_from_cart(vbt1u)
    tzu = mat2tr(float(psizu), float(thtzu))
    tzl = cadac_matmul(tzu, tul)
    shjz = cadac_matmul(tzl, np.asarray(sbtl, dtype=float))
    yss = -float(shjz[1])
    zss = -float(shjz[2])
    tyt1 = mat2tr(float(psiyt1), float(thtyt1))
    tyl = cadac_matmul(tyt1, np.asarray(tt1l, dtype=float))
    shjy = cadac_matmul(tyl, np.asarray(sbtl, dtype=float))
    dyrb = -float(shjy[1])
    return yss, zss, dyrb


def _plant_shazam_inputs(store):
    for name, value, kind in (
        ("VT1EL", VT1EL.copy(), "vec"),
        ("VBEL", VBEL.copy(), "vec"),
        ("TT1L", TT1L.copy(), "mat"),
        ("hbe", 5000.0, "real"),
        ("yss", 0.0, "real"),
        ("zss", 0.0, "real"),
        ("dyrb", 0.0, "real"),
        ("dzrb", 0.0, "real"),
        ("aspazx", 0.0, "real"),
        ("aspelx", 0.0, "real"),
        ("azintx", 0.0, "real"),
        ("elintx", 0.0, "real"),
    ):
        if name not in store:
            store.define(Field(name, value, kind, "data", "plant"))
        store.set(name, value)


@pytest.mark.parametrize(
    "g4shaz",
    [g4shaz5, g4shaz6],
    ids=["sraam5", "sraam6"],
)
def test_g4shaz_planted_sbtl_vbt1l_writes_yss_zss_dyrb(g4shaz):
    store = StateStore()
    _plant_shazam_inputs(store)
    want_yss, want_zss, want_dyrb = _fortran_yss_zss_dyrb(SBTL, VBT1L, VT1EL, VBEL, TT1L)

    g4shaz(store, SBTL, VBT1L)

    np.testing.assert_allclose(store.get("yss"), want_yss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("zss"), want_zss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("dyrb"), want_dyrb, rtol=RTOL, atol=ATOL)
    # Aspect az uses Fortran PI=3.14159 SIGN form
    vbeu = cadac_matmul(mat2tr(*polar_from_cart(VT1EL)[1:]), VBEL)
    _dv, psivu, thtvu = polar_from_cart(vbeu)
    want_aspazx = -copysign((_FTN_PI - abs(float(psivu))), float(psivu)) * DEG
    want_aspelx = -float(thtvu) * DEG
    np.testing.assert_allclose(store.get("aspazx"), want_aspazx, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("aspelx"), want_aspelx, rtol=RTOL, atol=ATOL)


def test_sraam5_and_sraam6_define_shazam_fields():
    for cls in (Sraam5Intercept, Sraam6Intercept):
        vehicle = type("V", (), {"store": StateStore()})()
        cls().define(vehicle)
        for name in ("yss", "zss", "dyrb", "mterm"):
            assert name in vehicle.store.names(), f"{cls.__name__} missing {name}"
