"""ROCKET3 A1: stage coeff select for MAERO 11/12/13 (MODULE.FOR)."""

from math import cos, sin
from types import SimpleNamespace

import numpy as np

from cadac.constants import DEG
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round3.rocket3.aerodynamics import Rocket3Aerodynamics

RTOL = 1e-12
ATOL = 1e-14

# Planted flight condition (not a table breakpoint — polynomials are closed-form)
VMACH = 2.5
ALPHAX = -5.5


def _fortran_stage_coeffs(maero: int, vmach: float, alphax: float, mprop: int):
    """Exact MODULE.FOR A1 formulas for MAERT=1 stages."""
    maert = int(maero / 10.0)
    maerv = maero - maert * 10
    assert maert == 1
    alpha = alphax / DEG
    calph = cos(alpha)
    salph = sin(alpha)
    caa = cnn = 0.0
    if maerv == 1:
        if mprop == 2:
            caa = 0.281 + 0.186 * vmach - 0.056 * vmach**2 + 0.00366 * vmach**3
        else:
            caa = 0.346 + 0.183 * vmach - 0.058 * vmach**2 + 0.00382 * vmach**3
        cnn = (5.006 - 0.519 * vmach + 0.031 * vmach**2) * alpha
    elif maerv == 2:
        if mprop == 2:
            caa = 0.236 - 0.043 * vmach + 0.0029 * vmach**2 - 0.00006 * vmach**3
        else:
            caa = 0.327 - 0.067 * vmach + 0.005 * vmach**2 - 0.0001 * vmach**3
        cnn = (1.714 - 0.038 * vmach + 0.0014 * vmach**2) * alpha
    elif maerv == 3:
        caa = 0.02
        cnn = 1.0 * alpha
    else:
        raise AssertionError(f"unexpected MAERV={maerv}")
    cdd = caa * calph + cnn * salph
    cll = cnn * calph - caa * salph
    cl = cll
    cd = cdd
    clovercd = cl / cd
    cn = cl * calph + cd * salph
    ca = cd * calph - cl * salph
    return cd, cl, ca, cn, clovercd


def _vehicle(maero: int, mprop: int = 2, vmach: float = VMACH, alphax: float = ALPHAX):
    store = StateStore()
    vehicle = SimpleNamespace(store=store)
    aero = Rocket3Aerodynamics()
    aero.define(vehicle)
    planted = {
        "maero": maero,
        "mprop": mprop,
        "vmach": vmach,
        "alphax": alphax,
    }
    for name, value in planted.items():
        if name not in store:
            kind = "int" if name in ("maero", "mprop") else "real"
            store.define(Field(name, value, kind, "out", "planted"))
        store.set(name, value)
    return vehicle, aero


def test_name_is_aerodynamics():
    assert Rocket3Aerodynamics().name == "aerodynamics"


def test_stage_11_burning_coeffs_match_fortran():
    """MAERO=11 (stage 1) burning polynomials → CD/CL/CA/CN."""
    vehicle, aero = _vehicle(maero=11, mprop=2)
    aero.execute(vehicle, None)
    store = vehicle.store
    cd, cl, ca, cn, clovercd = _fortran_stage_coeffs(11, VMACH, ALPHAX, 2)
    np.testing.assert_allclose(store.get("cd"), cd, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("cl"), cl, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ca"), ca, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("cn"), cn, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("clovercd"), clovercd, rtol=RTOL, atol=ATOL)


def test_stage_11_coast_uses_coast_caa():
    """MAERO=11 non-burning CA polynomial differs from burning."""
    vehicle, aero = _vehicle(maero=11, mprop=0)
    aero.execute(vehicle, None)
    store = vehicle.store
    cd_coast, _, _, _, _ = _fortran_stage_coeffs(11, VMACH, ALPHAX, 0)
    cd_burn, _, _, _, _ = _fortran_stage_coeffs(11, VMACH, ALPHAX, 2)
    assert abs(cd_coast - cd_burn) > 1e-6
    np.testing.assert_allclose(store.get("cd"), cd_coast, rtol=RTOL, atol=ATOL)


def test_stage_12_burning_coeffs_match_fortran():
    """MAERO=12 (stage 2) burning polynomials → CD/CL/CA/CN."""
    vehicle, aero = _vehicle(maero=12, mprop=2)
    aero.execute(vehicle, None)
    store = vehicle.store
    cd, cl, ca, cn, clovercd = _fortran_stage_coeffs(12, VMACH, ALPHAX, 2)
    np.testing.assert_allclose(store.get("cd"), cd, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("cl"), cl, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ca"), ca, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("cn"), cn, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("clovercd"), clovercd, rtol=RTOL, atol=ATOL)


def test_stage_13_coeffs_match_fortran():
    """MAERO=13 (stage 3) fixed CA=0.02, CN=α → CD/CL/CA/CN."""
    vehicle, aero = _vehicle(maero=13, mprop=2)
    aero.execute(vehicle, None)
    store = vehicle.store
    cd, cl, ca, cn, clovercd = _fortran_stage_coeffs(13, VMACH, ALPHAX, 2)
    np.testing.assert_allclose(store.get("cd"), cd, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("cl"), cl, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ca"), ca, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("cn"), cn, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("clovercd"), clovercd, rtol=RTOL, atol=ATOL)
