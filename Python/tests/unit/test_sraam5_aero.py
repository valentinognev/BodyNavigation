"""SRAAM5 A1: table lookup / body force coeffs at planted Mach/α (MODULE.FOR)."""

from math import acos, atan2, cos, sin, tan
from types import SimpleNamespace

import numpy as np

from cadac.constants import DEG, RAD
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat5.sraam5.aerodynamics import Sraam5Aerodynamics

RTOL = 1e-12
ATOL = 1e-14

# Exact ATBLOF/TNTBL2/ATBLSF grid point (mach=0.6, α_p=3°, alt=-100 ft, thrust off)
VMACH = 0.6
ALPHAX_DEG = 3.0
BETA = 0.0
OPTMET = 0.0
HBE = -100.0  # HBESK = HBE*(1+2.281*OPTMET) → -100 ft breakpoint
XCGIN = 60.46  # > 57.04 → ALPMAX = 45 (no α clamp)
MPROP = 0
AREA = 0.01824
AMASS = 91.7
AGRAV = 9.80675445
PDYNMC = 20000.0
ANPLIM = 50.0
TRLOAD = 3.0
TRALP = 1.0

# Fortran DATA at (mach=0.6, α=3°, thrust-off, CSFC at -100 ft)
CDT_TABLE = 0.459
CLT_TABLE = 0.425
CSFC_TABLE = 0.002


def _vehicle(**overrides):
    store = StateStore()
    vehicle = SimpleNamespace(store=store)
    aero = Sraam5Aerodynamics()
    aero.define(vehicle)
    planted = {
        "vmach": VMACH,
        "alpha": ALPHAX_DEG * RAD,
        "beta": BETA,
        "optmet": OPTMET,
        "hbe": HBE,
        "xcgin": XCGIN,
        "mprop": MPROP,
        "amass": AMASS,
        "agrav": AGRAV,
        "pdynmc": PDYNMC,
        "anplim": ANPLIM,
        "trload": TRLOAD,
        "tralp": TRALP,
        "area": AREA,
    }
    planted.update(overrides)
    for name, value in planted.items():
        if name not in store:
            kind = "int" if name == "mprop" else "real"
            store.define(Field(name, value, kind, "out", "planted"))
        store.set(name, value)
    return vehicle, aero


def _expected_force_coeffs(alpha, beta, cdt, clt, csfc):
    """Fortran A1 body-axis force coeffs from wind-axis CD/CLT (CRAD=DEG)."""
    crad = DEG
    alphap = acos(cos(alpha) * cos(beta))
    dum1 = tan(beta)
    dum2 = sin(alpha)
    phipp = np.copysign(1.570796, beta)
    if abs(dum2) > 1.0e-10:
        phipp = atan2(dum1, dum2)
    alphapx = abs(alphap * crad)
    cd = cdt + csfc
    ca = cd * cos(alphapx / crad) - clt * sin(alphapx / crad)
    cn = cd * sin(alphapx / crad) + clt * cos(alphapx / crad)
    if alphap < 0.0:
        cn = -cn
    cy = -cn * sin(phipp)
    cn = cn * cos(phipp)
    return ca, cn, cy, cd, alphapx, phipp


def test_name_is_aerodynamics():
    assert Sraam5Aerodynamics().name == "aerodynamics"


def test_a1_force_coeffs_at_planted_mach_alpha_thrust_off():
    """Grid-point look-up + CA/CN/CY transform matches MODULE.FOR A1."""
    vehicle, aero = _vehicle()
    aero.execute(vehicle, None)
    store = vehicle.store

    np.testing.assert_allclose(store.get("cdt"), CDT_TABLE, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("csfc"), CSFC_TABLE, rtol=RTOL, atol=ATOL)
    ca, cn, cy, cd, alphapx, phipp = _expected_force_coeffs(
        ALPHAX_DEG * RAD, BETA, CDT_TABLE, CLT_TABLE, CSFC_TABLE
    )
    np.testing.assert_allclose(store.get("cd"), cd, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ca"), ca, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("cn"), cn, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("cy"), cy, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("alphapx"), alphapx, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("phipp"), phipp, rtol=RTOL, atol=ATOL)

    alpx = abs(DEG * (ALPHAX_DEG * RAD))
    cnalp = DEG * (0.123 + 0.013 * alpx)
    np.testing.assert_allclose(store.get("cnalp"), cnalp, rtol=RTOL, atol=ATOL)
