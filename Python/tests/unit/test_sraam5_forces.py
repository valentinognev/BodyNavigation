"""SRAAM5 Forces A3 / A3TRA — body/vel DCM for MTURN paths (Fortran MODULE.FOR)."""

from math import cos, sin

import numpy as np

from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat5.sraam5.forces import Sraam5Forces

RTOL = 1e-12
ATOL = 1e-14

# Planted A3TRA angles (rad) — Fortran ALPHA/BETA/PHIBV
ALPHA = 0.12
BETA = -0.05
PHIBV = 0.3

# Minimal A3 force inputs (force path uses identity TLB)
AGRAV = 9.806635
AREA = 0.01824
PDYNMC = 4.0e4
CA = 0.3
CN = 1.2
CY = -0.1
FTHALT = 5000.0
AMASS = 90.0


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _zeros33():
    return np.zeros((3, 3), dtype=float)


def _eye33():
    return np.eye(3, dtype=float)


def _ready(*, mturn, alpha=ALPHA, beta=BETA, phibv=PHIBV):
    vehicle = _Vehicle()
    forces = Sraam5Forces()
    forces.define(vehicle)
    store = vehicle.store
    for name, value, ftype in (
        ("mturn", mturn, "int"),
        ("alpha", alpha, "real"),
        ("beta", beta, "real"),
        ("phibv", phibv, "real"),
        ("agrav", AGRAV, "real"),
        ("pdynmc", PDYNMC, "real"),
        ("area", AREA, "real"),
        ("ca", CA, "real"),
        ("cn", CN, "real"),
        ("cy", CY, "real"),
        ("fthalt", FTHALT, "real"),
        ("amass", AMASS, "real"),
        ("TLB", _eye33(), "mat"),
    ):
        if name not in store:
            store.define(Field(name, value, ftype, "data", "test"))
        else:
            store.set(name, value)
    return vehicle, forces


def _tbv_skid_to_turn(alpha, beta):
    """Fortran A3TRA MTURN=0 (yaw-to-turn) TBV."""
    calp, salp = cos(alpha), sin(alpha)
    cbet, sbet = cos(beta), sin(beta)
    return np.array(
        [
            [calp * cbet, -calp * sbet, -salp],
            [sbet, cbet, 0.0],
            [salp * cbet, -salp * sbet, calp],
        ],
        dtype=float,
    )


def _tbv_bank_to_turn(alpha, phibv):
    """Fortran A3TRA MTURN=1 (bank-to-turn) TBV."""
    calp, salp = cos(alpha), sin(alpha)
    cphi, sphi = cos(phibv), sin(phibv)
    return np.array(
        [
            [calp, salp * sphi, -salp * cphi],
            [0.0, cphi, sphi],
            [salp, -calp * sphi, calp * cphi],
        ],
        dtype=float,
    )


def test_name_is_forces():
    assert Sraam5Forces().name == "forces"


def test_a3tra_mturn0_skid_to_turn_tbv_matches_fortran():
    """A3TRA MTURN=0: TBV from ALPHA/BETA (body wrt flight-path)."""
    vehicle, forces = _ready(mturn=0)
    forces.execute(vehicle, None)
    expected = _tbv_skid_to_turn(ALPHA, BETA)
    np.testing.assert_allclose(vehicle.store.get("TBV"), expected, rtol=RTOL, atol=ATOL)


def test_a3tra_mturn1_bank_to_turn_tbv_matches_fortran():
    """A3TRA MTURN=1: TBV from ALPHA/PHIBV (body wrt flight-path)."""
    vehicle, forces = _ready(mturn=1)
    forces.execute(vehicle, None)
    expected = _tbv_bank_to_turn(ALPHA, PHIBV)
    np.testing.assert_allclose(vehicle.store.get("TBV"), expected, rtol=RTOL, atol=ATOL)


def test_a3_fspb_matches_fortran_force_sum():
    """A3: FAB/AMASS → FSPB with fractional aero coeffs (FRACA/N/Y=0)."""
    vehicle, forces = _ready(mturn=0)
    forces.execute(vehicle, None)
    fab = np.array(
        [
            FTHALT - CA * PDYNMC * AREA,
            CY * PDYNMC * AREA,
            -CN * PDYNMC * AREA,
        ],
        dtype=float,
    )
    fspb = fab / AMASS
    np.testing.assert_allclose(vehicle.store.get("FSPB"), fspb, rtol=RTOL, atol=ATOL)
    # FSPV = TVB @ FSPB with TVB = TBV^T
    tbv = _tbv_skid_to_turn(ALPHA, BETA)
    fspv = tbv.T @ fspb
    np.testing.assert_allclose(vehicle.store.get("FSPV"), fspv, rtol=RTOL, atol=ATOL)
