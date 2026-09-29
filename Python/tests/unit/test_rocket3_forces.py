"""ROCKET3 Forces A3 — specific force in velocity axes (Fortran MODULE.FOR A3)."""

from math import cos, sin

import numpy as np
import pytest

from cadac.constants import RAD
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round3.rocket3.forces import Rocket3Forces

RTOL = 1e-12
ATOL = 1e-14

# Planted A3 inputs (Fortran names; THRUSTX in kN)
PDYNMC = 5.0e4
SREF = 1.5
CD = 0.4
CL = 0.8
THRUSTX = 250.0  # kN
VMASS = 10800.0
ALPHAX = -5.5
PHIMVX = 20.0


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _expected_fspv(pdynmc, sref, cd, cl, thrustx, vmass, alphax, phimvx):
    """Fortran A3: FAPM aero+thrust in maneuver plane → FSPV / VMASS."""
    alpha = alphax * RAD
    phimv = phimvx * RAD
    fd = pdynmc * sref * cd
    fl = pdynmc * sref * cl
    thrust = thrustx * 1000.0
    fapm1 = -fd + thrust * cos(alpha)
    fapm3 = -(fl + thrust * sin(alpha))
    return np.array(
        [
            fapm1 / vmass,
            -sin(phimv) * fapm3 / vmass,
            cos(phimv) * fapm3 / vmass,
        ],
        dtype=float,
    )


def _ready(*, pdynmc=PDYNMC, sref=SREF, cd=CD, cl=CL, thrustx=THRUSTX,
           vmass=VMASS, alphax=ALPHAX, phimvx=PHIMVX):
    vehicle = _Vehicle()
    forces = Rocket3Forces()
    forces.define(vehicle)
    store = vehicle.store
    for name, value, module in (
        ("pdynmc", pdynmc, "environment"),
        ("sref", sref, "aerodynamics"),
        ("cd", cd, "aerodynamics"),
        ("cl", cl, "aerodynamics"),
        ("thrustx", thrustx, "propulsion"),
        ("vmass", vmass, "propulsion"),
        ("alphax", alphax, "aerodynamics"),
        ("phimvx", phimvx, "aerodynamics"),
    ):
        store.define(Field(name, value, "real", "out", module))
    return vehicle, forces


def test_name_is_forces():
    assert Rocket3Forces().name == "forces"


def test_a3_fspv_matches_fortran_force_sum():
    """A3: FD/FL + thrust(kN→N) → FSPV in velocity axes (feeds geographic EOM)."""
    vehicle, forces = _ready()
    expected = _expected_fspv(PDYNMC, SREF, CD, CL, THRUSTX, VMASS, ALPHAX, PHIMVX)

    forces.execute(vehicle, None)

    np.testing.assert_allclose(
        vehicle.store.get("FSPV"), expected, rtol=RTOL, atol=ATOL
    )


def test_a3_fspv_level_phimvx_0():
    """PHIMVX=0: lateral FSPV(2)=0; normal along -vertical in velocity frame."""
    vehicle, forces = _ready(phimvx=0.0)
    expected = _expected_fspv(PDYNMC, SREF, CD, CL, THRUSTX, VMASS, ALPHAX, 0.0)
    assert expected[1] == pytest.approx(0.0, abs=ATOL)

    forces.execute(vehicle, None)

    np.testing.assert_allclose(
        vehicle.store.get("FSPV"), expected, rtol=RTOL, atol=ATOL
    )
