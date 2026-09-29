"""AIM5 G1I MTARG=1 polar target init (Fortran MODULE.FOR G1I)."""

import math

import numpy as np

from cadac.constants import RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import StateStore
from cadac.vehicles.flat3.aim5.target import Aim5Target

RTOL = 1e-12
ATOL = 1e-12

# Planted values from brief / INLENV.ASC
MTARG = 1
HTE = 9000.0
DHTB = 5000.0
AZTLX = 10.0
SBEL = np.array([0.0, 0.0, -10000.0], dtype=float)


def _fortran_mtarg1_stbl():
    """Hand-computed G1I MTARG=1: DUMH, THTTL0, DBT1, MATCAR(STBL)."""
    dumh = float(SBEL[2]) + HTE
    thttl0 = math.atan2(dumh, DHTB)
    dbt1 = math.sqrt(DHTB**2 + dumh**2)
    az = AZTLX * RAD
    cel = math.cos(thttl0)
    sel = math.sin(thttl0)
    stbl = np.array(
        [
            dbt1 * cel * math.cos(az),
            dbt1 * cel * math.sin(az),
            -dbt1 * sel,
        ],
        dtype=float,
    )
    st1el0 = stbl + SBEL
    return {"STBL": stbl, "ST1EL0": st1el0, "ST1EL": st1el0.copy(), "dbt1": dbt1}


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(0.0, 0.01, 0.0, 0.0, None, 0)


def _aim5_aircraft_or_target():
    vehicle = _Vehicle()
    target = Aim5Target()
    target.define(vehicle)
    return vehicle, target


def _init_target(veh, target):
    target.initialize(veh, _ctx())


def test_name_is_target():
    assert Aim5Target().name == "target"


def test_aim5_mtarg_1_polar_init_stbl():
    # DUMH=SBEL(3)+HTE; THTTL0=atan2(DUMH,DHTB); DBT1=hypot; MATCAR(STBL,…)
    veh, target = _aim5_aircraft_or_target()
    veh.store.set("mtarg", MTARG)
    veh.store.set("hte", HTE)
    veh.store.set("dhtb", DHTB)
    veh.store.set("aztlx", AZTLX)
    # SBEL z = -10000 as in INLENV
    veh.store.set("SBEL", SBEL.copy())
    _init_target(veh, target)
    exp = _fortran_mtarg1_stbl()
    stbl = veh.store.get("STBL")
    assert stbl is not None
    np.testing.assert_allclose(stbl, exp["STBL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        veh.store.get("ST1EL0"), exp["ST1EL0"], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        veh.store.get("ST1EL"), exp["ST1EL"], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(veh.store.get("dbt1"), exp["dbt1"], rtol=RTOL, atol=ATOL)
