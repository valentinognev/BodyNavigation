"""CRUISE5 Fortran C2 MTURN=0 + MAUTL=1 sideslip angle hold (BETA=BETAC)."""

import numpy as np

from cadac.constants import DEG
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round3.cruise5.control import Cruise5Control

RTOL = 1e-12
ATOL = 1e-14
INT_STEP = 0.05

# Planted STT sideslip hold (rad) — Fortran C2 MAUTL=1 / MAUTP=1
BETAC = 0.05
ALPHAC = 0.02
MAUT = 11  # MAUTL=1 sideslip hold, MAUTP=1 AoA hold
MTURN = 0


def _ctx():
    return SimContext(0.0, INT_STEP, 0.0, 0.0, None, 0)


def _ready(**overrides):
    vehicle = type("V", (), {"store": StateStore()})()
    control = Cruise5Control()
    control.define(vehicle)
    store = vehicle.store
    for name, value, ftype in (
        ("tgv", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), "mat"),
        ("pdynmc", 5000.0, "real"),
        ("area", 0.929, "real"),
        ("thrust", 1500.0, "real"),
        ("mass", 1000.0, "real"),
        ("dvbe", 200.0, "real"),
    ):
        if name not in store.names():
            store.define(Field(name, value, ftype, "out", "test"))
        else:
            store.set(name, value)
    store.set("maut", MAUT)
    store.set("mturn", MTURN)
    store.set("mcontrol", 0)
    store.set("betac", BETAC)
    store.set("alphac", ALPHAC)
    for key, value in overrides.items():
        store.set(key, value)
    return vehicle, control


def test_mturn0_mautl1_holds_betac_as_betax():
    """Fortran C2 MTURN=0 MAUTL=1: BETA=BETAC → BETAX=CRAD*BETA; PHIBV=0."""
    vehicle, control = _ready()
    store = vehicle.store
    control.execute(vehicle, _ctx())

    np.testing.assert_allclose(store.get("betax"), BETAC * DEG, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("phimvx"), 0.0, rtol=RTOL, atol=ATOL)
    # MAUTP=1 companion: ALPHA=ALPHAC
    np.testing.assert_allclose(store.get("alphax"), ALPHAC * DEG, rtol=RTOL, atol=ATOL)
    assert store.get("TBV").shape == (3, 3)


def test_mturn0_mautl1_nonzero_betac_changes_betax():
    """Sideslip command must drive betax (not remain zero)."""
    vehicle, control = _ready(betac=0.12)
    store = vehicle.store
    control.execute(vehicle, _ctx())
    np.testing.assert_allclose(store.get("betax"), 0.12 * DEG, rtol=RTOL, atol=ATOL)
    assert abs(store.get("betax")) > 1.0  # deg, clearly nonzero
