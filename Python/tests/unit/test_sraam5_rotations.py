"""SRAAM5 D2 rotations — skid-to-turn WBVB (MTURN=0)."""

from math import cos, sin
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat5.sraam5.rotations import Sraam5Rotations

RTOL = 1e-12
ATOL = 1e-14

# Planted intermediate AoA / rates (rad, rad/s) — Fortran D2 MTURN=0
ALP = 0.25
ALPD = 0.05
BETD = -0.12


def _ctx():
    return SimpleNamespace(int_step=0.01)


def test_sraam5_d2_skid_to_turn_wbvb_from_alp_betd():
    """Fortran D2 MTURN=0: WBVB = [BETD*sin(ALP), ALPD, -BETD*cos(ALP)]."""
    vehicle = SimpleNamespace(store=StateStore())
    rot = Sraam5Rotations()
    rot.define(vehicle)
    store = vehicle.store

    assert store.get("mturn") == 0

    store.define(Field("alp", ALP, "real", "state", "control"))
    store.define(Field("alpd", ALPD, "real", "state", "control"))
    store.define(Field("betd", BETD, "real", "state", "control"))

    rot.execute(vehicle, _ctx())

    want = np.array(
        [BETD * sin(ALP), ALPD, -BETD * cos(ALP)],
        dtype=float,
    )
    got = store.get("WBVB")
    assert got == pytest.approx(want, rel=RTOL, abs=ATOL)
