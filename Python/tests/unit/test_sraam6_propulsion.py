from pathlib import Path

import numpy as np
import pytest

from cadac.env.us76 import atmosphere76
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.flat6.sraam6.propulsion import Sraam6Propulsion

SRAAM6 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/SRAAM6_250130/SRAAM6"
PROP = SRAAM6 / "sraam6_prop_deck.asc"

RTOL = 1e-12
ATOL = 1e-14
PSL = 101325.0
AEXIT = 0.0125
PRESS = atmosphere76(5000.0)[1]


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(0.0, 0.001, 0.0, 0.0, None, 0)


def _deck():
    _, tables = parse_asc_deck(PROP)
    return Datadeck.from_tables(tables)


def _ready(*, time=0.0, mprop=1, aexit=AEXIT, press=PRESS):
    vehicle = _Vehicle()
    prop = Sraam6Propulsion(_deck())
    prop.define(vehicle)
    store = vehicle.store
    store.define(Field("time", time, "real", "exec", "kinematics"))
    store.define(Field("press", press, "real", "out", "environment"))
    store.set("mprop", mprop)
    store.set("aexit", aexit)
    return vehicle, prop, _deck()


def test_mprop1_time0_thrust_matches_cpp_formula():
    vehicle, prop, deck = _ready(time=0.0, mprop=1, aexit=AEXIT, press=PRESS)
    prop.execute(vehicle, _ctx())
    tsl = deck.look_up("thrust_vs_time", 0.0)
    want = tsl + (PSL - PRESS) * AEXIT
    np.testing.assert_allclose(vehicle.store.get("thrust"), want, rtol=RTOL, atol=ATOL)


def test_mprop0_thrust_zero():
    vehicle, prop, _ = _ready(time=0.0, mprop=0)
    prop.execute(vehicle, _ctx())
    np.testing.assert_allclose(vehicle.store.get("thrust"), 0.0, rtol=RTOL, atol=ATOL)


def test_mprop2_raises():
    vehicle, prop, _ = _ready(time=0.0, mprop=2)
    with pytest.raises(ValueError):
        prop.execute(vehicle, _ctx())


def test_time_past_burnout_same_step_keeps_nozzle_thrust():
    vehicle, prop, deck = _ready(time=2.70, mprop=1)
    prop.execute(vehicle, _ctx())
    tsl = deck.look_up("thrust_vs_time", 2.70)
    want = tsl + (PSL - PRESS) * AEXIT
    assert vehicle.store.get("mprop") == 0
    np.testing.assert_allclose(vehicle.store.get("thrust"), want, rtol=RTOL, atol=ATOL)
    assert want != 0.0
    prop.execute(vehicle, _ctx())
    assert vehicle.store.get("mprop") == 0
    np.testing.assert_allclose(vehicle.store.get("thrust"), 0.0, rtol=RTOL, atol=ATOL)
