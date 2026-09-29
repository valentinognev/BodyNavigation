"""HYPER6 mprop 3 (LTG rocket) and 4 (constant-thrust rocket)."""

import numpy as np
import pytest

from cadac.constants import AGRAV
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round6.hyper6.propulsion import Hyper6Propulsion

RTOL = 1e-12
ATOL = 1e-14
DT = 0.01

# input_TV.asc exo / rocket deck values
VMASS0 = 16000.0
FMASS0 = 12000.0
ISP_FUEL = 306.0
FUEL_FLOW_RATE = 80.0
# LTG: guidance burntime such that fmass0/burntime matches TV cruise flow
BURNTIME = FMASS0 / FUEL_FLOW_RATE
MOI_ROLL_EXO_0 = 11.25e3
MOI_ROLL_EXO_1 = 6.75e3
MOI_TRANS_EXO_0 = 169e3
MOI_TRANS_EXO_1 = 101.4e3


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


class _BoomDeck:
    def look_up(self, *args, **kwargs):
        raise AssertionError("look_up must not be called for rocket mprop")


def _ctx(dt=DT):
    return SimContext(
        sim_time=0.0,
        int_step=dt,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _plant(store):
    for name, module in (
        ("rho", "environment"),
        ("vmach", "environment"),
        ("pdynmc", "environment"),
        ("dvba", "environment"),
        ("cd", "aerodynamics"),
        ("refa", "aerodynamics"),
        ("alphax", "kinematics"),
        ("isp_fuel", "guidance"),
        ("burntime", "guidance"),
    ):
        store.define(Field(name, 0.0, "real", "out", module))


def _ready(*, mprop, **kw):
    vehicle = _Vehicle()
    prop = Hyper6Propulsion(_BoomDeck())
    prop.define(vehicle)
    _plant(vehicle.store)
    store = vehicle.store
    data = {
        "mprop": mprop,
        "vmass0": VMASS0,
        "fmass0": FMASS0,
        "fuel_flow_rate": FUEL_FLOW_RATE,
        "isp_fuel": ISP_FUEL,
        "burntime": BURNTIME,
        "moi_roll_exo_0": MOI_ROLL_EXO_0,
        "moi_roll_exo_1": MOI_ROLL_EXO_1,
        "moi_trans_exo_0": MOI_TRANS_EXO_0,
        "moi_trans_exo_1": MOI_TRANS_EXO_1,
        "fmasse": 0.0,
        "fmassd": 0.0,
    }
    data.update(kw)
    for name, value in data.items():
        store.set(name, value)
    prop.initialize(vehicle, _ctx())
    return vehicle, prop


def test_hyper6_mprop_3_ltg_thrust():
    """mprop=3: spi=isp_fuel; flow=fmass0/burntime; exo MOI; no deck lookup."""
    vehicle, prop = _ready(mprop=3)
    store = vehicle.store
    prop.execute(vehicle, _ctx())

    flow = FMASS0 / BURNTIME
    want_thrust = ISP_FUEL * flow * AGRAV
    assert store.get("thrust") == pytest.approx(want_thrust, rel=RTOL, abs=ATOL)
    assert store.get("spi") == pytest.approx(ISP_FUEL, rel=RTOL, abs=ATOL)
    assert store.get("mprop") == 3
    assert store.get("vmass") == pytest.approx(VMASS0 - store.get("fmasse"), rel=RTOL, abs=ATOL)
    assert store.get("fmassr") == pytest.approx(FMASS0 - store.get("fmasse"), rel=RTOL, abs=ATOL)

    fmasse = store.get("fmasse")
    mass_ratio = fmasse / FMASS0
    want_roll = MOI_ROLL_EXO_0 + (MOI_ROLL_EXO_1 - MOI_ROLL_EXO_0) * mass_ratio
    want_trans = MOI_TRANS_EXO_0 + (MOI_TRANS_EXO_1 - MOI_TRANS_EXO_0) * mass_ratio
    ibbb = np.array(store.get("IBBB"), dtype=float)
    assert ibbb[0, 0] == pytest.approx(want_roll, rel=RTOL, abs=ATOL)
    assert ibbb[1, 1] == pytest.approx(want_trans, rel=RTOL, abs=ATOL)
    assert ibbb[2, 2] == pytest.approx(want_trans, rel=RTOL, abs=ATOL)


def test_hyper6_mprop_4_constant_thrust():
    """mprop=4: spi=isp_fuel; thrust from input fuel_flow_rate; exo MOI."""
    vehicle, prop = _ready(mprop=4)
    store = vehicle.store
    prop.execute(vehicle, _ctx())

    want_thrust = ISP_FUEL * FUEL_FLOW_RATE * AGRAV
    assert store.get("thrust") == pytest.approx(want_thrust, rel=RTOL, abs=ATOL)
    assert store.get("spi") == pytest.approx(ISP_FUEL, rel=RTOL, abs=ATOL)
    assert store.get("mprop") == 4
    assert store.get("vmass") == pytest.approx(VMASS0 - store.get("fmasse"), rel=RTOL, abs=ATOL)

    fmasse = store.get("fmasse")
    mass_ratio = fmasse / FMASS0
    want_roll = MOI_ROLL_EXO_0 + (MOI_ROLL_EXO_1 - MOI_ROLL_EXO_0) * mass_ratio
    want_trans = MOI_TRANS_EXO_0 + (MOI_TRANS_EXO_1 - MOI_TRANS_EXO_0) * mass_ratio
    ibbb = np.array(store.get("IBBB"), dtype=float)
    assert ibbb[0, 0] == pytest.approx(want_roll, rel=RTOL, abs=ATOL)
    assert ibbb[1, 1] == pytest.approx(want_trans, rel=RTOL, abs=ATOL)
    assert ibbb[2, 2] == pytest.approx(want_trans, rel=RTOL, abs=ATOL)
