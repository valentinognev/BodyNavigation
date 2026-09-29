"""SRAAM5 A2 propulsion — thrust/mass vs time table row (Fortran MODULE.FOR)."""

import numpy as np

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat5.sraam5.propulsion import Sraam5Propulsion

RTOL = 1e-12
ATOL = 1e-14

# Exact PROTIM / THRUST / WGT knot from Fortran A2 DATA (index 5).
T_ROW = 1.076
THRUST_SL_LBF = 7970.0
WGT_LB = 171.10
XCGIN_IN = 55.87

# SI path (inlar1 OPTMET=1); PRESS = English+SI sea-level constant so ALTCOR=0.
OPTMET = 1.0
PRESS_SL = 2116.0 + 99208.0 * OPTMET  # 101324 Pa


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(0.0, 0.001, 0.0, 0.0, None, 0)


def _ready(*, time=T_ROW, optmet=OPTMET, press=PRESS_SL):
    vehicle = _Vehicle()
    prop = Sraam5Propulsion()
    prop.define(vehicle)
    store = vehicle.store
    store.define(Field("time", time, "real", "exec", "kinematics"))
    store.define(Field("optmet", optmet, "real", "data", "environment"))
    store.define(Field("press", press, "real", "out", "environment"))
    return vehicle, prop


def _expected_burn(*, thrust_sl_lbf, wgt_lb, optmet, press):
    aexit = 0.1351 * (1.0 - 0.9071 * optmet)
    amass = wgt_lb * (1.0 + 13.59 * optmet) / 32.174
    altcor = ((2116.0 + 99208.0 * optmet) - press) * aexit
    fthalt = thrust_sl_lbf * (1.0 + 3.45 * optmet) + altcor
    return fthalt, amass


def test_a2_table_row_thrust_and_mass_match_fortran():
    """At PROTIM knot T=1.076, FTHALT/AMASS follow A2 TABLE + altitude formulas."""
    vehicle, prop = _ready(time=T_ROW)
    prop.execute(vehicle, _ctx())
    store = vehicle.store
    want_fthalt, want_amass = _expected_burn(
        thrust_sl_lbf=THRUST_SL_LBF,
        wgt_lb=WGT_LB,
        optmet=OPTMET,
        press=PRESS_SL,
    )
    assert store.get("mprop") == 1
    np.testing.assert_allclose(store.get("fthalt"), want_fthalt, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("amass"), want_amass, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("xcgin"), XCGIN_IN, rtol=RTOL, atol=ATOL)


def test_a2_past_burnout_mprop_off_zero_thrust():
    vehicle, prop = _ready(time=2.69)
    prop.execute(vehicle, _ctx())
    assert vehicle.store.get("mprop") == 0
    np.testing.assert_allclose(vehicle.store.get("fthalt"), 0.0, rtol=RTOL, atol=ATOL)
