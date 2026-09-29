"""SRAAM6 Fortran S2 AI acquisition radar — NTAG→MNAV 2→3 + bias/noise."""

from types import SimpleNamespace

import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import cart_from_pol, polar_from_cart
from cadac.vehicles.flat6.sraam6.ai_radar import Sraam6AiRadar

RTOL = 1e-12
ATOL = 1e-12

# Target STEL / shooter SSEL (Fortran ST1EL / ST2EL)
STEL = np.array([0.0, 0.0, -5000.0], dtype=float)
SSEL = np.array([10000.0, 2000.0, -5000.0], dtype=float)
VTEL = np.array([240.0, 0.0, 0.0], dtype=float)
EVT1EL = np.array([2.0, -1.0, 0.5], dtype=float)
BIASTD = 15.0
RANDTD = 3.0
BIASTA = 0.02
RANDTA = 0.003
BIASTE = -0.008
RANDTE = 0.0015
DTIMTU = 0.05
DTIMUP = 0.5


def _ctx(sim_time, int_step=0.001):
    return SimContext(
        sim_time=sim_time,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _ready(**params):
    vehicle = SimpleNamespace(store=StateStore(), health=1)
    radar = Sraam6AiRadar()
    radar.define(vehicle)
    store = vehicle.store
    defaults = {
        "ntag": 1,
        "dtimtu": DTIMTU,
        "dtimup": DTIMUP,
        "biastd": BIASTD,
        "randtd": RANDTD,
        "biasta": BIASTA,
        "randta": RANDTA,
        "biaste": BIASTE,
        "randte": RANDTE,
        "EVT1EL": EVT1EL.copy(),
        "STEL": STEL.copy(),
        "SSEL": SSEL.copy(),
        "VTEL": VTEL.copy(),
        "mguid": 3,
        "mnav": 0,
    }
    defaults.update(params)
    for name, value in defaults.items():
        if name not in store:
            if isinstance(value, np.ndarray):
                store.define(Field(name, value, "vec", "data", "plant"))
            elif isinstance(value, int):
                store.define(Field(name, value, "int", "data", "plant"))
            else:
                store.define(Field(name, value, "real", "data", "plant"))
        store.set(name, value)
    return vehicle, radar


def _expected_measurement():
    st2t1l = SSEL - STEL
    dt2t1, azt2t1, elt2t1 = polar_from_cart(st2t1l)
    dt2t1r = float(dt2t1) + BIASTD + RANDTD
    azt2tr = float(azt2t1) + BIASTA + RANDTA
    elt2tr = float(elt2t1) + BIASTE + RANDTE
    st2t1l = cart_from_pol(dt2t1r, azt2tr, elt2tr)
    st1cel = SSEL - st2t1l
    vt1cel = VTEL + EVT1EL
    return st1cel, vt1cel


def test_ntag_0_is_noop():
    vehicle, radar = _ready(ntag=0)
    radar.execute(vehicle, _ctx(0.0))
    assert vehicle.store.get("mnav") == 0
    assert vehicle.store.get("ntag") == 0


def test_ntag_1_starts_ai_and_sets_mnav_2_at_measurement():
    vehicle, radar = _ready(ntag=1)
    radar.execute(vehicle, _ctx(0.0))
    store = vehicle.store
    assert store.get("ntag") == 2
    assert store.get("mnav") == 2
    st1cel, vt1cel = _expected_measurement()
    assert np.allclose(store.get("ST1CEL"), st1cel, rtol=RTOL, atol=ATOL)
    assert np.allclose(store.get("VT1CEL"), vt1cel, rtol=RTOL, atol=ATOL)
    assert not np.allclose(store.get("ST1CEL"), STEL, rtol=RTOL, atol=ATOL)
    assert not np.allclose(store.get("VT1CEL"), VTEL, rtol=RTOL, atol=ATOL)


def test_after_dtimtu_mnav_becomes_3_and_overwrites_stel_vtel():
    """Update epoch: MNAV=3 and STEL/VTEL get biased AI stores for guidance."""
    vehicle, radar = _ready(ntag=1)
    radar.execute(vehicle, _ctx(0.0))
    assert vehicle.store.get("mnav") == 2
    radar.execute(vehicle, _ctx(DTIMTU))
    store = vehicle.store
    assert store.get("mnav") == 3
    st1cel, vt1cel = _expected_measurement()
    assert np.allclose(store.get("ST1CEL"), st1cel, rtol=RTOL, atol=ATOL)
    assert np.allclose(store.get("VT1CEL"), vt1cel, rtol=RTOL, atol=ATOL)
    assert np.allclose(store.get("STEL"), st1cel, rtol=RTOL, atol=ATOL)
    assert np.allclose(store.get("VTEL"), vt1cel, rtol=RTOL, atol=ATOL)


def test_mguid_6_skips_ai_radar():
    vehicle, radar = _ready(ntag=1, mguid=6)
    radar.execute(vehicle, _ctx(0.0))
    assert vehicle.store.get("mnav") == 0
    assert vehicle.store.get("ntag") == 1
