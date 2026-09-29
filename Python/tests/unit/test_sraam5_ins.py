"""SRAAM5 S4 INS — mins=1 error injection writes INS states (Fortran MODULE.FOR S4I)."""

from types import SimpleNamespace

import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.stoch import gauss, seed
from cadac.vehicles.flat5.sraam5.ins import PP0, Sraam5Ins, _cholesky

RTOL = 1e-12
ATOL = 1e-14
ZEROS3 = (0.0, 0.0, 0.0)

# Fortran S4I: discard 100 FNGAUS(0,1) then draw 9 for Cholesky init.
_S4I_DISCARD = 100
_S4I_INIT_DRAWS = 9

SBEL = np.array([100.0, -50.0, -7000.0], dtype=float)
VBEL = np.array([250.0, 10.0, -5.0], dtype=float)
HBE = 7000.0


def _ctx(int_step=0.0123):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _defined(mins=1):
    vehicle = SimpleNamespace(store=StateStore())
    ins = Sraam5Ins()
    ins.define(vehicle)
    vehicle.store.set("mins", mins)
    return vehicle, ins


def _plant_truth(store, sbel=SBEL, vbel=VBEL, hbe=HBE):
    for name, value, ftype in (
        ("SBEL", sbel, "vec"),
        ("VBEL", vbel, "vec"),
        ("hbe", hbe, "real"),
    ):
        if name not in store.names():
            store.define(Field(name, value, ftype, "out", "newton"))
        store.set(name, value)
    return {"SBEL": np.asarray(sbel, dtype=float), "VBEL": np.asarray(vbel, dtype=float)}


def _fortran_s4i_xx0(frax=0.0):
    """Replay MODULE.FOR S4I FNGAUS discard + Cholesky injection."""
    for _ in range(_S4I_DISCARD):
        gauss(0.0, 1.0)
    draws = np.array([gauss(0.0, 1.0) for _ in range(_S4I_INIT_DRAWS)], dtype=float)
    return _cholesky(PP0) @ draws * (1.0 + frax)


def test_name_is_ins():
    assert Sraam5Ins().name == "ins"


def test_mins1_initialize_writes_cholesky_error_states():
    """Fortran S4I mins=1: ESTTC/EVBE/RECE from Cholesky(PP0)@gauss*(1+FRAX); SBELC=SBEL+ESTTC."""
    seed(1234)
    vehicle, ins = _defined(mins=1)
    planted = _plant_truth(vehicle.store)
    vehicle.store.set("frax", 0.0)
    ins.initialize(vehicle, _ctx())

    seed(1234)
    xx0 = _fortran_s4i_xx0(frax=0.0)
    esttc = xx0[0:3]
    evbe = xx0[3:6]
    rece = xx0[6:9] * 0.001
    store = vehicle.store
    np.testing.assert_allclose(store.get("ESTTC"), esttc, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("EVBE"), evbe, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("RECE"), rece, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("SBELC"), esttc + planted["SBEL"], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        store.get("VBELC"), evbe + planted["VBEL"], rtol=RTOL, atol=ATOL
    )


def test_mins1_initialize_frax_scales_error_states():
    seed(99)
    vehicle, ins = _defined(mins=1)
    planted = _plant_truth(vehicle.store)
    vehicle.store.set("frax", 10.0)
    ins.initialize(vehicle, _ctx())

    seed(99)
    xx0 = _fortran_s4i_xx0(frax=10.0)
    store = vehicle.store
    np.testing.assert_allclose(store.get("ESTTC"), xx0[0:3], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("SBELC"), xx0[0:3] + planted["SBEL"], rtol=RTOL, atol=ATOL
    )


def test_mins0_initialize_copies_sbel_vbel():
    vehicle, ins = _defined(mins=0)
    planted = _plant_truth(vehicle.store)
    ins.initialize(vehicle, _ctx())
    np.testing.assert_allclose(
        vehicle.store.get("SBELC"), planted["SBEL"], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("VBELC"), planted["VBEL"], rtol=RTOL, atol=ATOL
    )


def test_mins_unknown_raises():
    vehicle, ins = _defined(mins=2)
    _plant_truth(vehicle.store)
    with pytest.raises(ValueError, match="unknown mins"):
        ins.initialize(vehicle, _ctx())
