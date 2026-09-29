"""CRUISE5 INS (MINS 0/1/2) + HBGM — Fortran MODULE.FOR S4I / S4 / S4GYRO / S4ACCL / S4ALT."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import AGRAV, REARTH
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.stoch import gauss, seed
from cadac.vehicles.round3.cruise5.vehicle import Cruise5

RTOL = 1e-12
ATOL = 1e-14
ZEROS3 = (0.0, 0.0, 0.0)
INT_STEP = 0.05

# Fortran S4I: discard 100 FNGAUS(0,1) then draw 9 for Cholesky init.
_S4I_DISCARD = 100
_S4I_INIT_DRAWS = 9

SBEL = np.array([-10000.0, 17320.0, -150.0], dtype=float)
VBEL = np.array([136.0, -235.5, 0.0], dtype=float)
SWREL = np.array([0.0, 0.0, -100.0], dtype=float)
TBL = np.eye(3, dtype=float)
TLB = np.eye(3, dtype=float)
FSPB = np.array([0.0, 0.0, -AGRAV], dtype=float)
WBEB = np.array([0.01, -0.02, 0.005], dtype=float)
HBG = 150.0
HBGS = 140.0
BIASAL = 2.0
RANDAL = 0.5


def _ctx(sim_time=0.0, int_step=INT_STEP):
    return SimContext(sim_time, int_step, 0.0, 0.0, None, 0)


def _ins_module():
    from cadac.vehicles.round3.cruise5.ins import Cruise5Ins, PP0, _cholesky

    return Cruise5Ins(), PP0, _cholesky


def _defined(mins=1):
    ins, _, _ = _ins_module()
    vehicle = SimpleNamespace(store=StateStore())
    ins.define(vehicle)
    vehicle.store.set("mins", mins)
    return vehicle, ins


def _plant_truth(store: StateStore):
    for name, value, ftype, role, module in (
        ("SBEL", SBEL.copy(), "vec", "state", "newton"),
        ("VBEL", VBEL.copy(), "vec", "state", "newton"),
        ("SWREL", SWREL.copy(), "vec", "out", "seeker"),
        ("TBL", TBL.copy(), "mat", "out", "newton"),
        ("TLB", TLB.copy(), "mat", "out", "newton"),
        ("FSPB", FSPB.copy(), "vec", "out", "forces"),
        ("WBEB", WBEB.copy(), "vec", "out", "newton"),
        ("dvbe", float(np.linalg.norm(VBEL)), "real", "out", "newton"),
        ("hbg", HBG, "real", "out", "guidance"),
        ("hbgs", HBGS, "real", "out", "guidance"),
        ("mguidp", 1, "int", "diag", "guidance"),
        ("mseek", 0, "int", "data", "seeker"),
        ("mgps", 0, "int", "data", "gps"),
        ("GUSTTCL", np.zeros(3), "vec", "out", "gps"),
        ("GUVBEL", np.zeros(3), "vec", "out", "gps"),
        ("GURECEL", np.zeros(3), "vec", "out", "gps"),
        ("GUFSPB", np.zeros(3), "vec", "out", "gps"),
        ("GUWBEB", np.zeros(3), "vec", "out", "gps"),
        ("SWALC", np.zeros(3), "vec", "out", "seeker"),
    ):
        if name not in store:
            store.define(Field(name, value, ftype, role, module))
        store.set(name, value)


def _fortran_s4i_xx0(pp0, cholesky, frax=0.0):
    for _ in range(_S4I_DISCARD):
        gauss(0.0, 1.0)
    draws = np.array([gauss(0.0, 1.0) for _ in range(_S4I_INIT_DRAWS)], dtype=float)
    return cholesky(pp0) @ draws * (1.0 + frax)


def test_cruise5_has_ins_module():
    vehicle = Cruise5("c1", aero_deck={}, prop_deck={})
    names = [m.name for m in vehicle.modules]
    assert "ins" in names
    assert names.index("gps") < names.index("ins")
    from cadac.vehicles.round3.cruise5.ins import Cruise5Ins

    assert any(isinstance(m, Cruise5Ins) for m in vehicle.modules)


def test_mins0_initialize_copies_truth_and_sbwlc():
    vehicle, ins = _defined(mins=0)
    _plant_truth(vehicle.store)
    ins.initialize(vehicle, _ctx())
    store = vehicle.store
    np.testing.assert_allclose(store.get("SBELC"), SBEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VBELC"), VBEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBWLC"), SBEL - SWREL, rtol=RTOL, atol=ATOL)


def test_mins1_initialize_writes_cholesky_error_states():
    """Fortran S4I MINS=1: ESTTC/EVBE/RECE from Cholesky(PP0)@gauss*(1+FRAX)."""
    _, pp0, cholesky = _ins_module()
    seed(1234)
    vehicle, ins = _defined(mins=1)
    _plant_truth(vehicle.store)
    vehicle.store.set("frax", 0.0)
    ins.initialize(vehicle, _ctx())

    seed(1234)
    xx0 = _fortran_s4i_xx0(pp0, cholesky, frax=0.0)
    esttc = xx0[0:3]
    evbe = xx0[3:6]
    rece = xx0[6:9] * 0.001
    store = vehicle.store
    np.testing.assert_allclose(store.get("ESTTC"), esttc, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("EVBE"), evbe, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("RECE"), rece, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBELC"), esttc + SBEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VBELC"), evbe + VBEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("SBWLC"), esttc + SBEL - SWREL, rtol=RTOL, atol=ATOL
    )


def test_mins2_initialize_also_injects_errors():
    """Fortran S4I ELSE covers MINS=1 and MINS=2."""
    _, pp0, cholesky = _ins_module()
    seed(7)
    vehicle, ins = _defined(mins=2)
    _plant_truth(vehicle.store)
    vehicle.store.set("frax", 0.0)
    ins.initialize(vehicle, _ctx())

    seed(7)
    xx0 = _fortran_s4i_xx0(pp0, cholesky, frax=0.0)
    np.testing.assert_allclose(vehicle.store.get("ESTTC"), xx0[0:3], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("EVBE"), xx0[3:6], rtol=RTOL, atol=ATOL)


def test_mins0_execute_copies_truth_and_writes_hbgm():
    vehicle, ins = _defined(mins=0)
    _plant_truth(vehicle.store)
    vehicle.store.set("biasal", BIASAL)
    vehicle.store.set("randal", RANDAL)
    vehicle.store.set("mguidp", 1)
    ins.initialize(vehicle, _ctx())
    ins.execute(vehicle, _ctx())
    store = vehicle.store
    np.testing.assert_allclose(store.get("TBLC"), TBL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FSPCB"), FSPB, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("WBECB"), WBEB, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBELC"), SBEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VBELC"), VBEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBWLC"), SBEL - SWREL, rtol=RTOL, atol=ATOL)
    assert store.get("ehbe") == pytest.approx(BIASAL + RANDAL)
    assert store.get("hbgm") == pytest.approx(HBG + BIASAL + RANDAL)


def test_hbgm_uses_hbgs_when_mguidp_eq_2():
    vehicle, ins = _defined(mins=0)
    _plant_truth(vehicle.store)
    vehicle.store.set("biasal", BIASAL)
    vehicle.store.set("randal", RANDAL)
    vehicle.store.set("mguidp", 2)
    ins.initialize(vehicle, _ctx())
    ins.execute(vehicle, _ctx())
    assert vehicle.store.get("hbgm") == pytest.approx(HBGS + BIASAL + RANDAL)


def test_mins1_execute_propagates_velocity_error_derivative():
    """MINS=1 sets EVBED from specific-force error (non-zero with accel bias)."""
    vehicle, ins = _defined(mins=1)
    _plant_truth(vehicle.store)
    # Deterministic instruments: accel bias only → EFSPB = EBIASA
    for name in (
        "EWALKG",
        "EUNBG",
        "EMISG",
        "ESCALG",
        "EBIASG",
        "EWALKA",
        "EMISA",
        "ESCALA",
    ):
        vehicle.store.set(name, np.zeros(3))
    vehicle.store.set("EBIASA", np.array([1.0, 0.0, 0.0], dtype=float))
    vehicle.store.set("frax", -1.0)  # XX0=0 via (1+FRAX)=0 after Cholesky@gauss
    seed(1)
    ins.initialize(vehicle, _ctx())
    # Force zero error states after init (frax=-1 zeros XX0)
    vehicle.store.set("ESTTC", np.zeros(3))
    vehicle.store.set("EVBE", np.zeros(3))
    vehicle.store.set("RECE", np.zeros(3))
    vehicle.store.set("RECED", np.zeros(3))
    vehicle.store.set("EVBED", np.zeros(3))
    vehicle.store.set("ESTTCD", np.zeros(3))
    ins.execute(vehicle, _ctx(int_step=INT_STEP))
    # With RECE=0, TBLC=I, FSPCB=FSPB+EBIASA, EF=EBIASA → EVBED≈(1,0,0)
    np.testing.assert_allclose(
        vehicle.store.get("EVBED"), [1.0, 0.0, 0.0], rtol=1e-9, atol=1e-9
    )
    # EVBE integrated from 0 with EVBED=(1,0,0)
    assert abs(float(vehicle.store.get("EVBE")[0])) > ATOL


def test_mins2_doppler_holds_velocity_error():
    """Fortran MINS=2: EVBED not updated from EF → initial EVBE held (after integrate of 0)."""
    vehicle, ins = _defined(mins=2)
    _plant_truth(vehicle.store)
    for name in (
        "EWALKG",
        "EUNBG",
        "EMISG",
        "ESCALG",
        "EBIASG",
        "EWALKA",
        "EMISA",
        "ESCALA",
        "EBIASA",
    ):
        vehicle.store.set(name, np.zeros(3))
    vehicle.store.set("frax", -1.0)
    seed(2)
    ins.initialize(vehicle, _ctx())
    evbe0 = np.array([3.0, -4.0, 5.0], dtype=float)
    vehicle.store.set("ESTTC", np.zeros(3))
    vehicle.store.set("EVBE", evbe0.copy())
    vehicle.store.set("RECE", np.zeros(3))
    vehicle.store.set("RECED", np.zeros(3))
    vehicle.store.set("EVBED", np.zeros(3))
    vehicle.store.set("ESTTCD", np.zeros(3))
    ins.execute(vehicle, _ctx(int_step=INT_STEP))
    # EVBED stays zero (Doppler) so EVBE unchanged; ESTTC integrates EVBE
    np.testing.assert_allclose(
        vehicle.store.get("EVBED"), np.zeros(3), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(vehicle.store.get("EVBE"), evbe0, rtol=RTOL, atol=ATOL)


def test_mgps2_handshake_resets_to_one_and_applies_updates():
    """Fortran S4: MGPS=2 → MGPS=1; ESTTC/EVBE/RECE -= GUSTTCL/GUVBEL/GURECEL."""
    vehicle, ins = _defined(mins=1)
    _plant_truth(vehicle.store)
    for name in (
        "EWALKG",
        "EUNBG",
        "EMISG",
        "ESCALG",
        "EBIASG",
        "EWALKA",
        "EMISA",
        "ESCALA",
        "EBIASA",
    ):
        vehicle.store.set(name, np.zeros(3))
    vehicle.store.set("frax", -1.0)
    seed(3)
    ins.initialize(vehicle, _ctx())
    esttc0 = np.array([10.0, 20.0, 30.0], dtype=float)
    evbe0 = np.array([1.0, 2.0, 3.0], dtype=float)
    rece0 = np.array([0.01, 0.02, 0.03], dtype=float)
    gust = np.array([1.0, 2.0, 3.0], dtype=float)
    guv = np.array([0.1, 0.2, 0.3], dtype=float)
    gur = np.array([0.001, 0.002, 0.003], dtype=float)
    vehicle.store.set("ESTTC", esttc0.copy())
    vehicle.store.set("EVBE", evbe0.copy())
    vehicle.store.set("RECE", rece0.copy())
    vehicle.store.set("RECED", np.ones(3))
    vehicle.store.set("EVBED", np.ones(3))
    vehicle.store.set("ESTTCD", np.ones(3))
    vehicle.store.set("mgps", 2)
    vehicle.store.set("GUSTTCL", gust)
    vehicle.store.set("GUVBEL", guv)
    vehicle.store.set("GURECEL", gur)
    ins.execute(vehicle, _ctx(int_step=INT_STEP))
    store = vehicle.store
    assert store.get("mgps") == 1
    # After integrate then GPS subtract; with zero instruments RECE/EVBE/ESTTC
    # still change from integration, but derivatives must be zeroed after update.
    np.testing.assert_allclose(store.get("RECED"), np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("EVBED"), np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ESTTCD"), np.zeros(3), rtol=RTOL, atol=ATOL)
    # Exact subtract relative to post-integrate would need replaying integrate;
    # verify direction: updates reduce the planted states vs no-GPS baseline.
    vehicle2, ins2 = _defined(mins=1)
    _plant_truth(vehicle2.store)
    for name in (
        "EWALKG",
        "EUNBG",
        "EMISG",
        "ESCALG",
        "EBIASG",
        "EWALKA",
        "EMISA",
        "ESCALA",
        "EBIASA",
    ):
        vehicle2.store.set(name, np.zeros(3))
    vehicle2.store.set("frax", -1.0)
    seed(3)
    ins2.initialize(vehicle2, _ctx())
    vehicle2.store.set("ESTTC", esttc0.copy())
    vehicle2.store.set("EVBE", evbe0.copy())
    vehicle2.store.set("RECE", rece0.copy())
    vehicle2.store.set("RECED", np.ones(3))
    vehicle2.store.set("EVBED", np.ones(3))
    vehicle2.store.set("ESTTCD", np.ones(3))
    vehicle2.store.set("mgps", 0)
    ins2.execute(vehicle2, _ctx(int_step=INT_STEP))
    np.testing.assert_allclose(
        store.get("ESTTC"),
        np.asarray(vehicle2.store.get("ESTTC"), dtype=float) - gust,
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        store.get("EVBE"),
        np.asarray(vehicle2.store.get("EVBE"), dtype=float) - guv,
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        store.get("RECE"),
        np.asarray(vehicle2.store.get("RECE"), dtype=float) - gur,
        rtol=RTOL,
        atol=ATOL,
    )


def test_mgps_bias_does_not_rewrite_wbecb():
    """Fortran S4: GUWBEB corrects EWBEB for error ODEs only; WBECB stays S4GYRO output."""
    vehicle, ins = _defined(mins=1)
    _plant_truth(vehicle.store)
    for name in (
        "EWALKG",
        "EUNBG",
        "EMISG",
        "ESCALG",
        "EBIASG",
        "EWALKA",
        "EMISA",
        "ESCALA",
        "EBIASA",
    ):
        vehicle.store.set(name, np.zeros(3))
    vehicle.store.set("frax", -1.0)
    seed(4)
    ins.initialize(vehicle, _ctx())
    vehicle.store.set("ESTTC", np.zeros(3))
    vehicle.store.set("EVBE", np.zeros(3))
    vehicle.store.set("RECE", np.zeros(3))
    vehicle.store.set("RECED", np.zeros(3))
    vehicle.store.set("EVBED", np.zeros(3))
    vehicle.store.set("ESTTCD", np.zeros(3))
    guwbeb = np.array([0.1, -0.2, 0.05], dtype=float)
    vehicle.store.set("mgps", 1)
    vehicle.store.set("GUWBEB", guwbeb)
    vehicle.store.set("GUFSPB", np.array([0.5, 0.0, 0.0], dtype=float))
    # With zero instruments, S4GYRO EWBEB=0 → WBECB = WBEB before bias subtract.
    expected_wbecb = WBEB.copy()
    ins.execute(vehicle, _ctx(int_step=INT_STEP))
    np.testing.assert_allclose(
        vehicle.store.get("WBECB"), expected_wbecb, rtol=RTOL, atol=ATOL
    )
    # Bias did affect the error used in attitude ODEs (EWBEB stored is post-subtract).
    np.testing.assert_allclose(
        vehicle.store.get("EWBEB"), -guwbeb, rtol=RTOL, atol=ATOL
    )


def test_mins_unknown_raises():
    vehicle, ins = _defined(mins=3)
    _plant_truth(vehicle.store)
    with pytest.raises(ValueError, match="unknown mins"):
        ins.initialize(vehicle, _ctx())
