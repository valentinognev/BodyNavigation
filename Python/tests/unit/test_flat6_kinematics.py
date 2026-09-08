import math
from types import SimpleNamespace

import numpy as np

from cadac.constants import DEG, EPS, PI, RAD
from cadac.eom.flat6 import Flat6Kinematics
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import incidence_angles, quat_to_dcm


def _vehicle():
    store = StateStore()
    vehicle = SimpleNamespace(store=store)
    kin = Flat6Kinematics()
    kin.define(vehicle)
    return vehicle, kin


def _plant(store, vbal, wbeb=(0.0, 0.0, 0.0), dvba=None):
    vbal = np.asarray(vbal, dtype=float)
    if dvba is None:
        dvba = float(np.linalg.norm(vbal))
    if "VBAL" not in store.names():
        store.define(Field("VBAL", (0.0, 0.0, 0.0), "vec", "out", "environment"))
    if "dvba" not in store.names():
        store.define(Field("dvba", 0.0, "real", "out", "environment"))
    if "WBEB" not in store.names():
        store.define(Field("WBEB", (0.0, 0.0, 0.0), "vec", "out", "euler"))
    store.set("VBAL", vbal)
    store.set("WBEB", np.asarray(wbeb, dtype=float))
    store.set("dvba", float(dvba))


def _vbeb(alpha0x, beta0x, dvbe):
    salp = math.sin(alpha0x * RAD)
    calp = math.cos(alpha0x * RAD)
    sbet = math.sin(beta0x * RAD)
    cbet = math.cos(beta0x * RAD)
    return np.array(
        [calp * cbet * dvbe, sbet * dvbe, salp * cbet * dvbe], dtype=float
    )


def _cpp_mat3tr(psi, tht, phi):
    amat = np.zeros((3, 3))
    spsi = math.sin(psi)
    cpsi = math.cos(psi)
    stht = math.sin(tht)
    ctht = math.cos(tht)
    sphi = math.sin(phi)
    cphi = math.cos(phi)
    amat[0, 0] = cpsi * ctht
    amat[1, 0] = cpsi * stht * sphi - spsi * cphi
    amat[2, 0] = cpsi * stht * cphi + spsi * sphi
    amat[0, 1] = spsi * ctht
    amat[1, 1] = spsi * stht * sphi + cpsi * cphi
    amat[2, 1] = spsi * stht * cphi - cpsi * sphi
    amat[0, 2] = -stht
    amat[1, 2] = ctht * sphi
    amat[2, 2] = ctht * cphi
    return amat


def _cpp_quat(psiblx, thtblx, phiblx):
    spsi = math.sin(psiblx / (2.0 * DEG))
    cpsi = math.cos(psiblx / (2.0 * DEG))
    stht = math.sin(thtblx / (2.0 * DEG))
    ctht = math.cos(thtblx / (2.0 * DEG))
    sphi = math.sin(phiblx / (2.0 * DEG))
    cphi = math.cos(phiblx / (2.0 * DEG))
    q0 = cpsi * ctht * cphi + spsi * stht * sphi
    q1 = cpsi * ctht * sphi - spsi * stht * cphi
    q2 = cpsi * stht * cphi + spsi * ctht * sphi
    q3 = -cpsi * stht * sphi + spsi * ctht * cphi
    return q0, q1, q2, q3


def test_incidence_phip_eps_negative_vbab3():
    alpha, beta, alpp, phip = incidence_angles(
        np.array([1.0, 0.5 * EPS, -1.0]), 2.0
    )
    # |vbab2| < EPS → PI when vbab3 < 0
    assert phip == PI


def test_incidence_phip_zero_when_both_zero():
    _, _, _, phip = incidence_angles(np.array([1.0, 0.0, 0.0]), 1.0)
    assert phip == 0.0


def test_name_is_kinematics():
    assert Flat6Kinematics().name == "kinematics"


def test_ck_default_is_50():
    vehicle, _kin = _vehicle()
    assert vehicle.store.get("ck") == 50.0


def test_define_does_not_register_euler_newton_or_plane_fields():
    store = StateStore()
    Flat6Kinematics().define(SimpleNamespace(store=store))
    for name in ("WBEB", "dvba", "VBAL", "hbe", "VBEL", "VBEB", "trcode", "tralppx"):
        assert name not in store.names()


def test_initialize_does_not_write_alphax():
    vehicle, kin = _vehicle()
    store = vehicle.store
    store.set("thtblx", 1.0)
    kin.initialize(vehicle, None)
    assert store.get("alphax") == 0.0
    assert np.all(np.isfinite(store.get("TBL")))
    assert not np.allclose(store.get("TBL"), 0.0)


def test_initialize_quats_and_tbl_from_euler_degrees():
    vehicle, kin = _vehicle()
    store = vehicle.store
    store.set("psiblx", 10.0)
    store.set("thtblx", 1.0)
    store.set("phiblx", -5.0)
    kin.initialize(vehicle, None)
    q0, q1, q2, q3 = _cpp_quat(10.0, 1.0, -5.0)
    assert store.get("q0") == q0
    assert store.get("q1") == q1
    assert store.get("q2") == q2
    assert store.get("q3") == q3
    np.testing.assert_array_equal(
        store.get("TBL"),
        _cpp_mat3tr(10.0 / DEG, 1.0 / DEG, -5.0 / DEG),
    )


def test_execute_after_init_tbl_finite_and_alphax_approx_1():
    vehicle, kin = _vehicle()
    store = vehicle.store
    store.set("psiblx", 0.0)
    store.set("thtblx", 1.0)
    store.set("phiblx", 0.0)
    kin.initialize(vehicle, None)
    vbeb = _vbeb(alpha0x=1.0, beta0x=0.0, dvbe=180.0)
    vbal = store.get("TBL").T @ vbeb
    _plant(store, vbal, wbeb=(0.0, 0.0, 0.0), dvba=float(np.linalg.norm(vbal)))
    kin.execute(vehicle, SimpleNamespace(int_step=0.001))
    assert np.all(np.isfinite(store.get("TBL")))
    np.testing.assert_allclose(store.get("alphax"), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        store.get("TBL"),
        quat_to_dcm(store.get("q0"), store.get("q1"), store.get("q2"), store.get("q3")),
        rtol=1e-12,
        atol=1e-14,
    )


def test_execute_zero_rates_identity_uses_stored_slope_integrate():
    vehicle, kin = _vehicle()
    store = vehicle.store
    kin.initialize(vehicle, None)
    _plant(store, (180.0, 0.0, 0.0), wbeb=(1.0, 0.0, 0.0), dvba=180.0)
    kin.execute(vehicle, SimpleNamespace(int_step=0.001))
    assert store.get("q0") == 1.0
    assert store.get("q1") == 0.00025
    assert store.get("q2") == 0.0
    assert store.get("q3") == 0.0
    assert store.get("q0d") == 0.0
    assert store.get("q1d") == 0.5
    assert store.get("q2d") == 0.0
    assert store.get("q3d") == 0.0


def test_execute_second_step_uses_stored_quaternion_derivatives():
    vehicle, kin = _vehicle()
    store = vehicle.store
    kin.initialize(vehicle, None)
    _plant(store, (180.0, 0.0, 0.0), wbeb=(1.0, 0.0, 0.0), dvba=180.0)
    ctx = SimpleNamespace(int_step=0.001)
    kin.execute(vehicle, ctx)
    q0, q1, q2, q3 = (store.get(n) for n in ("q0", "q1", "q2", "q3"))
    q0d, q1d, q2d, q3d = (store.get(n) for n in ("q0d", "q1d", "q2d", "q3d"))
    ck = store.get("ck")
    quat_metric = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3
    erq = 1.0 - quat_metric
    pp, qq, rr = 1.0, 0.0, 0.0
    new_q0d = 0.5 * (-pp * q1 - qq * q2 - rr * q3) + ck * erq * q0
    new_q1d = 0.5 * (pp * q0 + rr * q2 - qq * q3) + ck * erq * q1
    new_q2d = 0.5 * (qq * q0 - rr * q1 + pp * q3) + ck * erq * q2
    new_q3d = 0.5 * (rr * q0 + qq * q1 - pp * q2) + ck * erq * q3
    want_q0 = integrate(new_q0d, q0d, q0, ctx.int_step)
    want_q1 = integrate(new_q1d, q1d, q1, ctx.int_step)
    want_q2 = integrate(new_q2d, q2d, q2, ctx.int_step)
    want_q3 = integrate(new_q3d, q3d, q3, ctx.int_step)
    kin.execute(vehicle, ctx)
    assert store.get("q0") == want_q0
    assert store.get("q1") == want_q1
    assert store.get("q2") == want_q2
    assert store.get("q3") == want_q3
    assert store.get("q0d") == new_q0d
    assert store.get("q1d") == new_q1d


def test_etbl_is_diagonal_orthogonality_not_first_row():
    vehicle, kin = _vehicle()
    store = vehicle.store
    kin.initialize(vehicle, None)
    _plant(store, (180.0, 0.0, 0.0), wbeb=(0.0, 0.0, 0.0), dvba=180.0)
    kin.execute(vehicle, SimpleNamespace(int_step=0.001))
    np.testing.assert_allclose(store.get("etbl"), 0.0, atol=1e-15)
    tlb = store.get("TLB")
    tbl = store.get("TBL")
    ubl = tlb @ tbl
    want = math.sqrt(
        (ubl[0, 0] - 1.0) ** 2 + (ubl[1, 1] - 1.0) ** 2 + (ubl[2, 2] - 1.0) ** 2
    )
    np.testing.assert_allclose(store.get("etbl"), want, atol=1e-15)
    row_bug = math.sqrt(
        (ubl[0, 0] - 1.0) ** 2 + (ubl[0, 1] - 1.0) ** 2 + (ubl[0, 2] - 1.0) ** 2
    )
    assert abs(store.get("etbl") - row_bug) > 0.5


def test_beta_from_sideslip_vbeb():
    vehicle, kin = _vehicle()
    store = vehicle.store
    kin.initialize(vehicle, None)
    vbeb = _vbeb(alpha0x=0.0, beta0x=5.0, dvbe=180.0)
    _plant(store, store.get("TBL").T @ vbeb)
    kin.execute(vehicle, SimpleNamespace(int_step=0.001))
    np.testing.assert_allclose(store.get("betax"), 5.0, atol=1e-6)
    np.testing.assert_allclose(store.get("alphax"), 0.0, atol=1e-6)


def test_phip_forced_to_0_or_pi_when_vbab2_below_eps():
    vehicle, kin = _vehicle()
    store = vehicle.store
    kin.initialize(vehicle, None)
    _plant(store, (180.0, 1e-11, 1.0), dvba=180.0)
    kin.execute(vehicle, SimpleNamespace(int_step=0.001))
    np.testing.assert_allclose(store.get("phipx"), 0.0, atol=1e-12)
    _plant(store, (180.0, 1e-11, -1.0), dvba=180.0)
    kin.execute(vehicle, SimpleNamespace(int_step=0.001))
    np.testing.assert_allclose(store.get("phipx"), PI * DEG, atol=1e-12)


def test_alpp_clamps_dum_with_cadac_sign():
    vehicle, kin = _vehicle()
    store = vehicle.store
    kin.initialize(vehicle, None)
    _plant(store, (200.0, 0.0, 0.0), dvba=180.0)
    kin.execute(vehicle, SimpleNamespace(int_step=0.001))
    np.testing.assert_allclose(store.get("alppx"), 0.0, atol=1e-12)


def test_pitch_90_uses_gimbal_lock_branch():
    vehicle, kin = _vehicle()
    store = vehicle.store
    store.set("q0", 0.6)
    store.set("q2", 1.0)
    _plant(store, (180.0, 0.0, 0.0), dvba=180.0)
    kin.execute(vehicle, SimpleNamespace(int_step=0.001))
    assert np.isfinite(store.get("thtblx"))
    assert np.isfinite(store.get("psiblx"))
    assert np.isfinite(store.get("phiblx"))
    np.testing.assert_allclose(store.get("thtbl"), PI / 2.0, atol=1e-12)
    np.testing.assert_allclose(store.get("thtblx"), DEG * PI / 2.0, atol=1e-12)


def test_cpsi_clamp_uses_cadac_sign_not_numpy_sign():
    vehicle, kin = _vehicle()
    store = vehicle.store
    kin.initialize(vehicle, None)
    _plant(store, (180.0, 0.0, 0.0), dvba=180.0)
    kin.execute(vehicle, SimpleNamespace(int_step=0.001))
    want = DEG * math.acos((1.0 - EPS) * 1.0) * 1
    np.testing.assert_allclose(store.get("psiblx"), want, atol=1e-12)


def test_skip_trcode_when_plane_names_absent():
    vehicle, kin = _vehicle()
    store = vehicle.store
    kin.initialize(vehicle, None)
    _plant(store, (180.0, 0.0, 0.0), dvba=180.0)
    kin.execute(vehicle, SimpleNamespace(int_step=0.001))
    assert "trcode" not in store.names()


def test_trcode_5_when_alphax_exceeds_tralppx():
    vehicle, kin = _vehicle()
    store = vehicle.store
    kin.initialize(vehicle, None)
    store.define(Field("trcode", 0.0, "real", "init", "aerodynamics"))
    store.define(Field("tralppx", 16.0, "real", "data", "aerodynamics"))
    store.define(Field("tralpnx", -6.0, "real", "data", "aerodynamics"))
    store.define(Field("trbetx", 5.0, "real", "data", "aerodynamics"))
    vbeb = _vbeb(alpha0x=20.0, beta0x=0.0, dvbe=180.0)
    _plant(store, store.get("TBL").T @ vbeb)
    kin.execute(vehicle, SimpleNamespace(int_step=0.001))
    assert store.get("trcode") == 5.0


def test_trcode_6_when_alphax_below_tralpnx():
    vehicle, kin = _vehicle()
    store = vehicle.store
    kin.initialize(vehicle, None)
    store.define(Field("trcode", 0.0, "real", "init", "aerodynamics"))
    store.define(Field("tralppx", 16.0, "real", "data", "aerodynamics"))
    store.define(Field("tralpnx", -6.0, "real", "data", "aerodynamics"))
    store.define(Field("trbetx", 5.0, "real", "data", "aerodynamics"))
    vbeb = _vbeb(alpha0x=-10.0, beta0x=0.0, dvbe=180.0)
    _plant(store, store.get("TBL").T @ vbeb)
    kin.execute(vehicle, SimpleNamespace(int_step=0.001))
    assert store.get("trcode") == 6.0


def test_trcode_7_when_abs_betax_exceeds_trbetx():
    vehicle, kin = _vehicle()
    store = vehicle.store
    kin.initialize(vehicle, None)
    store.define(Field("trcode", 0.0, "real", "init", "aerodynamics"))
    store.define(Field("tralppx", 16.0, "real", "data", "aerodynamics"))
    store.define(Field("tralpnx", -6.0, "real", "data", "aerodynamics"))
    store.define(Field("trbetx", 5.0, "real", "data", "aerodynamics"))
    vbeb = _vbeb(alpha0x=0.0, beta0x=8.0, dvbe=180.0)
    _plant(store, store.get("TBL").T @ vbeb)
    kin.execute(vehicle, SimpleNamespace(int_step=0.001))
    assert store.get("trcode") == 7.0
