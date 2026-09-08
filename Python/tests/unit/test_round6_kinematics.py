import math
from types import SimpleNamespace

import numpy as np

from cadac.constants import DEG, EPS, PI, RAD, WEII3
from cadac.eom.round6 import Round6Kinematics
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import cadac_inverse, cadac_matmul, mat3tr
from cadac.math.wgs84 import cad_tdi84, cad_tgi84


def _ctx(sim_time=0.0, int_step=0.01, event_time=0.0, out_fact=0.0):
    return SimContext(
        sim_time=sim_time,
        int_step=int_step,
        event_time=event_time,
        out_fact=out_fact,
        combus=None,
        vehicle_slot=0,
    )


def _vehicle():
    store = StateStore()
    vehicle = SimpleNamespace(store=store)
    kin = Round6Kinematics()
    kin.define(vehicle)
    return vehicle, kin


def _plant(
    store,
    *,
    lonx=10.0,
    latx=10.0,
    alt=10000.0,
    wbib=(0.0, 0.0, 0.0),
    vbed=(1000.0, 0.0, 0.0),
    vaed=(0.0, 0.0, 0.0),
    vbii=(1000.0, 0.0, 0.0),
    dvba=None,
):
    vbed = np.asarray(vbed, dtype=float)
    vaed = np.asarray(vaed, dtype=float)
    if dvba is None:
        dvba = float(np.linalg.norm(vbed - vaed))
    for name, value, ftype, role, module in (
        ("lonx", lonx, "real", "init/diag", "newton"),
        ("latx", latx, "real", "init/diag", "newton"),
        ("alt", alt, "real", "init/out", "newton"),
        ("WBIB", wbib, "vec", "out", "euler"),
        ("VBED", vbed, "vec", "state", "newton"),
        ("VAED", vaed, "vec", "out", "environment"),
        ("VBII", vbii, "vec", "state", "newton"),
        ("dvba", dvba, "real", "out", "environment"),
    ):
        if name not in store.names():
            store.define(Field(name, value, ftype, role, module))
        store.set(name, value)


def _skew(vec):
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def _cadac_matmul(a, b):
    """C++ Matrix::operator* row-major ijk (no BLAS)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    squeeze = False
    if b.ndim == 1:
        b = b.reshape(-1, 1)
        squeeze = True
    nrow, nmid = a.shape
    ncol = b.shape[1]
    result = np.zeros((nrow, ncol), dtype=float)
    for i in range(nrow * ncol):
        r = i // ncol
        c = i % ncol
        acc = 0.0
        for k in range(nmid):
            acc += a[r, k] * b[k, c]
        result[r, c] = acc
    if squeeze:
        return result.reshape(nrow)
    return result


def _cpp_step(tbi, tbid, wbib, int_step):
    tbid_new = _cadac_matmul(-_skew(wbib), tbi)
    tbi = integrate(tbid_new, tbid, tbi, int_step)
    tbid = tbid_new
    unit = np.eye(3)
    ee = unit - _cadac_matmul(tbi, tbi.T.copy())
    tbi = tbi + _cadac_matmul(ee, tbi) * 0.5
    e1 = ee[0, 0]
    e2 = ee[1, 1]
    e3 = ee[2, 2]
    ortho_error = math.sqrt(e1 * e1 + e2 * e2 + e3 * e3)
    return tbi, tbid, ortho_error


def test_name_is_kinematics():
    assert Round6Kinematics().name == "kinematics"


def test_define_does_not_register_newton_euler_or_hyper_fields():
    store = StateStore()
    Round6Kinematics().define(SimpleNamespace(store=store))
    for name in (
        "lonx",
        "latx",
        "alt",
        "SBII",
        "VBED",
        "WBIB",
        "VAED",
        "dvba",
        "VBII",
        "ck",
        "trcode",
    ):
        assert name not in store.names()


def test_define_registers_def_kinematics_fields():
    store = StateStore()
    Round6Kinematics().define(SimpleNamespace(store=store))
    for name in (
        "time",
        "event_time",
        "int_step_new",
        "out_step_fact",
        "TBD",
        "TBI",
        "TBID",
        "ortho_error",
        "psibd",
        "thtbd",
        "phibd",
        "psibdx",
        "thtbdx",
        "phibdx",
        "alppx",
        "phipx",
        "alphax",
        "betax",
        "alphaix",
        "betaix",
    ):
        assert name in store.names()
    np.testing.assert_allclose(store.get("TBD"), np.zeros((3, 3)))
    np.testing.assert_allclose(store.get("TBI"), np.zeros((3, 3)))
    np.testing.assert_allclose(store.get("TBID"), np.zeros((3, 3)))


def test_initialize_climb_ics_tbd_finite():
    vehicle, kin = _vehicle()
    store = vehicle.store
    store.set("thtbdx", 2.5)
    _plant(store, lonx=10.0, latx=10.0, alt=10000.0)
    kin.initialize(vehicle, _ctx())
    tbd = store.get("TBD")
    assert np.all(np.isfinite(tbd))
    assert not np.allclose(tbd, 0.0)


def test_initialize_tbd_tbi_from_euler_degrees_and_tdi84():
    vehicle, kin = _vehicle()
    store = vehicle.store
    store.set("psibdx", 0.0)
    store.set("thtbdx", 2.5)
    store.set("phibdx", 0.0)
    _plant(store, lonx=10.0, latx=10.0, alt=10000.0)
    ctx = _ctx(sim_time=0.0, int_step=0.01)
    kin.initialize(vehicle, ctx)
    want_tbd = mat3tr(0.0 * RAD, 2.5 * RAD, 0.0 * RAD)
    want_tdi = cad_tdi84(10.0 * RAD, 10.0 * RAD, 10000.0, 0.0)
    np.testing.assert_allclose(store.get("TBD"), want_tbd, rtol=1e-12)
    np.testing.assert_allclose(store.get("TBD")[0, 2], -math.sin(2.5 * RAD), rtol=1e-12)
    np.testing.assert_array_equal(store.get("TBI"), _cadac_matmul(want_tbd, want_tdi))


def test_initialize_seeds_time_and_int_step_new_from_ctx():
    vehicle, kin = _vehicle()
    store = vehicle.store
    _plant(store)
    ctx = _ctx(sim_time=1.25, int_step=0.01)
    kin.initialize(vehicle, ctx)
    np.testing.assert_allclose(store.get("time"), 1.25, rtol=1e-12)
    np.testing.assert_allclose(store.get("int_step_new"), 0.01, rtol=1e-12)


def test_initialize_does_not_write_alphax():
    vehicle, kin = _vehicle()
    store = vehicle.store
    store.set("thtbdx", 2.5)
    _plant(store)
    kin.initialize(vehicle, _ctx())
    assert store.get("alphax") == 0.0


def test_execute_zero_wbib_tbd_stays_orthonormal():
    vehicle, kin = _vehicle()
    store = vehicle.store
    store.set("thtbdx", 2.5)
    _plant(store, lonx=10.0, latx=10.0, alt=10000.0, wbib=(0.0, 0.0, 0.0))
    ctx = _ctx(sim_time=0.0, int_step=0.01)
    kin.initialize(vehicle, ctx)
    kin.execute(vehicle, ctx)
    tbd = store.get("TBD")
    assert np.all(np.isfinite(tbd))
    np.testing.assert_allclose(tbd @ tbd.T, np.eye(3), atol=1e-12)
    assert store.get("ortho_error") < 1e-12


def test_execute_sets_ctx_int_step_and_out_fact():
    vehicle, kin = _vehicle()
    store = vehicle.store
    _plant(store)
    kin.initialize(vehicle, _ctx(int_step=0.01))
    store.set("int_step_new", 0.02)
    store.set("out_step_fact", 0.5)
    ctx = _ctx(int_step=0.01, out_fact=0.0)
    kin.execute(vehicle, ctx)
    np.testing.assert_allclose(ctx.int_step, 0.02, rtol=1e-12)
    np.testing.assert_allclose(ctx.out_fact, 0.5, rtol=1e-12)


def test_execute_event_time_from_ctx():
    vehicle, kin = _vehicle()
    store = vehicle.store
    _plant(store)
    ctx = _ctx(sim_time=3.0, int_step=0.01, event_time=0.4)
    kin.initialize(vehicle, ctx)
    kin.execute(vehicle, ctx)
    np.testing.assert_allclose(store.get("time"), 3.0, rtol=1e-12)
    np.testing.assert_allclose(store.get("event_time"), 0.4, rtol=1e-12)


def test_execute_stored_slope_integrate():
    vehicle, kin = _vehicle()
    store = vehicle.store
    store.set("thtbdx", 2.5)
    wbib = np.array([0.1, 0.0, 0.0])
    _plant(store, wbib=wbib)
    ctx = _ctx(int_step=0.01)
    kin.initialize(vehicle, ctx)
    tbi0 = store.get("TBI").copy()
    tbid0 = store.get("TBID").copy()
    want_tbi, want_tbid, want_ortho = _cpp_step(tbi0, tbid0, wbib, ctx.int_step)
    kin.execute(vehicle, ctx)
    np.testing.assert_allclose(store.get("TBI"), want_tbi, rtol=1e-12)
    np.testing.assert_allclose(store.get("TBID"), want_tbid, rtol=1e-12)
    np.testing.assert_allclose(store.get("ortho_error"), want_ortho, rtol=1e-12)

    tbi1 = store.get("TBI").copy()
    tbid1 = store.get("TBID").copy()
    want_tbi, want_tbid, want_ortho = _cpp_step(tbi1, tbid1, wbib, ctx.int_step)
    kin.execute(vehicle, ctx)
    np.testing.assert_allclose(store.get("TBI"), want_tbi, rtol=1e-12)
    np.testing.assert_allclose(store.get("TBID"), want_tbid, rtol=1e-12)
    np.testing.assert_allclose(store.get("ortho_error"), want_ortho, rtol=1e-12)


def test_execute_alphax_betax_from_air_relative_velocity():
    vehicle, kin = _vehicle()
    store = vehicle.store
    _plant(store, lonx=10.0, latx=10.0, alt=10000.0)
    ctx = _ctx()
    kin.initialize(vehicle, ctx)
    alpha0x = 2.5
    beta0x = 5.0
    dvbe = 1000.0
    vbeb = np.array(
        [
            math.cos(alpha0x * RAD) * math.cos(beta0x * RAD) * dvbe,
            math.sin(beta0x * RAD) * dvbe,
            math.sin(alpha0x * RAD) * math.cos(beta0x * RAD) * dvbe,
        ]
    )
    tbd = store.get("TBD")
    vbed = tbd.T @ vbeb
    _plant(store, vbed=vbed, vaed=(0.0, 0.0, 0.0), dvba=float(np.linalg.norm(vbeb)))
    kin.execute(vehicle, ctx)
    np.testing.assert_allclose(store.get("alphax"), alpha0x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(store.get("betax"), beta0x, rtol=1e-12, atol=1e-12)


def test_execute_alpp_phip_from_air_relative_velocity():
    vehicle, kin = _vehicle()
    store = vehicle.store
    _plant(store)
    ctx = _ctx()
    kin.initialize(vehicle, ctx)
    tbd = store.get("TBD")
    vbab = np.array([1000.0, 100.0, 50.0])
    vbed = tbd.T @ vbab
    dvba = float(np.linalg.norm(vbab))
    _plant(store, vbed=vbed, vaed=(0.0, 0.0, 0.0), dvba=dvba)
    kin.execute(vehicle, ctx)
    dum = vbab[0] / dvba
    alpp = math.acos(dum)
    phip = math.atan2(vbab[1], vbab[2])
    np.testing.assert_allclose(store.get("alppx"), alpp * DEG, rtol=1e-12)
    np.testing.assert_allclose(store.get("phipx"), phip * DEG, rtol=1e-12)


def test_execute_inertial_incidence_angles():
    vehicle, kin = _vehicle()
    store = vehicle.store
    _plant(store)
    ctx = _ctx()
    kin.initialize(vehicle, ctx)
    vbib = np.array([1000.0, 30.0, 40.0])
    tbi = store.get("TBI")
    vbii = tbi.T @ vbib
    _plant(store, vbii=vbii)
    kin.execute(vehicle, ctx)
    dvbi = float(np.linalg.norm(vbib))
    np.testing.assert_allclose(
        store.get("alphaix"), math.atan2(vbib[2], vbib[0]) * DEG, rtol=1e-12
    )
    np.testing.assert_allclose(
        store.get("betaix"), math.asin(vbib[1] / dvbi) * DEG, rtol=1e-12
    )


def test_phip_forced_to_0_or_pi_when_vbab2_below_eps():
    vehicle, kin = _vehicle()
    store = vehicle.store
    _plant(store)
    ctx = _ctx()
    kin.initialize(vehicle, ctx)
    tbd = store.get("TBD")
    vbab = np.array([180.0, 1e-11, 1.0])
    _plant(store, vbed=tbd.T @ vbab, vaed=(0.0, 0.0, 0.0), dvba=180.0)
    kin.execute(vehicle, ctx)
    np.testing.assert_allclose(store.get("phipx"), 0.0, atol=1e-12)
    vbab = np.array([180.0, 1e-11, -1.0])
    _plant(store, vbed=tbd.T @ vbab, vaed=(0.0, 0.0, 0.0), dvba=180.0)
    kin.execute(vehicle, ctx)
    np.testing.assert_allclose(store.get("phipx"), PI * DEG, atol=1e-12)


def test_alpp_clamps_dum_with_cadac_sign():
    vehicle, kin = _vehicle()
    store = vehicle.store
    _plant(store)
    ctx = _ctx()
    kin.initialize(vehicle, ctx)
    tbd = store.get("TBD")
    vbab = np.array([200.0, 0.0, 0.0])
    _plant(store, vbed=tbd.T @ vbab, vaed=(0.0, 0.0, 0.0), dvba=180.0)
    kin.execute(vehicle, ctx)
    np.testing.assert_allclose(store.get("alppx"), 0.0, atol=1e-12)


def test_pitch_90_uses_gimbal_lock_branch():
    vehicle, kin = _vehicle()
    store = vehicle.store
    store.set("thtbdx", 90.0)
    _plant(store)
    ctx = _ctx()
    kin.initialize(vehicle, ctx)
    kin.execute(vehicle, ctx)
    assert np.isfinite(store.get("thtbdx"))
    assert np.isfinite(store.get("psibdx"))
    assert np.isfinite(store.get("phibdx"))
    np.testing.assert_allclose(store.get("thtbd"), PI / 2.0, atol=1e-12)
    np.testing.assert_allclose(store.get("thtbdx"), DEG * PI / 2.0, atol=1e-12)


def test_skip_trcode_when_absent():
    vehicle, kin = _vehicle()
    store = vehicle.store
    _plant(store)
    ctx = _ctx()
    kin.initialize(vehicle, ctx)
    kin.execute(vehicle, ctx)
    assert "trcode" not in store.names()


def test_euler_angles_recovered_after_zero_rate_step():
    vehicle, kin = _vehicle()
    store = vehicle.store
    store.set("psibdx", 0.0)
    store.set("thtbdx", 2.5)
    store.set("phibdx", 0.0)
    _plant(store, wbib=(0.0, 0.0, 0.0))
    ctx = _ctx()
    kin.initialize(vehicle, ctx)
    kin.execute(vehicle, ctx)
    np.testing.assert_allclose(store.get("thtbdx"), 2.5, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(store.get("psibdx"), 0.0, atol=1e-12)
    np.testing.assert_allclose(store.get("phibdx"), 0.0, atol=1e-12)
    np.testing.assert_allclose(store.get("thtbd"), 2.5 * RAD, rtol=1e-12, atol=1e-12)


def test_vertical_launch_earth_rate_step_uses_singularity_euler_branch():
    # ROCKET6 insertion: thtbdx=90. First kinematics step includes euler-init
    # WBIB = TBI·ω_earth. Numpy |TBD13| stays a few ulps below 1 (C++ Matrix
    # multiply often goes over), so the C++ `fabs(tbd13)<1` asin path divides
    # by ~1e-8 cosine and returns ~177 deg roll vs harvested 180.
    vehicle, kin = _vehicle()
    store = vehicle.store
    store.set("psibdx", -83.0)
    store.set("thtbdx", 90.0)
    store.set("phibdx", 0.0)
    ctx = _ctx(int_step=0.001)
    _plant(store, lonx=-120.49, latx=34.68, alt=100.0, wbib=(0.0, 0.0, 0.0))
    kin.initialize(vehicle, ctx)
    tbi = store.get("TBI")
    weii = np.array([0.0, 0.0, WEII3], dtype=float)
    store.set("WBIB", tbi @ weii)
    kin.execute(vehicle, ctx)
    np.testing.assert_allclose(store.get("phibdx"), 180.0, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(store.get("psibdx"), 90.0, rtol=0.0, atol=1e-3)
    np.testing.assert_allclose(store.get("thtbdx"), 90.0, rtol=0.0, atol=1e-5)


def test_earth_rate_step_tbd13_matches_cadac_ijk_multiply_not_numpy_at():
    # Numpy @ is ~1 ulp off C++ Matrix::operator* per 3x3. That accumulation
    # seeds the ROCKET6 t=69–73 pitch Schmitt 1 ms extra on-time.
    vehicle, kin = _vehicle()
    store = vehicle.store
    store.set("psibdx", -83.0)
    store.set("thtbdx", 90.0)
    store.set("phibdx", 0.0)
    ctx = _ctx(int_step=0.001)
    _plant(store, lonx=-120.49, latx=34.68, alt=100.0, wbib=(0.0, 0.0, 0.0))
    kin.initialize(vehicle, ctx)
    tbi = store.get("TBI")
    weii = np.array([0.0, 0.0, WEII3], dtype=float)
    store.set("WBIB", _cadac_matmul(tbi, weii))
    tbi0 = store.get("TBI").copy()
    tbid0 = store.get("TBID").copy()
    wbib = store.get("WBIB").copy()
    tdi = cad_tdi84(-120.49 * RAD, 34.68 * RAD, 100.0, 0.0)
    want_tbi, _, _ = _cpp_step(tbi0, tbid0, wbib, ctx.int_step)
    want_tbd = _cadac_matmul(want_tbi, tdi.T.copy())
    kin.execute(vehicle, ctx)
    np.testing.assert_array_equal(store.get("TBI"), want_tbi)
    assert store.get("TBD")[0, 2] == want_tbd[0, 2]


def test_cadac_inverse_matches_cpp_adjoint_over_det_not_numpy():
    amat = np.array(
        [[2.0, 0.1, 0.0], [0.1, 3.0, -0.2], [0.0, -0.2, 4.0]],
        dtype=float,
    )
    got = cadac_inverse(amat)
    det = amat[0, 0] * (amat[1, 1] * amat[2, 2] - amat[1, 2] * amat[2, 1])
    det -= amat[0, 1] * (amat[1, 0] * amat[2, 2] - amat[1, 2] * amat[2, 0])
    det += amat[0, 2] * (amat[1, 0] * amat[2, 1] - amat[1, 1] * amat[2, 0])
    np.testing.assert_allclose(cadac_matmul(amat, got), np.eye(3), atol=1e-15)
    numpy_inv = np.linalg.inv(amat)
    assert not np.array_equal(got, numpy_inv)


def test_cad_tgi84_matches_cadac_ijk_multiply_not_numpy_at():
    lon = -120.49 * RAD
    lat = 34.68 * RAD
    alt = 45000.0
    time = 69.9
    got = cad_tgi84(lon, lat, alt, time)
    tdi = cad_tdi84(lon, lat, alt, time)
    from cadac.math.wgs84 import FLATTENING, SMAJOR_AXIS

    r0 = SMAJOR_AXIS * (
        1.0
        - FLATTENING * (1.0 - np.cos(2.0 * lat)) / 2.0
        + 5.0 * FLATTENING**2 * (1.0 - np.cos(4.0 * lat)) / 16.0
    )
    dd = FLATTENING * np.sin(2.0 * lat) * (1.0 - FLATTENING / 2.0 - alt / r0)
    tgd = np.zeros((3, 3))
    tgd[0, 0] = np.cos(dd)
    tgd[2, 2] = np.cos(dd)
    tgd[1, 1] = 1.0
    tgd[2, 0] = np.sin(dd)
    tgd[0, 2] = -np.sin(dd)
    want = cadac_matmul(tgd, tdi)
    np.testing.assert_array_equal(got, want)
