from types import SimpleNamespace

import numpy as np

from cadac.constants import DEG, RAD, WEII3
from cadac.eom.round6 import Round6Euler
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore

# GHAME IBBB0 from HYPER6 Hyper::init_propulsion.
_GHAME_IBBB = np.array(
    [
        [1.573e6, 0.0, 0.38e6],
        [0.0, 31.6e6, 0.0],
        [0.38e6, 0.0, 32.54e6],
    ],
    dtype=float,
)
_ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
_ZEROS3 = (0.0, 0.0, 0.0)


def _ctx(int_step=0.01):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _vehicle():
    store = StateStore()
    vehicle = SimpleNamespace(store=store)
    euler = Round6Euler()
    euler.define(vehicle)
    return vehicle, euler


def _plant(store, ibbb=_GHAME_IBBB, fmb=(0.0, 0.0, 0.0), tbi=None):
    if tbi is None:
        tbi = np.eye(3)
    if "IBBB" not in store.names():
        store.define(Field("IBBB", _ZEROS33, "mat", "out", "propulsion"))
    if "FMB" not in store.names():
        store.define(Field("FMB", _ZEROS3, "vec", "out", "forces"))
    if "TBI" not in store.names():
        store.define(Field("TBI", _ZEROS33, "mat", "state", "kinematics"))
    store.set("IBBB", np.asarray(ibbb, dtype=float))
    store.set("FMB", np.asarray(fmb, dtype=float))
    store.set("TBI", np.asarray(tbi, dtype=float))


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


def _cpp_wbib_init(ppx, qqx, rrx, tbi):
    wbeb = np.array([ppx * RAD, qqx * RAD, rrx * RAD], dtype=float)
    weii = np.array([0.0, 0.0, WEII3], dtype=float)
    return wbeb, wbeb + tbi @ weii


def _cpp_euler_step(fmb, ibbb, wbib, wbibd, tbi, dt):
    wacc_next = np.linalg.inv(ibbb) @ (fmb - _skew(wbib) @ ibbb @ wbib)
    wbib_new = integrate(wacc_next, wbibd, wbib, dt)
    wbii = tbi.T @ wbib_new
    weii = np.array([0.0, 0.0, WEII3], dtype=float)
    wbeb = wbib_new - tbi @ weii
    return wbib_new, wacc_next, wbeb, wbii


def test_name_is_euler():
    assert Round6Euler().name == "euler"


def test_define_registers_def_euler_fields():
    store = StateStore()
    Round6Euler().define(SimpleNamespace(store=store))
    for name in ("ppx", "qqx", "rrx", "WBEB", "WBIB", "WBIBD", "WBII"):
        assert name in store.names()
    assert store.get("ppx") == 0.0
    assert store.get("qqx") == 0.0
    assert store.get("rrx") == 0.0
    np.testing.assert_array_equal(store.get("WBEB"), np.zeros(3))
    np.testing.assert_array_equal(store.get("WBIB"), np.zeros(3))
    np.testing.assert_array_equal(store.get("WBIBD"), np.zeros(3))
    np.testing.assert_array_equal(store.get("WBII"), np.zeros(3))
    assert store.field("ppx").role == "out"
    assert store.field("ppx").outputs == ("plot",)
    assert store.field("qqx").outputs == ("plot",)
    assert store.field("rrx").outputs == ("plot",)
    assert store.field("WBEB").role == "diag"
    assert store.field("WBIB").role == "state"
    assert store.field("WBIBD").role == "state"
    assert store.field("WBII").role == "out"


def test_define_does_not_register_plane_forces_or_kinematics_tbi():
    store = StateStore()
    Round6Euler().define(SimpleNamespace(store=store))
    for name in ("IBBB", "FMB", "TBI", "eng_ang_mom", "WBEBD"):
        assert name not in store.names()


def test_initialize_wbib_includes_earth_rate_identity_tbi():
    vehicle, euler = _vehicle()
    store = vehicle.store
    store.set("ppx", 10.0)
    store.set("qqx", -2.0)
    store.set("rrx", 0.5)
    _plant(store, tbi=np.eye(3))
    euler.initialize(vehicle, None)
    wbeb_want, wbib_want = _cpp_wbib_init(10.0, -2.0, 0.5, np.eye(3))
    np.testing.assert_allclose(store.get("WBIB"), wbib_want, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(store.get("WBIBD"), np.zeros(3))
    assert store.get("ppx") == 10.0
    assert store.get("qqx") == -2.0
    assert store.get("rrx") == 0.5
    np.testing.assert_array_equal(store.get("WBEB"), np.zeros(3))
    np.testing.assert_allclose(wbib_want, wbeb_want + np.array([0.0, 0.0, WEII3]))


def test_initialize_wbib_rotates_earth_rate_with_tbi():
    vehicle, euler = _vehicle()
    store = vehicle.store
    store.set("ppx", 10.0)
    tbi = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    _plant(store, tbi=tbi)
    euler.initialize(vehicle, None)
    _wbeb_want, wbib_want = _cpp_wbib_init(10.0, 0.0, 0.0, tbi)
    np.testing.assert_allclose(store.get("WBIB"), wbib_want, rtol=1e-12, atol=1e-14)
    assert not np.allclose(store.get("WBIB")[2], WEII3)


def test_zero_fmb_identity_tbi_ppx10_one_step_matches_cpp_replica():
    vehicle, euler = _vehicle()
    store = vehicle.store
    store.set("ppx", 10.0)
    _plant(store, ibbb=_GHAME_IBBB, fmb=(0.0, 0.0, 0.0), tbi=np.eye(3))
    euler.initialize(vehicle, None)
    wbib_old = store.get("WBIB").copy()
    wbibd_old = store.get("WBIBD").copy()
    tbi = store.get("TBI").copy()
    dt = 0.01
    euler.execute(vehicle, SimpleNamespace(int_step=dt))

    wbib_want, wacc_want, wbeb_want, wbii_want = _cpp_euler_step(
        np.zeros(3),
        _GHAME_IBBB,
        wbib_old,
        wbibd_old,
        tbi,
        dt,
    )
    np.testing.assert_allclose(store.get("WBIB"), wbib_want, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("WBIBD"), wacc_want, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("WBEB"), wbeb_want, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("WBII"), wbii_want, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("ppx"), wbeb_want[0] * DEG, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("qqx"), wbeb_want[1] * DEG, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("rrx"), wbeb_want[2] * DEG, rtol=1e-12, atol=1e-14)
    assert abs(store.get("ppx") - 10.0 * RAD) > 1.0


def test_execute_uses_ctx_int_step():
    vehicle, euler = _vehicle()
    store = vehicle.store
    store.set("ppx", 10.0)
    _plant(store)
    euler.initialize(vehicle, None)
    wbib_old = store.get("WBIB").copy()
    wbibd_old = store.get("WBIBD").copy()
    tbi = store.get("TBI").copy()
    ctx = _ctx(int_step=0.01)
    euler.execute(vehicle, ctx)
    wbib_want, wacc_want, wbeb_want, _wbii = _cpp_euler_step(
        np.zeros(3),
        _GHAME_IBBB,
        wbib_old,
        wbibd_old,
        tbi,
        ctx.int_step,
    )
    np.testing.assert_allclose(store.get("WBIB"), wbib_want, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("WBIBD"), wacc_want, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("ppx"), wbeb_want[0] * DEG, rtol=1e-12, atol=1e-14)


def test_execute_second_step_uses_stored_wbibd():
    vehicle, euler = _vehicle()
    store = vehicle.store
    store.set("ppx", 10.0)
    _plant(store, ibbb=_GHAME_IBBB, fmb=(0.0, 0.0, 0.0), tbi=np.eye(3))
    euler.initialize(vehicle, None)
    ctx = SimpleNamespace(int_step=0.01)
    euler.execute(vehicle, ctx)
    wbib_old = store.get("WBIB").copy()
    wbibd_old = store.get("WBIBD").copy()
    fmb = store.get("FMB").copy()
    ibbb = store.get("IBBB").copy()
    tbi = store.get("TBI").copy()
    euler.execute(vehicle, ctx)
    wbib_want, wacc_want, wbeb_want, wbii_want = _cpp_euler_step(
        fmb, ibbb, wbib_old, wbibd_old, tbi, ctx.int_step
    )
    np.testing.assert_allclose(store.get("WBIB"), wbib_want, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("WBIBD"), wacc_want, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("WBEB"), wbeb_want, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("WBII"), wbii_want, rtol=1e-12, atol=1e-14)


def test_spherical_inertia_zero_fmb_holds_wbib():
    vehicle, euler = _vehicle()
    store = vehicle.store
    store.set("ppx", 10.0)
    store.set("qqx", 3.0)
    store.set("rrx", -1.0)
    _plant(store, ibbb=np.eye(3) * 10000.0, fmb=(0.0, 0.0, 0.0), tbi=np.eye(3))
    euler.initialize(vehicle, None)
    wbib_old = store.get("WBIB").copy()
    euler.execute(vehicle, SimpleNamespace(int_step=0.01))
    np.testing.assert_allclose(store.get("WBIB"), wbib_old, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("WBIBD"), np.zeros(3), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("ppx"), 10.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(store.get("qqx"), 3.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(store.get("rrx"), -1.0, rtol=1e-12, atol=1e-12)


def test_nonzero_fmb_one_step_matches_cpp_replica():
    vehicle, euler = _vehicle()
    store = vehicle.store
    store.set("ppx", 10.0)
    store.set("qqx", 2.0)
    fmb = (100.0, -50.0, 25.0)
    _plant(store, ibbb=_GHAME_IBBB, fmb=fmb, tbi=np.eye(3))
    euler.initialize(vehicle, None)
    wbib_old = store.get("WBIB").copy()
    wbibd_old = store.get("WBIBD").copy()
    tbi = store.get("TBI").copy()
    dt = 0.01
    euler.execute(vehicle, SimpleNamespace(int_step=dt))
    wbib_want, wacc_want, wbeb_want, wbii_want = _cpp_euler_step(
        np.array(fmb, dtype=float),
        _GHAME_IBBB,
        wbib_old,
        wbibd_old,
        tbi,
        dt,
    )
    np.testing.assert_allclose(store.get("WBIB"), wbib_want, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("WBIBD"), wacc_want, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("WBEB"), wbeb_want, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("WBII"), wbii_want, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("ppx"), wbeb_want[0] * DEG, rtol=1e-12, atol=1e-14)


def test_terminate_exists_and_is_pass():
    vehicle, euler = _vehicle()
    euler.terminate(vehicle, None)
