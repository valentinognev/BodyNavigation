from types import SimpleNamespace

import numpy as np

from cadac.constants import DEG, RAD
from cadac.eom.flat6 import Flat6Euler
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore

# F-16 IBBB / engine momentum from FALCON6 Plane::init_aerodynamics.
_F16_IBBB = np.array(
    [
        [12875.0, 0.0, -1331.4],
        [0.0, 75673.0, 0.0],
        [-1331.4, 0.0, 85551.0],
    ],
    dtype=float,
)
_F16_ENG_ANG_MOM = 70000.0
_ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def _vehicle():
    store = StateStore()
    vehicle = SimpleNamespace(store=store)
    euler = Flat6Euler()
    euler.define(vehicle)
    return vehicle, euler


def _plant(store, ibbb=_F16_IBBB, eng_ang_mom=_F16_ENG_ANG_MOM, fmb=(0.0, 0.0, 0.0)):
    if "IBBB" not in store.names():
        store.define(Field("IBBB", _ZEROS33, "mat", "init", "aerodynamics"))
    if "eng_ang_mom" not in store.names():
        store.define(Field("eng_ang_mom", 0.0, "real", "init", "aerodynamics"))
    if "FMB" not in store.names():
        store.define(Field("FMB", (0.0, 0.0, 0.0), "vec", "out", "forces"))
    store.set("IBBB", np.asarray(ibbb, dtype=float))
    store.set("eng_ang_mom", float(eng_ang_mom))
    store.set("FMB", np.asarray(fmb, dtype=float))


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


def _cpp_euler_step(fmb, ibbb, eng_ang_mom, wbeb, wbebd, dt):
    l_engine = np.array([eng_ang_mom, 0.0, 0.0], dtype=float)
    wacc_next = np.linalg.inv(ibbb) @ (fmb - _skew(wbeb) @ (ibbb @ wbeb + l_engine))
    wbeb_new = integrate(wacc_next, wbebd, wbeb, dt)
    return wbeb_new, wacc_next


def test_name_is_euler():
    assert Flat6Euler().name == "euler"


def test_define_registers_rates_and_state():
    vehicle, _euler = _vehicle()
    store = vehicle.store
    assert store.get("ppx") == 0.0
    assert store.get("qqx") == 0.0
    assert store.get("rrx") == 0.0
    np.testing.assert_array_equal(store.get("WBEB"), np.zeros(3))
    np.testing.assert_array_equal(store.get("WBEBD"), np.zeros(3))
    assert store.field("ppx").outputs == ("plot",)
    assert store.field("qqx").outputs == ("plot",)
    assert store.field("rrx").outputs == ("plot",)


def test_define_does_not_register_plane_forces_or_newton_fields():
    store = StateStore()
    Flat6Euler().define(SimpleNamespace(store=store))
    for name in (
        "IBBB",
        "eng_ang_mom",
        "FMB",
        "hbe",
        "VBEL",
        "VBEB",
        "TBL",
        "alphax",
    ):
        assert name not in store.names()


def test_initialize_wbeb_from_ppx_qqx_rrx_deg_per_s():
    vehicle, euler = _vehicle()
    store = vehicle.store
    store.set("ppx", 10.0)
    store.set("qqx", -2.0)
    store.set("rrx", 0.5)
    euler.initialize(vehicle, None)
    np.testing.assert_array_equal(
        store.get("WBEB"),
        np.array([10.0 * RAD, -2.0 * RAD, 0.5 * RAD]),
    )
    np.testing.assert_array_equal(store.get("WBEBD"), np.zeros(3))
    assert store.get("ppx") == 10.0
    assert store.get("qqx") == -2.0
    assert store.get("rrx") == 0.5


def test_zero_fmb_nonzero_ppx_one_step_matches_cpp_replica():
    vehicle, euler = _vehicle()
    store = vehicle.store
    store.set("ppx", 10.0)
    _plant(store, ibbb=_F16_IBBB, eng_ang_mom=_F16_ENG_ANG_MOM, fmb=(0.0, 0.0, 0.0))
    euler.initialize(vehicle, None)
    wbeb_old = store.get("WBEB").copy()
    wbebd_old = store.get("WBEBD").copy()
    dt = 0.001
    euler.execute(vehicle, SimpleNamespace(int_step=dt))

    wbeb_want, wacc_want = _cpp_euler_step(
        np.zeros(3),
        _F16_IBBB,
        _F16_ENG_ANG_MOM,
        wbeb_old,
        wbebd_old,
        dt,
    )
    np.testing.assert_allclose(store.get("WBEB"), wbeb_want, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("WBEBD"), wacc_want, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("ppx"), wbeb_want[0] * DEG, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("qqx"), wbeb_want[1] * DEG, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("rrx"), wbeb_want[2] * DEG, rtol=1e-12, atol=1e-14)
    assert store.get("qqx") != 0.0


def test_execute_second_step_uses_stored_wbebd():
    vehicle, euler = _vehicle()
    store = vehicle.store
    store.set("ppx", 10.0)
    _plant(store, ibbb=_F16_IBBB, eng_ang_mom=_F16_ENG_ANG_MOM, fmb=(0.0, 0.0, 0.0))
    euler.initialize(vehicle, None)
    ctx = SimpleNamespace(int_step=0.001)
    euler.execute(vehicle, ctx)
    wbeb_old = store.get("WBEB").copy()
    wbebd_old = store.get("WBEBD").copy()
    fmb = store.get("FMB").copy()
    ibbb = store.get("IBBB").copy()
    eng_ang_mom = store.get("eng_ang_mom")
    euler.execute(vehicle, ctx)
    wbeb_want, wacc_want = _cpp_euler_step(
        fmb, ibbb, eng_ang_mom, wbeb_old, wbebd_old, ctx.int_step
    )
    np.testing.assert_allclose(store.get("WBEB"), wbeb_want, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("WBEBD"), wacc_want, rtol=1e-12, atol=1e-14)


def test_spherical_inertia_zero_engine_zero_fmb_holds_rates():
    vehicle, euler = _vehicle()
    store = vehicle.store
    store.set("ppx", 10.0)
    store.set("qqx", 3.0)
    store.set("rrx", -1.0)
    _plant(store, ibbb=np.eye(3) * 10000.0, eng_ang_mom=0.0, fmb=(0.0, 0.0, 0.0))
    euler.initialize(vehicle, None)
    wbeb_old = store.get("WBEB").copy()
    euler.execute(vehicle, SimpleNamespace(int_step=0.001))
    np.testing.assert_allclose(store.get("WBEB"), wbeb_old, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("WBEBD"), np.zeros(3), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("ppx"), 10.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(store.get("qqx"), 3.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(store.get("rrx"), -1.0, rtol=1e-12, atol=1e-12)


def test_nonzero_fmb_one_step_matches_cpp_replica():
    vehicle, euler = _vehicle()
    store = vehicle.store
    store.set("ppx", 10.0)
    store.set("qqx", 2.0)
    fmb = (100.0, -50.0, 25.0)
    _plant(store, ibbb=_F16_IBBB, eng_ang_mom=_F16_ENG_ANG_MOM, fmb=fmb)
    euler.initialize(vehicle, None)
    wbeb_old = store.get("WBEB").copy()
    wbebd_old = store.get("WBEBD").copy()
    dt = 0.001
    euler.execute(vehicle, SimpleNamespace(int_step=dt))
    wbeb_want, wacc_want = _cpp_euler_step(
        np.array(fmb, dtype=float),
        _F16_IBBB,
        _F16_ENG_ANG_MOM,
        wbeb_old,
        wbebd_old,
        dt,
    )
    np.testing.assert_allclose(store.get("WBEB"), wbeb_want, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("WBEBD"), wacc_want, rtol=1e-12, atol=1e-14)
