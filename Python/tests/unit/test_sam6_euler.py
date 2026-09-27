from pathlib import Path

import numpy as np
import pytest

from cadac.constants import DEG
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.sam6.euler import Sam6Euler

RTOL = 1e-12
ATOL = 1e-14

DEFINED = (
    "ppd",
    "pp",
    "qqd",
    "qq",
    "rrd",
    "rr",
    "ppx",
    "qqx",
    "rrx",
    "WBEB",
)
ROLES = {
    "ppd": "state",
    "pp": "state",
    "qqd": "state",
    "qq": "state",
    "rrd": "state",
    "rr": "state",
    "ppx": "out",
    "qqx": "out",
    "rrx": "out",
    "WBEB": "diag",
}
OUTPUTS = {
    "ppd": (),
    "pp": (),
    "qqd": (),
    "qq": (),
    "rrd": (),
    "rr": (),
    "ppx": ("plot",),
    "qqx": ("plot",),
    "rrx": ("plot",),
    "WBEB": (),
}
VEC_FIELDS = ("WBEB",)
NOT_DEFINED = ("FMB", "ai11", "ai33", "IBBB", "eng_ang_mom", "WBEBD")


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(int_step=0.001):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _plant(store, *, fmb, ai11, ai33):
    store.define(Field("FMB", fmb, "vec", "out", "forces"))
    store.define(Field("ai11", ai11, "real", "out", "propulsion"))
    store.define(Field("ai33", ai33, "real", "out", "propulsion"))


def _ready(*, fmb=(1.0, 0.0, 0.0), ai11=2.9, ai33=440.0, pp=0.0, qq=0.0, rr=0.0):
    vehicle = _Vehicle()
    euler = Sam6Euler()
    euler.define(vehicle)
    euler.initialize(vehicle, _ctx())
    _plant(vehicle.store, fmb=fmb, ai11=ai11, ai33=ai33)
    vehicle.store.set("pp", pp)
    vehicle.store.set("qq", qq)
    vehicle.store.set("rr", rr)
    return vehicle, euler


def _cpp_step(fmb, ai11, ai33, ppd, pp, qqd, qq, rrd, rr, dt):
    fmb1, fmb2, fmb3 = fmb
    ppd_new = fmb1 / ai11
    pp = integrate(ppd_new, ppd, pp, dt)
    ppd = ppd_new
    qqd_new = ((ai33 - ai11) * pp * rr + fmb2) / ai33
    qq = integrate(qqd_new, qqd, qq, dt)
    qqd = qqd_new
    rrd_new = (-(ai33 - ai11) * pp * qq + fmb3) / ai33
    rr = integrate(rrd_new, rrd, rr, dt)
    rrd = rrd_new
    return ppd, pp, qqd, qq, rrd, rr


def test_name_is_euler():
    assert Sam6Euler().name == "euler"


def test_define_cpp_fields():
    vehicle = _Vehicle()
    Sam6Euler().define(vehicle)
    store = vehicle.store
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        assert field.module == "euler"
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        if name in VEC_FIELDS:
            assert field.type == "vec"
            np.testing.assert_allclose(store.get(name), np.zeros(3), rtol=RTOL, atol=ATOL)
        else:
            assert field.type == "real"
            assert _approx(store.get(name), 0.0), name


def test_define_does_not_register_forces_or_inertia():
    vehicle = _Vehicle()
    Sam6Euler().define(vehicle)
    for name in NOT_DEFINED:
        assert name not in vehicle.store.names()


def test_initialize_is_pass():
    vehicle = _Vehicle()
    euler = Sam6Euler()
    euler.define(vehicle)
    vehicle.store.set("pp", 0.1)
    vehicle.store.set("ppx", 5.0)
    before = {name: np.array(vehicle.store.get(name), copy=True) for name in DEFINED}
    assert euler.initialize(vehicle, _ctx()) is None
    for name in DEFINED:
        np.testing.assert_allclose(
            vehicle.store.get(name), before[name], rtol=RTOL, atol=ATOL
        )


def test_roll_torque_one_step_matches_integrate():
    vehicle, euler = _ready(fmb=(1.0, 0.0, 0.0), ai11=2.9, ai33=440.0)
    dt = 0.001
    euler.execute(vehicle, _ctx(dt))
    store = vehicle.store
    want_pp = integrate(1.0 / 2.9, 0.0, 0.0, dt)
    np.testing.assert_allclose(store.get("pp"), want_pp, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("qq"), 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("rr"), 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ppd"), 1.0 / 2.9, rtol=RTOL, atol=ATOL)


def test_writes_wbeb_and_deg_rates():
    vehicle, euler = _ready(fmb=(1.0, 0.0, 0.0), ai11=2.9, ai33=440.0)
    dt = 0.001
    euler.execute(vehicle, _ctx(dt))
    store = vehicle.store
    pp = store.get("pp")
    qq = store.get("qq")
    rr = store.get("rr")
    np.testing.assert_allclose(store.get("WBEB"), [pp, qq, rr], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ppx"), pp * DEG, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("qqx"), qq * DEG, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("rrx"), rr * DEG, rtol=RTOL, atol=ATOL)


def test_pitch_yaw_coupling_uses_updated_pp():
    ai11 = 2.9
    ai33 = 440.0
    dt = 0.001
    fmb = (1.0, 0.0, 0.0)
    vehicle, euler = _ready(fmb=fmb, ai11=ai11, ai33=ai33, pp=0.0, qq=0.0, rr=1.0)
    euler.execute(vehicle, _ctx(dt))
    want = _cpp_step(fmb, ai11, ai33, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, dt)
    store = vehicle.store
    np.testing.assert_allclose(store.get("ppd"), want[0], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("pp"), want[1], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("qqd"), want[2], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("qq"), want[3], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("rrd"), want[4], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("rr"), want[5], rtol=RTOL, atol=ATOL)
    stale_qqd = ((ai33 - ai11) * 0.0 * 1.0 + 0.0) / ai33
    stale_qq = integrate(stale_qqd, 0.0, 0.0, dt)
    assert store.get("qq") != pytest.approx(stale_qq, rel=RTOL, abs=ATOL)
    stale_rrd = (-(ai33 - ai11) * store.get("pp") * 0.0 + 0.0) / ai33
    assert store.get("rrd") != pytest.approx(stale_rrd, rel=RTOL, abs=ATOL)


def test_second_step_uses_stored_ppd():
    vehicle, euler = _ready(fmb=(1.0, 0.0, 0.0), ai11=2.9, ai33=440.0)
    dt = 0.001
    ctx = _ctx(dt)
    euler.execute(vehicle, ctx)
    store = vehicle.store
    ppd = store.get("ppd")
    pp = store.get("pp")
    qqd = store.get("qqd")
    qq = store.get("qq")
    rrd = store.get("rrd")
    rr = store.get("rr")
    euler.execute(vehicle, ctx)
    want = _cpp_step((1.0, 0.0, 0.0), 2.9, 440.0, ppd, pp, qqd, qq, rrd, rr, dt)
    np.testing.assert_allclose(store.get("pp"), want[1], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ppd"), want[0], rtol=RTOL, atol=ATOL)
    fresh = integrate(1.0 / 2.9, 0.0, pp, dt)
    assert store.get("pp") != pytest.approx(fresh, rel=RTOL, abs=ATOL)


def test_is_not_flat6_euler_subclass():
    from cadac.eom.flat6 import Flat6Euler

    assert not issubclass(Sam6Euler, Flat6Euler)


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.flat6.sam6.euler as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6Euler" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src


def test_terminate_exists_and_is_pass():
    vehicle, euler = _ready()
    assert euler.terminate(vehicle, _ctx()) is None
