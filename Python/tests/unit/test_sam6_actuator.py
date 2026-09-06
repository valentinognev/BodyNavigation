from pathlib import Path

import math

import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.sam6.actuator import Sam6Actuator

RTOL = 1e-12
ATOL = 1e-14

DT = 0.001
DLIMX = 28.0
DDLIMX = 600.0
WNACT = 600.0
ZETACT = 0.7

DEFINED = (
    "mact",
    "dlimx",
    "ddlimx",
    "wnact",
    "zetact",
    "dpx",
    "dqx",
    "drx",
    "delx1",
    "delx2",
    "delx3",
    "delx4",
    "dxd1",
    "dxd2",
    "dxd3",
    "dxd4",
    "dx1",
    "dx2",
    "dx3",
    "dx4",
    "ddxd1",
    "ddxd2",
    "ddxd3",
    "ddxd4",
    "ddx1",
    "ddx2",
    "ddx3",
    "ddx4",
    "delcx1",
    "delcx2",
    "delcx3",
    "delcx4",
)
ROLES = {
    "mact": "data",
    "dlimx": "data",
    "ddlimx": "data",
    "wnact": "data",
    "zetact": "data",
    "dpx": "out",
    "dqx": "out",
    "drx": "out",
    "delx1": "diag",
    "delx2": "diag",
    "delx3": "diag",
    "delx4": "diag",
    "dxd1": "state",
    "dxd2": "state",
    "dxd3": "state",
    "dxd4": "state",
    "dx1": "state",
    "dx2": "state",
    "dx3": "state",
    "dx4": "state",
    "ddxd1": "state",
    "ddxd2": "state",
    "ddxd3": "state",
    "ddxd4": "state",
    "ddx1": "state",
    "ddx2": "state",
    "ddx3": "state",
    "ddx4": "state",
    "delcx1": "diag",
    "delcx2": "diag",
    "delcx3": "diag",
    "delcx4": "diag",
}
OUTPUTS = {
    "mact": (),
    "dlimx": (),
    "ddlimx": (),
    "wnact": (),
    "zetact": (),
    "dpx": ("plot",),
    "dqx": ("plot",),
    "drx": ("plot",),
    "delx1": ("scrn", "plot"),
    "delx2": ("scrn", "plot"),
    "delx3": ("scrn", "plot"),
    "delx4": ("scrn", "plot"),
    "dxd1": (),
    "dxd2": (),
    "dxd3": (),
    "dxd4": (),
    "dx1": (),
    "dx2": (),
    "dx3": (),
    "dx4": (),
    "ddxd1": (),
    "ddxd2": (),
    "ddxd3": (),
    "ddxd4": (),
    "ddx1": (),
    "ddx2": (),
    "ddx3": (),
    "ddx4": (),
    "delcx1": (),
    "delcx2": (),
    "delcx3": (),
    "delcx4": (),
}
INT_FIELDS = ("mact",)
NOT_DEFINED = (
    "dpcx",
    "dqcx",
    "drcx",
    "time",
    "delacx",
    "delecx",
    "delrcx",
    "FAPB",
    "FMB",
    "DX",
    "DDX",
    "DXD",
    "DDXD",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ctx(int_step=DT):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _commands(store, *, dpcx=0.0, dqcx=0.0, drcx=0.0):
    store.define(Field("dpcx", dpcx, "real", "out", "control"))
    store.define(Field("dqcx", dqcx, "real", "out", "control"))
    store.define(Field("drcx", drcx, "real", "out", "control"))


def _ready(
    *,
    mact=0,
    dlimx=DLIMX,
    ddlimx=DDLIMX,
    wnact=WNACT,
    zetact=ZETACT,
    dpcx=0.0,
    dqcx=0.0,
    drcx=0.0,
):
    vehicle = _Vehicle()
    act = Sam6Actuator()
    act.define(vehicle)
    act.initialize(vehicle, _ctx())
    _commands(vehicle.store, dpcx=dpcx, dqcx=dqcx, drcx=drcx)
    store = vehicle.store
    store.set("mact", mact)
    store.set("dlimx", dlimx)
    store.set("ddlimx", ddlimx)
    store.set("wnact", wnact)
    store.set("zetact", zetact)
    return vehicle, act


def test_name_is_actuator():
    assert Sam6Actuator().name == "actuator"


def test_define_registers_cpp_fields_not_commands():
    vehicle = _Vehicle()
    Sam6Actuator().define(vehicle)
    store = vehicle.store
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        assert field.module == "actuator"
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        if name in INT_FIELDS:
            assert field.type == "int"
            assert store.get(name) == 0
        else:
            assert field.type == "real"
            assert _approx(store.get(name), 0.0), name
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_initialize_is_pass():
    vehicle = _Vehicle()
    act = Sam6Actuator()
    act.define(vehicle)
    vehicle.store.set("dx1", 1.0)
    vehicle.store.set("dpx", 5.0)
    before = {name: vehicle.store.get(name) for name in DEFINED}
    assert act.initialize(vehicle, _ctx()) is None
    for name in DEFINED:
        assert _approx(vehicle.store.get(name), before[name]), name


def test_mact_0_roll_command_mixes_to_four_equal_fins():
    vehicle, act = _ready(mact=0, dlimx=28.0, dpcx=10.0, dqcx=0.0, drcx=0.0)
    act.execute(vehicle, _ctx())
    store = vehicle.store
    for name in ("delx1", "delx2", "delx3", "delx4"):
        assert _approx(store.get(name), -10.0), name
    assert _approx(store.get("dpx"), 10.0)
    assert _approx(store.get("dqx"), 0.0)
    assert _approx(store.get("drx"), 0.0)
    assert _approx(store.get("delcx1"), -10.0)
    assert _approx(store.get("delcx2"), -10.0)
    assert _approx(store.get("delcx3"), -10.0)
    assert _approx(store.get("delcx4"), -10.0)


def test_mact_1_same_position_limit_path_as_mact_0():
    vehicle, act = _ready(mact=1, dlimx=28.0, dpcx=10.0, dqcx=0.0, drcx=0.0)
    act.execute(vehicle, _ctx())
    store = vehicle.store
    for name in ("delx1", "delx2", "delx3", "delx4"):
        assert _approx(store.get(name), -10.0), name
    assert _approx(store.get("dpx"), 10.0)
    assert _approx(store.get("dqx"), 0.0)
    assert _approx(store.get("drx"), 0.0)


def test_mact_0_limits_fins_beyond_dlimx():
    vehicle, act = _ready(mact=0, dlimx=28.0, dpcx=40.0, dqcx=0.0, drcx=0.0)
    act.execute(vehicle, _ctx())
    store = vehicle.store
    for name in ("delx1", "delx2", "delx3", "delx4"):
        assert _approx(store.get(name), -28.0), name
    assert _approx(store.get("dpx"), 28.0)
    assert _approx(store.get("delcx1"), -40.0)


def test_mact_0_pitch_yaw_mix():
    vehicle, act = _ready(mact=0, dlimx=28.0, dpcx=0.0, dqcx=10.0, drcx=2.0)
    act.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("delx1"), -2.0)
    assert _approx(store.get("delx2"), 10.0)
    assert _approx(store.get("delx3"), 2.0)
    assert _approx(store.get("delx4"), -10.0)
    assert _approx(store.get("dpx"), 0.0)
    assert _approx(store.get("dqx"), 10.0)
    assert _approx(store.get("drx"), 2.0)


def test_mact_0_does_not_update_states():
    vehicle, act = _ready(mact=0, dpcx=10.0)
    store = vehicle.store
    store.set("dx1", 1.0)
    store.set("ddx1", 2.0)
    store.set("dxd1", 3.0)
    store.set("ddxd1", 4.0)
    act.execute(vehicle, _ctx())
    assert _approx(store.get("dx1"), 1.0)
    assert _approx(store.get("ddx1"), 2.0)
    assert _approx(store.get("dxd1"), 3.0)
    assert _approx(store.get("ddxd1"), 4.0)


def test_mact_2_one_step_delx1_finite_and_lags():
    vehicle, act = _ready(
        mact=2,
        dlimx=28.0,
        wnact=600.0,
        zetact=0.7,
        dpcx=10.0,
        dqcx=0.0,
        drcx=0.0,
    )
    act.execute(vehicle, _ctx(DT))
    store = vehicle.store
    assert math.isfinite(store.get("delx1"))
    assert store.get("delx1") != pytest.approx(-10.0, rel=RTOL, abs=ATOL)
    assert abs(store.get("delx1")) <= DLIMX


def test_mact_2_one_step_matches_cadac_trapezoid():
    vehicle, act = _ready(
        mact=2,
        dlimx=28.0,
        ddlimx=600.0,
        wnact=600.0,
        zetact=0.7,
        dpcx=10.0,
        dqcx=0.0,
        drcx=0.0,
    )
    act.execute(vehicle, _ctx(DT))
    store = vehicle.store
    # states start 0; delcx1=-10; dx1=integrate(0,0,0,0.001)=0
    # ddxd1=600**2*(-10)=-3600000; ddx1=integrate(-3600000,0,0,0.001)=-1800
    assert _approx(store.get("delx1"), 0.0)
    assert _approx(store.get("dx1"), 0.0)
    assert _approx(store.get("dxd1"), 0.0)
    assert _approx(store.get("ddxd1"), -3600000.0)
    assert _approx(store.get("ddx1"), -1800.0)
    assert _approx(store.get("dpx"), 0.0)
    for i in range(1, 5):
        assert _approx(store.get(f"delx{i}"), 0.0)
        assert _approx(store.get(f"ddx{i}"), -1800.0)


def test_mact_2_second_step_uses_stored_slope_and_rate_limit():
    vehicle, act = _ready(mact=2, dpcx=10.0, dqcx=0.0, drcx=0.0)
    ctx = _ctx(DT)
    act.execute(vehicle, ctx)
    act.execute(vehicle, ctx)
    store = vehicle.store
    # step 1: ddx=-1800. step 2: |ddx|>600 → ddx=-600; dx=integrate(-600,0,0,0.001)=-0.3
    # edx=-9.7; ddxd_new=360000*-9.7-2*0.7*600*-600=-2988000
    # ddx=integrate(-2988000,-3600000,-600,0.001)=-3894; iflag zeros ddxd
    assert _approx(store.get("dx1"), -0.3)
    assert _approx(store.get("delx1"), -0.3)
    assert _approx(store.get("dxd1"), -600.0)
    assert _approx(store.get("ddx1"), -3894.0)
    assert _approx(store.get("ddxd1"), 0.0)


def test_mact_2_position_limit_zeros_rate_when_same_sign():
    vehicle, act = _ready(mact=2, dpcx=10.0, dqcx=0.0, drcx=0.0)
    store = vehicle.store
    for i in range(1, 5):
        store.set(f"dx{i}", 30.0)
        store.set(f"ddx{i}", 10.0)
    act.execute(vehicle, _ctx(DT))
    # abs(30)>28 → dx=28, same-sign rate zeroed; integrate(0,0,28,0.001)=28
    for i in range(1, 5):
        assert _approx(store.get(f"dx{i}"), 28.0)
        assert _approx(store.get(f"delx{i}"), 28.0)
        assert _approx(store.get(f"dxd{i}"), 0.0)


def test_mact_3_raises():
    vehicle, act = _ready(mact=3, dpcx=10.0)
    with pytest.raises(ValueError):
        act.execute(vehicle, _ctx())
    assert _approx(vehicle.store.get("dpx"), 0.0)
    assert _approx(vehicle.store.get("delx1"), 0.0)


def test_cadac_sign_zero_is_plus_one_not_numpy_sign():
    vehicle, act = _ready(mact=0, dlimx=-5.0, dpcx=0.0, dqcx=0.0, drcx=0.0)
    act.execute(vehicle, _ctx())
    store = vehicle.store
    for name in ("delx1", "delx2", "delx3", "delx4"):
        assert _approx(store.get(name), -5.0), name
    assert _approx(store.get("dpx"), 5.0)
    assert np.sign(0.0) == 0.0


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.sam6.actuator as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src
    assert "Hyper6Actuator" not in src
    assert "Plane6Actuator" not in src
    assert "np.sign" not in src
    assert "_cadac_sign" not in src


def test_terminate_exists_and_is_pass():
    vehicle, act = _ready(mact=0, dpcx=10.0)
    assert act.terminate(vehicle, _ctx()) is None
    assert _approx(vehicle.store.get("dpx"), 0.0)
    assert _approx(vehicle.store.get("delx1"), 0.0)
