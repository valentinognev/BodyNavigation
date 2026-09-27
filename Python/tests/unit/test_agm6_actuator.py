from types import SimpleNamespace

import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.agm6.actuator import Agm6Actuator

RTOL = 1e-12
ATOL = 1e-14

DT = 0.001
DLIMX = 20.0
DDLIMX = 600.0
WNACT = 62.8
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
INT_FIELDS = ("mact",)
DATA_REALS = ("dlimx", "ddlimx", "wnact", "zetact")
OUT_PLOT = ("dpx", "dqx", "drx")
DIA_DELX = ("delx1", "delx2", "delx3", "delx4")
DIA_DELCX = ("delcx1", "delcx2", "delcx3", "delcx4")
STATE_DXD = ("dxd1", "dxd2", "dxd3", "dxd4")
STATE_DX = ("dx1", "dx2", "dx3", "dx4")
STATE_DDXD = ("ddxd1", "ddxd2", "ddxd3", "ddxd4")
STATE_DDX = ("ddx1", "ddx2", "ddx3", "ddx4")
COMMANDS = ("dpcx", "dqcx", "drcx")
PLANE6_COMMANDS = ("delacx", "delecx", "delrcx")
VEC_LAYOUT = ("DXD", "DX", "DDXD", "DDX")


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _ctx(dt=DT):
    return SimContext(
        sim_time=0.0,
        int_step=dt,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _commands(store, dpcx=0.0, dqcx=0.0, drcx=0.0):
    store.define(Field("dpcx", dpcx, "real", "out", "control"))
    store.define(Field("dqcx", dqcx, "real", "out", "control"))
    store.define(Field("drcx", drcx, "real", "out", "control"))


def _mix(dpcx, dqcx, drcx):
    return (
        -dpcx + dqcx - drcx,
        -dpcx + dqcx + drcx,
        +dpcx + dqcx - drcx,
        +dpcx + dqcx + drcx,
    )


def _back_convert(delx1, delx2, delx3, delx4):
    dpx = (-delx1 - delx2 + delx3 + delx4) / 4.0
    dqx = (+delx1 + delx2 + delx3 + delx4) / 4.0
    drx = (-delx1 + delx2 - delx3 + delx4) / 4.0
    return dpx, dqx, drx


def _limit_pos(delcx, dlimx):
    delx = delcx
    if abs(delx) > dlimx:
        delx = dlimx * _sign(delx)
    return delx


def _fin_scnd(delcx, dxd, dx, ddxd, ddx, dlimx, ddlimx, wnact, zetact, dt):
    if abs(dx) > dlimx:
        dx = dlimx * _sign(dx)
        if dx * ddx > 0:
            ddx = 0.0
    iflag = 0
    if abs(ddx) > ddlimx:
        iflag = 1
        ddx = ddlimx * _sign(ddx)
    dxd_new = ddx
    dx = integrate(dxd_new, dxd, dx, dt)
    dxd = dxd_new
    edx = delcx - dx
    ddxd_new = wnact * wnact * edx - 2.0 * zetact * wnact * dxd
    ddx = integrate(ddxd_new, ddxd, ddx, dt)
    ddxd = ddxd_new
    if iflag and ddx * ddxd > 0:
        ddxd = 0.0
    return dxd, dx, ddxd, ddx


def _expected(store, dt):
    mact = store.get("mact")
    dlimx = store.get("dlimx")
    dpcx = store.get("dpcx")
    dqcx = store.get("dqcx")
    drcx = store.get("drcx")
    delcx1, delcx2, delcx3, delcx4 = _mix(dpcx, dqcx, drcx)
    out = {
        "delcx1": delcx1,
        "delcx2": delcx2,
        "delcx3": delcx3,
        "delcx4": delcx4,
    }
    for name in STATE_DXD + STATE_DX + STATE_DDXD + STATE_DDX:
        out[name] = store.get(name)
    if mact < 2:
        delx1 = _limit_pos(delcx1, dlimx)
        delx2 = _limit_pos(delcx2, dlimx)
        delx3 = _limit_pos(delcx3, dlimx)
        delx4 = _limit_pos(delcx4, dlimx)
        dpx, dqx, drx = _back_convert(delx1, delx2, delx3, delx4)
        out.update(
            {
                "delx1": delx1,
                "delx2": delx2,
                "delx3": delx3,
                "delx4": delx4,
                "dpx": dpx,
                "dqx": dqx,
                "drx": drx,
            }
        )
        return out
    if mact == 2:
        ddlimx = store.get("ddlimx")
        wnact = store.get("wnact")
        zetact = store.get("zetact")
        dxd1, dx1, ddxd1, ddx1 = _fin_scnd(
            delcx1,
            store.get("dxd1"),
            store.get("dx1"),
            store.get("ddxd1"),
            store.get("ddx1"),
            dlimx,
            ddlimx,
            wnact,
            zetact,
            dt,
        )
        dxd2, dx2, ddxd2, ddx2 = _fin_scnd(
            delcx2,
            store.get("dxd2"),
            store.get("dx2"),
            store.get("ddxd2"),
            store.get("ddx2"),
            dlimx,
            ddlimx,
            wnact,
            zetact,
            dt,
        )
        dxd3, dx3, ddxd3, ddx3 = _fin_scnd(
            delcx3,
            store.get("dxd3"),
            store.get("dx3"),
            store.get("ddxd3"),
            store.get("ddx3"),
            dlimx,
            ddlimx,
            wnact,
            zetact,
            dt,
        )
        dxd4, dx4, ddxd4, ddx4 = _fin_scnd(
            delcx4,
            store.get("dxd4"),
            store.get("dx4"),
            store.get("ddxd4"),
            store.get("ddx4"),
            dlimx,
            ddlimx,
            wnact,
            zetact,
            dt,
        )
        dpx, dqx, drx = _back_convert(dx1, dx2, dx3, dx4)
        out.update(
            {
                "dxd1": dxd1,
                "dxd2": dxd2,
                "dxd3": dxd3,
                "dxd4": dxd4,
                "dx1": dx1,
                "dx2": dx2,
                "dx3": dx3,
                "dx4": dx4,
                "ddxd1": ddxd1,
                "ddxd2": ddxd2,
                "ddxd3": ddxd3,
                "ddxd4": ddxd4,
                "ddx1": ddx1,
                "ddx2": ddx2,
                "ddx3": ddx3,
                "ddx4": ddx4,
                "delx1": dx1,
                "delx2": dx2,
                "delx3": dx3,
                "delx4": dx4,
                "dpx": dpx,
                "dqx": dqx,
                "drx": drx,
            }
        )
        return out
    raise ValueError(f"unknown mact {mact}")


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _assert_step(store, want):
    for name in (
        "dpx",
        "dqx",
        "drx",
        "delx1",
        "delx2",
        "delx3",
        "delx4",
        "delcx1",
        "delcx2",
        "delcx3",
        "delcx4",
        *STATE_DXD,
        *STATE_DX,
        *STATE_DDXD,
        *STATE_DDX,
    ):
        assert _approx(store.get(name), want[name])


def _ready(
    *,
    mact=0,
    dlimx=DLIMX,
    ddlimx=DDLIMX,
    wnact=WNACT,
    zetact=ZETACT,
    dpcx=0.0,
    dqcx=5.0,
    drcx=0.0,
    dxd=(0.0, 0.0, 0.0, 0.0),
    dx=(0.0, 0.0, 0.0, 0.0),
    ddxd=(0.0, 0.0, 0.0, 0.0),
    ddx=(0.0, 0.0, 0.0, 0.0),
):
    vehicle = SimpleNamespace(store=StateStore())
    act = Agm6Actuator()
    act.define(vehicle)
    _commands(vehicle.store, dpcx=dpcx, dqcx=dqcx, drcx=drcx)
    store = vehicle.store
    store.set("mact", mact)
    store.set("dlimx", dlimx)
    store.set("ddlimx", ddlimx)
    store.set("wnact", wnact)
    store.set("zetact", zetact)
    for i, name in enumerate(STATE_DXD):
        store.set(name, dxd[i])
    for i, name in enumerate(STATE_DX):
        store.set(name, dx[i])
    for i, name in enumerate(STATE_DDXD):
        store.set(name, ddxd[i])
    for i, name in enumerate(STATE_DDX):
        store.set(name, ddx[i])
    act.initialize(vehicle, _ctx())
    return vehicle, act


def test_name_is_actuator():
    assert Agm6Actuator.name == "actuator"
    assert Agm6Actuator().name == "actuator"


def test_define_registers_cpp_fields_not_commands():
    vehicle = SimpleNamespace(store=StateStore())
    Agm6Actuator().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    for name in DEFINED:
        assert store.field(name).module == "actuator"
    for name in INT_FIELDS:
        assert store.get(name) == 0
        assert store.field(name).type == "int"
        assert store.field(name).role == "data"
        assert store.field(name).outputs == ()
    for name in DATA_REALS:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "data"
        assert store.field(name).outputs == ()
    for name in OUT_PLOT:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "out"
        assert store.field(name).outputs == ("plot",)
    for name in DIA_DELX + DIA_DELCX:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "diag"
        assert store.field(name).outputs == ()
    for name in STATE_DXD + STATE_DX + STATE_DDXD + STATE_DDX:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "state"
        assert store.field(name).outputs == ()
    for name in COMMANDS + PLANE6_COMMANDS + VEC_LAYOUT:
        assert name not in store.names()
    assert "time" not in store.names()


def test_initialize_is_pass():
    vehicle, _act = _ready(mact=0, dqcx=5.0)
    store = vehicle.store
    assert store.get("dqx") == 0.0
    assert store.get("dx1") == 0.0
    assert store.get("ddx1") == 0.0


def test_mact_0_pitch_command_round_trip_within_dlimx():
    vehicle, act = _ready(mact=0, dlimx=20.0, dpcx=0.0, dqcx=5.0, drcx=0.0)
    want = _expected(vehicle.store, DT)
    act.execute(vehicle, _ctx())
    _assert_step(vehicle.store, want)
    store = vehicle.store
    assert abs(store.get("dqx")) <= 20.0
    assert store.get("dpx") == 0.0
    assert store.get("dqx") == 5.0
    assert store.get("drx") == 0.0
    assert store.get("delcx1") == 5.0
    assert store.get("delcx2") == 5.0
    assert store.get("delcx3") == 5.0
    assert store.get("delcx4") == 5.0
    assert store.get("dx1") == 0.0
    assert store.get("ddx1") == 0.0


def test_mact_0_four_fin_mix_round_trip():
    vehicle, act = _ready(mact=0, dpcx=1.0, dqcx=2.0, drcx=3.0)
    want = _expected(vehicle.store, DT)
    act.execute(vehicle, _ctx())
    _assert_step(vehicle.store, want)
    store = vehicle.store
    assert store.get("delcx1") == -1.0 + 2.0 - 3.0
    assert store.get("delcx2") == -1.0 + 2.0 + 3.0
    assert store.get("delcx3") == +1.0 + 2.0 - 3.0
    assert store.get("delcx4") == +1.0 + 2.0 + 3.0
    assert store.get("dpx") == 1.0
    assert store.get("dqx") == 2.0
    assert store.get("drx") == 3.0


def test_mact_0_position_limit_then_back_convert():
    vehicle, act = _ready(mact=0, dlimx=3.0, dpcx=1.0, dqcx=2.0, drcx=3.0)
    want = _expected(vehicle.store, DT)
    act.execute(vehicle, _ctx())
    _assert_step(vehicle.store, want)
    store = vehicle.store
    assert store.get("delcx2") == 4.0
    assert store.get("delcx4") == 6.0
    assert store.get("delx2") == 3.0
    assert store.get("delx4") == 3.0
    assert store.get("dpx") != 1.0
    assert abs(store.get("delx1")) <= 3.0
    assert abs(store.get("delx2")) <= 3.0
    assert abs(store.get("delx3")) <= 3.0
    assert abs(store.get("delx4")) <= 3.0


def test_mact_1_is_position_limit_like_mact_0():
    vehicle, act = _ready(mact=1, dlimx=20.0, dpcx=0.0, dqcx=5.0, drcx=0.0)
    want = _expected(vehicle.store, DT)
    act.execute(vehicle, _ctx())
    _assert_step(vehicle.store, want)
    assert vehicle.store.get("dqx") == 5.0


def test_mact_2_step_finite_fins_matches_cadac():
    vehicle, act = _ready(
        mact=2,
        dlimx=20.0,
        ddlimx=600.0,
        wnact=62.8,
        zetact=0.7,
        dpcx=0.0,
        dqcx=5.0,
        drcx=0.0,
    )
    want = _expected(vehicle.store, DT)
    act.execute(vehicle, _ctx(DT))
    _assert_step(vehicle.store, want)
    store = vehicle.store
    for name in DIA_DELX + STATE_DX + STATE_DDX:
        assert np.isfinite(store.get(name))
    assert store.get("dqx") != 5.0
    assert abs(store.get("dqx")) <= 20.0
    assert abs(store.get("ddx1")) <= 600.0


def test_mact_2_uses_degrees_not_radians():
    vehicle, act = _ready(mact=2, dqcx=5.0)
    act.execute(vehicle, _ctx())
    rad_like = 5.0 * np.pi / 180.0
    assert vehicle.store.get("dqx") != pytest.approx(rad_like, rel=1e-6)
    assert abs(vehicle.store.get("ddx1")) > 1.0


def test_mact_3_raises():
    vehicle, act = _ready(mact=3, dqcx=5.0)
    with pytest.raises(ValueError):
        act.execute(vehicle, _ctx())


def test_cadac_sign_zero_is_plus_one_not_numpy_sign():
    vehicle, act = _ready(
        mact=0,
        dlimx=-5.0,
        dpcx=0.0,
        dqcx=0.0,
        drcx=0.0,
    )
    act.execute(vehicle, _ctx())
    assert vehicle.store.get("delx1") == -5.0
    assert vehicle.store.get("delx2") == -5.0
    assert vehicle.store.get("delx3") == -5.0
    assert vehicle.store.get("delx4") == -5.0
    assert vehicle.store.get("dqx") == -5.0
    assert vehicle.store.get("dpx") == 0.0
    assert vehicle.store.get("drx") == 0.0
    assert np.sign(0.0) == 0.0


def test_position_limit_zeros_rate_when_same_sign():
    vehicle, act = _ready(
        mact=2,
        dqcx=5.0,
        dx=(21.0, 21.0, 21.0, 21.0),
        ddx=(10.0, 10.0, 10.0, 10.0),
    )
    want = _expected(vehicle.store, DT)
    assert want["dx1"] == DLIMX
    act.execute(vehicle, _ctx())
    _assert_step(vehicle.store, want)


def test_position_limit_keeps_rate_when_opposite_sign():
    vehicle, act = _ready(
        mact=2,
        dqcx=5.0,
        dx=(21.0, 21.0, 21.0, 21.0),
        ddx=(-10.0, -10.0, -10.0, -10.0),
    )
    want = _expected(vehicle.store, DT)
    act.execute(vehicle, _ctx())
    _assert_step(vehicle.store, want)
    assert want["dx1"] != DLIMX
    assert want["dx1"] < DLIMX


def test_rate_limit_iflag_zeros_ddxd_when_same_sign():
    vehicle, act = _ready(
        mact=2,
        dqcx=100.0,
        ddx=(700.0, 700.0, 700.0, 700.0),
    )
    want = _expected(vehicle.store, DT)
    assert want["ddxd1"] == 0.0
    act.execute(vehicle, _ctx())
    _assert_step(vehicle.store, want)


def test_stored_slope_second_step():
    vehicle, act = _ready(mact=2, dqcx=5.0)
    ctx = _ctx()
    want1 = _expected(vehicle.store, DT)
    act.execute(vehicle, ctx)
    _assert_step(vehicle.store, want1)
    want2 = _expected(vehicle.store, DT)
    act.execute(vehicle, ctx)
    _assert_step(vehicle.store, want2)
    assert vehicle.store.get("dqx") != want1["dqx"]
    assert abs(vehicle.store.get("dqx")) <= DLIMX
    assert vehicle.store.get("dqx") != 5.0


def test_does_not_require_time():
    vehicle, act = _ready(mact=0, dqcx=5.0)
    assert "time" not in vehicle.store.names()
    act.execute(vehicle, _ctx())
    assert "time" not in vehicle.store.names()


def test_terminate_exists_and_is_pass():
    vehicle, act = _ready(mact=0, dqcx=5.0)
    assert act.terminate(vehicle, _ctx()) is None
    assert vehicle.store.get("dqx") == 0.0
