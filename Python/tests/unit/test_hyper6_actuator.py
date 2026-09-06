from types import SimpleNamespace

import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.hyper6.actuator import Hyper6Actuator

RTOL = 1e-12
ATOL = 1e-14

DT = 0.01
DLIMX = 20.0
DDLIMX = 400.0
WNACT = 50.0
ZETACT = 0.7

DEFINED = (
    "mact",
    "dlimx",
    "ddlimx",
    "wnact",
    "zetact",
    "delax",
    "delex",
    "delrx",
    "elvlx",
    "elvrx",
    "elvlcx",
    "elvrcx",
    "DXD",
    "DX",
    "DDXD",
    "DDX",
)
INT_FIELDS = ("mact",)
VEC_FIELDS = ("DXD", "DX", "DDXD", "DDX")
DIA_PLOT = ("elvlx", "elvrx")
DIA_NO_OUT = ("elvlcx", "elvrcx")
COMMANDS = ("delacx", "delecx", "delrcx")
CONTROL_FORCE_VEHICLE = (
    "ancomx",
    "alcomx",
    "FAPB",
    "FMB",
    "vmass",
)


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


def _commands(store, delacx=0.0, delecx=0.0, delrcx=0.0):
    store.define(Field("delacx", delacx, "real", "out", "control"))
    store.define(Field("delecx", delecx, "real", "out", "control"))
    store.define(Field("delrcx", delrcx, "real", "out", "control"))


def _ready(
    *,
    mact=2,
    dlimx=DLIMX,
    ddlimx=DDLIMX,
    wnact=WNACT,
    zetact=ZETACT,
    delacx=0.0,
    delecx=1.0,
    delrcx=0.0,
    dxd=(0.0, 0.0, 0.0),
    dx=(0.0, 0.0, 0.0),
    ddxd=(0.0, 0.0, 0.0),
    ddx=(0.0, 0.0, 0.0),
):
    vehicle = SimpleNamespace(store=StateStore())
    act = Hyper6Actuator()
    act.define(vehicle)
    _commands(vehicle.store, delacx=delacx, delecx=delecx, delrcx=delrcx)
    store = vehicle.store
    store.set("mact", mact)
    store.set("dlimx", dlimx)
    store.set("ddlimx", ddlimx)
    store.set("wnact", wnact)
    store.set("zetact", zetact)
    store.set("DXD", dxd)
    store.set("DX", dx)
    store.set("DDXD", ddxd)
    store.set("DDX", ddx)
    act.initialize(vehicle, _ctx())
    return vehicle, act


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _actuator_scnd(actcx, dxd, dx, ddxd, ddx, dlimx, ddlimx, wnact, zetact, dt):
    dxd = np.asarray(dxd, dtype=float).copy()
    dx = np.asarray(dx, dtype=float).copy()
    ddxd = np.asarray(ddxd, dtype=float).copy()
    ddx = np.asarray(ddx, dtype=float).copy()
    for i in range(3):
        if abs(dx[i]) > dlimx:
            dx[i] = dlimx * _sign(dx[i])
            if dx[i] * ddx[i] > 0:
                ddx[i] = 0.0
        iflag = 0
        if abs(ddx[i]) > ddlimx:
            iflag = 1
            ddx[i] = ddlimx * _sign(ddx[i])
        dxd_new = ddx[i]
        dx[i] = integrate(dxd_new, dxd[i], dx[i], dt)
        dxd[i] = dxd_new
        edx = actcx[i] - dx[i]
        ddxd_new = wnact * wnact * edx - 2.0 * zetact * wnact * dxd[i]
        ddx[i] = integrate(ddxd_new, ddxd[i], ddx[i], dt)
        ddxd[i] = ddxd_new
        if iflag and ddx[i] * ddxd[i] > 0:
            ddxd[i] = 0.0
    return dx, dxd, ddx, ddxd


def _actcx(store):
    elvlcx = store.get("delecx") + store.get("delacx")
    elvrcx = store.get("delecx") - store.get("delacx")
    return np.array([elvlcx, elvrcx, store.get("delrcx")], dtype=float)


def _from_actx(actx, elvlcx, elvrcx):
    elvlx = float(actx[0])
    elvrx = float(actx[1])
    delrx = float(actx[2])
    return {
        "delax": (elvlx - elvrx) / 2.0,
        "delex": (elvlx + elvrx) / 2.0,
        "delrx": delrx,
        "elvlx": elvlx,
        "elvrx": elvrx,
        "elvlcx": elvlcx,
        "elvrcx": elvrcx,
    }


def _expected(store, dt):
    mact = store.get("mact")
    dlimx = store.get("dlimx")
    actcx = _actcx(store)
    elvlcx = float(actcx[0])
    elvrcx = float(actcx[1])
    if mact == 0:
        actx = actcx.copy()
        for i in range(3):
            if abs(actx[i]) > dlimx:
                actx[i] = dlimx * _sign(actx[i])
        out = _from_actx(actx, elvlcx, elvrcx)
        out["ACTX"] = actx
        out["DXD"] = np.asarray(store.get("DXD"), dtype=float).copy()
        out["DX"] = np.asarray(store.get("DX"), dtype=float).copy()
        out["DDXD"] = np.asarray(store.get("DDXD"), dtype=float).copy()
        out["DDX"] = np.asarray(store.get("DDX"), dtype=float).copy()
        return out
    if mact == 2:
        dx, dxd, ddx, ddxd = _actuator_scnd(
            actcx,
            store.get("DXD"),
            store.get("DX"),
            store.get("DDXD"),
            store.get("DDX"),
            dlimx,
            store.get("ddlimx"),
            store.get("wnact"),
            store.get("zetact"),
            dt,
        )
        out = _from_actx(dx, elvlcx, elvrcx)
        out["ACTX"] = dx.copy()
        out["DXD"] = dxd
        out["DX"] = dx
        out["DDXD"] = ddxd
        out["DDX"] = ddx
        return out
    raise ValueError(f"unknown mact {mact}")


def _assert_step(store, want):
    for name in (
        "delax",
        "delex",
        "delrx",
        "elvlx",
        "elvrx",
        "elvlcx",
        "elvrcx",
    ):
        assert _approx(store.get(name), want[name])
    np.testing.assert_allclose(store.get("DXD"), want["DXD"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("DX"), want["DX"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("DDXD"), want["DDXD"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("DDX"), want["DDX"], rtol=RTOL, atol=ATOL)


def test_name_is_actuator():
    assert Hyper6Actuator().name == "actuator"


def test_define_registers_cpp_fields_not_commands():
    vehicle = SimpleNamespace(store=StateStore())
    Hyper6Actuator().define(vehicle)
    store = vehicle.store
    for name in DEFINED:
        assert name in store.names()
        assert store.field(name).module == "actuator"
    for name in INT_FIELDS:
        assert store.get(name) == 0
        assert store.field(name).type == "int"
        assert store.field(name).role == "data"
    for name in ("dlimx", "ddlimx", "wnact", "zetact"):
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "data"
    for name in ("delax", "delex", "delrx"):
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "out"
        assert store.field(name).outputs == ("scrn", "plot")
    for name in DIA_PLOT:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "dia"
        assert store.field(name).outputs == ("plot",)
    for name in DIA_NO_OUT:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "dia"
        assert store.field(name).outputs == ()
    zeros = np.zeros(3)
    for name in VEC_FIELDS:
        np.testing.assert_array_equal(store.get(name), zeros)
        assert store.get(name).shape == (3,)
        assert store.field(name).type == "vec"
        assert store.field(name).role == "state"
        assert store.field(name).outputs == ()
    for name in COMMANDS:
        assert name not in store.names()
    for name in CONTROL_FORCE_VEHICLE:
        assert name not in store.names()


def test_initialize_leaves_states_zero():
    vehicle, _act = _ready(mact=2, delecx=1.0)
    store = vehicle.store
    assert store.get("delex") == 0.0
    np.testing.assert_array_equal(store.get("DX"), np.zeros(3))
    np.testing.assert_array_equal(store.get("DDX"), np.zeros(3))
    np.testing.assert_array_equal(store.get("DXD"), np.zeros(3))
    np.testing.assert_array_equal(store.get("DDXD"), np.zeros(3))


def test_mact_2_step_command_elevator_lags_and_respects_dlimx():
    vehicle, act = _ready(mact=2, delecx=1.0, delacx=0.0, delrcx=0.0)
    act.execute(vehicle, _ctx(DT))
    store = vehicle.store
    delex = store.get("delex")
    assert delex != 1.0
    assert abs(delex) <= DLIMX
    assert store.get("delax") == 0.0
    assert store.get("delrx") == 0.0
    assert store.get("DDX")[0] != 0.0
    assert abs(store.get("DDX")[0]) <= DDLIMX
    assert abs(store.get("DDX")[1]) <= DDLIMX


def test_mact_2_one_step_matches_cadac_formulas():
    vehicle, act = _ready(mact=2, delecx=1.0)
    want = _expected(vehicle.store, DT)
    act.execute(vehicle, _ctx())
    _assert_step(vehicle.store, want)
    assert vehicle.store.get("delex") != 1.0
    assert abs(vehicle.store.get("delex")) <= DLIMX


def test_mact_2_uses_degrees_not_radians():
    vehicle, act = _ready(mact=2, delecx=1.0)
    act.execute(vehicle, _ctx())
    rad_like = 1.0 * np.pi / 180.0
    assert vehicle.store.get("delex") != pytest.approx(rad_like, rel=1e-6)
    assert abs(vehicle.store.get("DDX")[0]) > 1.0
    assert abs(vehicle.store.get("DDX")[1]) > 1.0


def test_mact_2_mixes_aileron_and_elevator_to_elevons():
    vehicle, act = _ready(mact=2, delacx=1.0, delecx=1.0, delrcx=0.0)
    want = _expected(vehicle.store, DT)
    act.execute(vehicle, _ctx())
    _assert_step(vehicle.store, want)
    assert vehicle.store.get("elvlcx") == 2.0
    assert vehicle.store.get("elvrcx") == 0.0
    assert vehicle.store.get("DDX")[0] != vehicle.store.get("DDX")[1]


def test_mact_0_copies_elevon_commands_with_position_limit():
    vehicle, act = _ready(
        mact=0,
        delacx=25.0,
        delecx=1.0,
        delrcx=-30.0,
        dx=(1.0, 2.0, 3.0),
        ddx=(4.0, 5.0, 6.0),
        dxd=(7.0, 8.0, 9.0),
        ddxd=(10.0, 11.0, 12.0),
    )
    want = _expected(vehicle.store, DT)
    act.execute(vehicle, _ctx())
    _assert_step(vehicle.store, want)
    assert vehicle.store.get("elvlcx") == 26.0
    assert vehicle.store.get("elvrcx") == -24.0
    assert vehicle.store.get("elvlx") == DLIMX
    assert vehicle.store.get("elvrx") == -DLIMX
    assert vehicle.store.get("delax") == DLIMX
    assert vehicle.store.get("delex") == 0.0
    assert vehicle.store.get("delrx") == -DLIMX
    np.testing.assert_array_equal(vehicle.store.get("DX"), (1.0, 2.0, 3.0))
    np.testing.assert_array_equal(vehicle.store.get("DDX"), (4.0, 5.0, 6.0))


def test_mact_1_raises():
    vehicle, act = _ready(mact=1, delecx=1.0)
    with pytest.raises(ValueError):
        act.execute(vehicle, _ctx())


def test_mact_other_raises():
    vehicle, act = _ready(mact=3, delecx=1.0)
    with pytest.raises(ValueError):
        act.execute(vehicle, _ctx())


def test_position_limit_zeros_rate_when_same_sign():
    vehicle, act = _ready(
        mact=2,
        delecx=1.0,
        dx=(21.0, 21.0, 0.0),
        ddx=(10.0, 10.0, 0.0),
    )
    want = _expected(vehicle.store, DT)
    assert want["DX"][0] == DLIMX
    assert want["DX"][1] == DLIMX
    act.execute(vehicle, _ctx())
    _assert_step(vehicle.store, want)


def test_position_limit_keeps_rate_when_opposite_sign():
    vehicle, act = _ready(
        mact=2,
        delecx=1.0,
        dx=(21.0, 21.0, 0.0),
        ddx=(-10.0, -10.0, 0.0),
    )
    want = _expected(vehicle.store, DT)
    act.execute(vehicle, _ctx())
    _assert_step(vehicle.store, want)
    assert want["DX"][0] != DLIMX
    assert want["DX"][0] < DLIMX


def test_rate_limit_iflag_zeros_ddxd_when_same_sign():
    vehicle, act = _ready(
        mact=2,
        delecx=100.0,
        ddx=(500.0, 500.0, 0.0),
    )
    want = _expected(vehicle.store, DT)
    assert want["DDXD"][0] == 0.0
    assert want["DDXD"][1] == 0.0
    act.execute(vehicle, _ctx())
    _assert_step(vehicle.store, want)


def test_rate_limit_keeps_ddxd_when_opposite_sign():
    vehicle, act = _ready(
        mact=2,
        delecx=1.0,
        ddx=(500.0, 500.0, 0.0),
    )
    want = _expected(vehicle.store, DT)
    assert want["DDXD"][0] != 0.0
    act.execute(vehicle, _ctx())
    _assert_step(vehicle.store, want)


def test_cadac_sign_zero_is_plus_one_not_numpy_sign():
    vehicle, act = _ready(
        mact=0,
        dlimx=-5.0,
        delacx=0.0,
        delecx=0.0,
        delrcx=0.0,
    )
    act.execute(vehicle, _ctx())
    assert vehicle.store.get("elvlx") == -5.0
    assert vehicle.store.get("elvrx") == -5.0
    assert vehicle.store.get("delrx") == -5.0
    assert vehicle.store.get("delex") == -5.0
    assert vehicle.store.get("delax") == 0.0
    assert np.sign(0.0) == 0.0


def test_stored_slope_second_step():
    vehicle, act = _ready(mact=2, delecx=1.0)
    ctx = _ctx()
    want1 = _expected(vehicle.store, DT)
    act.execute(vehicle, ctx)
    _assert_step(vehicle.store, want1)
    want2 = _expected(vehicle.store, DT)
    act.execute(vehicle, ctx)
    _assert_step(vehicle.store, want2)
    assert vehicle.store.get("delex") != want1["delex"]
    assert abs(vehicle.store.get("delex")) <= DLIMX
    assert vehicle.store.get("delex") != 1.0


def test_terminate_exists_and_is_pass():
    vehicle, act = _ready(mact=0, delecx=1.0)
    assert act.terminate(vehicle, _ctx()) is None
