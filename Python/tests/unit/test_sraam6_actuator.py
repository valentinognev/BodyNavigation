from types import SimpleNamespace

import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.sraam6.actuator import Sraam6Actuator

RTOL = 1e-12
ATOL = 1e-14

DT = 0.001
DLIMX = 28.0
DDLIMX = 600.0
WNACT = 100.0
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
    dxd=(0.0, 0.0, 0.0, 0.0),
    dx=(0.0, 0.0, 0.0, 0.0),
    ddxd=(0.0, 0.0, 0.0, 0.0),
    ddx=(0.0, 0.0, 0.0, 0.0),
):
    vehicle = SimpleNamespace(store=StateStore())
    act = Sraam6Actuator()
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
    assert Sraam6Actuator.name == "actuator"
    assert Sraam6Actuator().name == "actuator"


def test_define_registers_cpp_fields_not_commands():
    vehicle = SimpleNamespace(store=StateStore())
    Sraam6Actuator().define(vehicle)
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


def test_mact_0_limits_pitch_command_to_dlimx():
    vehicle, act = _ready(mact=0, dpcx=0.0, dqcx=40.0, drcx=0.0, dlimx=28.0)
    act.execute(vehicle, _ctx())
    store = vehicle.store
    assert abs(store.get("dqx")) <= 28.0
    assert store.get("dqx") == pytest.approx(28.0, rel=RTOL, abs=ATOL)
    assert store.get("dpx") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert store.get("drx") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert store.get("delcx1") == pytest.approx(40.0, rel=RTOL, abs=ATOL)
    assert store.get("delcx2") == pytest.approx(40.0, rel=RTOL, abs=ATOL)
    assert store.get("delcx3") == pytest.approx(40.0, rel=RTOL, abs=ATOL)
    assert store.get("delcx4") == pytest.approx(40.0, rel=RTOL, abs=ATOL)
    for name in DIA_DELX:
        assert store.get(name) == pytest.approx(28.0, rel=RTOL, abs=ATOL)
    assert store.get("dx1") == 0.0
    assert store.get("ddx1") == 0.0


def test_mact_2_step_lags_command_and_respects_dlimx():
    vehicle, act = _ready(mact=2, dpcx=0.0, dqcx=1.0, drcx=0.0)
    act.execute(vehicle, _ctx(DT))
    store = vehicle.store
    assert abs(store.get("dqx")) <= DLIMX
    assert store.get("dqx") != 1.0
    for name in DIA_DELX + STATE_DX + STATE_DDX:
        assert np.isfinite(store.get(name))
    assert abs(store.get("ddx1")) <= DDLIMX


def test_mact_1_uses_position_limit_path():
    vehicle, act = _ready(mact=1, dpcx=0.0, dqcx=40.0, drcx=0.0, dlimx=28.0)
    act.execute(vehicle, _ctx())
    store = vehicle.store
    assert abs(store.get("dqx")) <= 28.0
    assert store.get("dqx") == pytest.approx(28.0, rel=RTOL, abs=ATOL)
    assert store.get("dpx") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert store.get("drx") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    for name in DIA_DELX:
        assert store.get(name) == pytest.approx(28.0, rel=RTOL, abs=ATOL)
    assert store.get("dx1") == 0.0
    assert store.get("ddx1") == 0.0


def test_mact_3_raises():
    vehicle, act = _ready(mact=3, dqcx=1.0)
    with pytest.raises(ValueError):
        act.execute(vehicle, _ctx())


def test_mact_negative_raises():
    vehicle, act = _ready(mact=-1, dqcx=1.0)
    with pytest.raises(ValueError):
        act.execute(vehicle, _ctx())


def test_mact_0_four_fin_mix_round_trip():
    vehicle, act = _ready(mact=0, dpcx=1.0, dqcx=2.0, drcx=3.0)
    act.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("delcx1") == pytest.approx(-1.0 + 2.0 - 3.0, rel=RTOL, abs=ATOL)
    assert store.get("delcx2") == pytest.approx(-1.0 + 2.0 + 3.0, rel=RTOL, abs=ATOL)
    assert store.get("delcx3") == pytest.approx(+1.0 + 2.0 - 3.0, rel=RTOL, abs=ATOL)
    assert store.get("delcx4") == pytest.approx(+1.0 + 2.0 + 3.0, rel=RTOL, abs=ATOL)
    assert store.get("dpx") == pytest.approx(1.0, rel=RTOL, abs=ATOL)
    assert store.get("dqx") == pytest.approx(2.0, rel=RTOL, abs=ATOL)
    assert store.get("drx") == pytest.approx(3.0, rel=RTOL, abs=ATOL)


def test_mact_2_first_step_matches_cadac_formulas():
    vehicle, act = _ready(mact=2, dpcx=0.0, dqcx=1.0, drcx=0.0)
    act.execute(vehicle, _ctx(DT))
    store = vehicle.store
    ddxd_new = WNACT * WNACT * 1.0
    ddx = (ddxd_new + 0.0) * DT / 2.0
    assert store.get("delcx1") == pytest.approx(1.0, rel=RTOL, abs=ATOL)
    assert store.get("dx1") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert store.get("dqx") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert store.get("ddxd1") == pytest.approx(ddxd_new, rel=RTOL, abs=ATOL)
    assert store.get("ddx1") == pytest.approx(ddx, rel=RTOL, abs=ATOL)
    assert store.get("ddx1") == pytest.approx(5.0, rel=RTOL, abs=ATOL)


def test_mact_2_uses_degrees_not_radians():
    vehicle, act = _ready(mact=2, dqcx=1.0)
    act.execute(vehicle, _ctx())
    rad_like = 1.0 * np.pi / 180.0
    assert vehicle.store.get("dqx") != pytest.approx(rad_like, rel=1e-6)
    assert abs(vehicle.store.get("ddx1")) > 1.0


def test_cadac_sign_zero_is_plus_one_not_numpy_sign():
    vehicle, act = _ready(mact=0, dlimx=-5.0, dpcx=0.0, dqcx=0.0, drcx=0.0)
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
        dqcx=1.0,
        dx=(29.0, 29.0, 29.0, 29.0),
        ddx=(10.0, 10.0, 10.0, 10.0),
    )
    act.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("dx1") == pytest.approx(DLIMX, rel=RTOL, abs=ATOL)
    assert store.get("dqx") == pytest.approx(DLIMX, rel=RTOL, abs=ATOL)
    edx = 1.0 - DLIMX
    ddxd_new = WNACT * WNACT * edx
    ddx = (ddxd_new + 0.0) * DT / 2.0
    assert store.get("ddxd1") == pytest.approx(ddxd_new, rel=RTOL, abs=ATOL)
    assert store.get("ddx1") == pytest.approx(ddx, rel=RTOL, abs=ATOL)


def test_rate_limit_iflag_zeros_ddxd_when_same_sign():
    vehicle, act = _ready(
        mact=2,
        dqcx=100.0,
        ddx=(700.0, 700.0, 700.0, 700.0),
    )
    act.execute(vehicle, _ctx())
    store = vehicle.store
    dx = (DDLIMX + 0.0) * DT / 2.0
    edx = 100.0 - dx
    ddxd_new = WNACT * WNACT * edx - 2.0 * ZETACT * WNACT * DDLIMX
    assert ddxd_new > 0.0
    assert store.get("dx1") == pytest.approx(dx, rel=RTOL, abs=ATOL)
    assert store.get("ddxd1") == pytest.approx(0.0, rel=RTOL, abs=ATOL)


def test_stored_slope_second_step():
    vehicle, act = _ready(mact=2, dqcx=1.0)
    ctx = _ctx()
    act.execute(vehicle, ctx)
    first = vehicle.store.get("dqx")
    act.execute(vehicle, ctx)
    second = vehicle.store.get("dqx")
    dx = (5.0 + 0.0) * DT / 2.0
    edx = 1.0 - dx
    ddxd_new = WNACT * WNACT * edx - 2.0 * ZETACT * WNACT * 5.0
    ddx = 5.0 + (ddxd_new + 10000.0) * DT / 2.0
    assert first == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert second == pytest.approx(dx, rel=RTOL, abs=ATOL)
    assert second != 1.0
    assert abs(second) <= DLIMX
    assert vehicle.store.get("ddx1") == pytest.approx(ddx, rel=RTOL, abs=ATOL)


def test_does_not_require_time():
    vehicle, act = _ready(mact=0, dqcx=1.0)
    assert "time" not in vehicle.store.names()
    act.execute(vehicle, _ctx())
    assert "time" not in vehicle.store.names()


def test_initialize_and_terminate_are_pass():
    vehicle, act = _ready(mact=0, dqcx=1.0)
    assert vehicle.store.get("dqx") == 0.0
    assert act.terminate(vehicle, _ctx()) is None
    assert vehicle.store.get("dqx") == 0.0
