from math import cos, sin

import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round6.rocket6.tvc import Rocket6Tvc

RTOL = 1e-12
ATOL = 1e-14

# insertion TVC data (input.asc)
MTVC = 2
GTVC = 1.0
PARM = 16.84
TVCLIMX = 10.0
DTVCLIMX = 200.0
ZETTVC = 0.7
WNTVC = 100.0
DT = 0.001
DELECX = 1.0
DELRCX = 0.0
THRUST = 1.0e6
XCG = 10.53

DEFINED = (
    "mtvc",
    "tvclimx",
    "dtvclimx",
    "wntvc",
    "zettvc",
    "factgtvc",
    "gtvc",
    "parm",
    "FPB",
    "FMPB",
    "etax",
    "zetx",
    "etacx",
    "zetcx",
    "etasd",
    "zetad",
    "etas",
    "zeta",
    "detasd",
    "dzetad",
    "detas",
    "dzeta",
)

ROLES = {
    "mtvc": "data",
    "tvclimx": "data",
    "dtvclimx": "data",
    "wntvc": "data",
    "zettvc": "data",
    "factgtvc": "data",
    "gtvc": "data",
    "parm": "data",
    "FPB": "out",
    "FMPB": "out",
    "etax": "diag",
    "zetx": "diag",
    "etacx": "diag",
    "zetcx": "diag",
    "etasd": "state",
    "zetad": "state",
    "etas": "state",
    "zeta": "state",
    "detasd": "state",
    "dzetad": "state",
    "detas": "state",
    "dzeta": "state",
}

OUTPUTS = {
    "mtvc": (),
    "tvclimx": (),
    "dtvclimx": (),
    "wntvc": (),
    "zettvc": (),
    "factgtvc": (),
    "gtvc": (),
    "parm": (),
    "FPB": (),
    "FMPB": (),
    "etax": ("plot",),
    "zetx": ("plot",),
    "etacx": (),
    "zetcx": (),
    "etasd": (),
    "zetad": (),
    "etas": (),
    "zeta": (),
    "detasd": (),
    "dzetad": (),
    "detas": (),
    "dzeta": (),
}

INT_FIELDS = ("mtvc",)
VEC_FIELDS = ("FPB", "FMPB")
NOT_DEFINED = (
    "delacx",
    "delecx",
    "delrcx",
    "thrust",
    "xcg",
    "time",
    "pdynmc",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


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


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _plant_externals(store, *, delecx=DELECX, delrcx=DELRCX, thrust=THRUST, xcg=XCG):
    store.define(Field("delecx", delecx, "real", "out", "control"))
    store.define(Field("delrcx", delrcx, "real", "out", "control"))
    store.define(Field("thrust", thrust, "real", "out", "propulsion"))
    store.define(Field("xcg", xcg, "real", "out", "propulsion"))


def _ready(
    *,
    mtvc=MTVC,
    gtvc=GTVC,
    parm=PARM,
    tvclimx=TVCLIMX,
    dtvclimx=DTVCLIMX,
    zettvc=ZETTVC,
    wntvc=WNTVC,
    delecx=DELECX,
    delrcx=DELRCX,
    thrust=THRUST,
    xcg=XCG,
    **states,
):
    vehicle = _Vehicle()
    tvc = Rocket6Tvc()
    tvc.define(vehicle)
    _plant_externals(vehicle.store, delecx=delecx, delrcx=delrcx, thrust=thrust, xcg=xcg)
    store = vehicle.store
    store.set("mtvc", mtvc)
    store.set("gtvc", gtvc)
    store.set("parm", parm)
    store.set("tvclimx", tvclimx)
    store.set("dtvclimx", dtvclimx)
    store.set("zettvc", zettvc)
    store.set("wntvc", wntvc)
    for name, value in states.items():
        store.set(name, value)
    tvc.initialize(vehicle, _ctx())
    return vehicle, tvc


def _axis_scnd(command, pos_d, pos, rate_d, rate, tvclimx, dtvclimx, wntvc, zettvc, dt):
    if abs(pos) > tvclimx * RAD:
        pos = tvclimx * RAD * _sign(pos)
        if pos * rate > 0.0:
            rate = 0.0
    iflag = 0
    if abs(rate) > dtvclimx * RAD:
        iflag = 1
        rate = dtvclimx * RAD * _sign(rate)
    pos_d_new = rate
    pos = integrate(pos_d_new, pos_d, pos, dt)
    pos_d = pos_d_new
    err = command - pos
    rate_d_new = wntvc * wntvc * err - 2.0 * zettvc * wntvc * pos_d
    rate = integrate(rate_d_new, rate_d, rate, dt)
    rate_d = rate_d_new
    if iflag and rate * rate_d > 0.0:
        rate_d = 0.0
    return pos, pos_d, rate, rate_d


def _expected(store, dt):
    mtvc = store.get("mtvc")
    if mtvc == 0:
        return {
            "FPB": np.asarray(store.get("FPB"), dtype=float).copy(),
            "FMPB": np.asarray(store.get("FMPB"), dtype=float).copy(),
            "etax": store.get("etax"),
            "zetx": store.get("zetx"),
            "etacx": store.get("etacx"),
            "zetcx": store.get("zetcx"),
            "etasd": store.get("etasd"),
            "zetad": store.get("zetad"),
            "etas": store.get("etas"),
            "zeta": store.get("zeta"),
            "detasd": store.get("detasd"),
            "dzetad": store.get("dzetad"),
            "detas": store.get("detas"),
            "dzeta": store.get("dzeta"),
        }
    if mtvc != 2:
        raise ValueError(f"unknown mtvc {mtvc}")
    gtvc = store.get("gtvc")
    parm = store.get("parm")
    xcg = store.get("xcg")
    thrust = store.get("thrust")
    etac = gtvc * store.get("delecx") * RAD
    zetc = gtvc * store.get("delrcx") * RAD
    tvclimx = store.get("tvclimx")
    dtvclimx = store.get("dtvclimx")
    wntvc = store.get("wntvc")
    zettvc = store.get("zettvc")
    etas, etasd, detas, detasd = _axis_scnd(
        etac,
        store.get("etasd"),
        store.get("etas"),
        store.get("detasd"),
        store.get("detas"),
        tvclimx,
        dtvclimx,
        wntvc,
        zettvc,
        dt,
    )
    zeta, zetad, dzeta, dzetad = _axis_scnd(
        zetc,
        store.get("zetad"),
        store.get("zeta"),
        store.get("dzetad"),
        store.get("dzeta"),
        tvclimx,
        dtvclimx,
        wntvc,
        zettvc,
        dt,
    )
    eta = etas
    zet = zeta
    fpb0 = cos(eta) * cos(zet) * thrust
    fpb1 = cos(eta) * sin(zet) * thrust
    fpb2 = -sin(eta) * thrust
    arm = parm - xcg
    return {
        "FPB": np.array([fpb0, fpb1, fpb2], dtype=float),
        "FMPB": np.array([0.0, arm * fpb2, -arm * fpb1], dtype=float),
        "etax": eta * DEG,
        "zetx": zet * DEG,
        "etacx": etac * DEG,
        "zetcx": zetc * DEG,
        "etasd": etasd,
        "zetad": zetad,
        "etas": etas,
        "zeta": zeta,
        "detasd": detasd,
        "dzetad": dzetad,
        "detas": detas,
        "dzeta": dzeta,
    }


def _assert_step(store, want):
    for name in (
        "etax",
        "zetx",
        "etacx",
        "zetcx",
        "etasd",
        "zetad",
        "etas",
        "zeta",
        "detasd",
        "dzetad",
        "detas",
        "dzeta",
    ):
        assert np.isfinite(store.get(name)), name
        assert _approx(store.get(name), want[name]), name
    np.testing.assert_allclose(store.get("FPB"), want["FPB"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FMPB"), want["FMPB"], rtol=RTOL, atol=ATOL)


def test_name_is_tvc():
    assert Rocket6Tvc().name == "tvc"


def test_define_registers_cpp_fields_not_externals():
    vehicle = _Vehicle()
    Rocket6Tvc().define(vehicle)
    store = vehicle.store
    for name in DEFINED:
        assert name in store.names(), name
        field = store.field(name)
        assert field.module == "tvc"
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        if name in INT_FIELDS:
            assert field.type == "int"
            assert store.get(name) == 0
        elif name in VEC_FIELDS:
            assert field.type == "vec"
            assert np.array_equal(store.get(name), np.zeros(3))
        else:
            assert field.type == "real"
            assert store.get(name) == 0.0
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_initialize_is_noop():
    vehicle = _Vehicle()
    tvc = Rocket6Tvc()
    tvc.define(vehicle)
    store = vehicle.store
    store.set("mtvc", 2)
    store.set("gtvc", GTVC)
    tvc.initialize(vehicle, _ctx())
    assert store.get("mtvc") == 2
    assert store.get("gtvc") == GTVC
    np.testing.assert_array_equal(store.get("FPB"), np.zeros(3))
    np.testing.assert_array_equal(store.get("FMPB"), np.zeros(3))
    assert store.get("etax") == 0.0
    assert store.get("etas") == 0.0
    assert store.get("detas") == 0.0


def test_mtvc0_no_raise_leaves_fpb_zeros():
    # Break: mtvc=0 treated as unknown, or FPB written from thrust.
    vehicle, tvc = _ready(mtvc=0, thrust=THRUST, delecx=DELECX)
    tvc.execute(vehicle, _ctx(DT))
    store = vehicle.store
    np.testing.assert_array_equal(store.get("FPB"), np.zeros(3))
    np.testing.assert_array_equal(store.get("FMPB"), np.zeros(3))
    assert store.get("etax") == 0.0
    assert store.get("etas") == 0.0


def test_mtvc2_delecx_step_limits_etax_and_fpb0_finite():
    # Break: missing second-order path, no FPB, or etax in rad vs deg limiter.
    vehicle, tvc = _ready(mtvc=2, delecx=DELECX, gtvc=GTVC, thrust=THRUST)
    tvc.execute(vehicle, _ctx(DT))
    store = vehicle.store
    assert abs(store.get("etax")) <= TVCLIMX
    assert np.isfinite(store.get("FPB")[0])
    assert store.get("FPB")[0] != 0.0


def test_mtvc1_raises():
    vehicle, tvc = _ready(mtvc=1, delecx=DELECX)
    with pytest.raises(ValueError):
        tvc.execute(vehicle, _ctx())


def test_mtvc3_raises():
    vehicle, tvc = _ready(mtvc=3, delecx=DELECX)
    with pytest.raises(ValueError):
        tvc.execute(vehicle, _ctx())


def test_other_mtvc_raises():
    for mtvc in (-1, 4):
        vehicle, tvc = _ready(mtvc=mtvc, delecx=DELECX)
        with pytest.raises(ValueError):
            tvc.execute(vehicle, _ctx())


def test_execute_matches_cadac_formulas():
    vehicle, tvc = _ready(mtvc=2, delecx=DELECX, delrcx=0.5, gtvc=GTVC, thrust=THRUST)
    want = _expected(vehicle.store, DT)
    tvc.execute(vehicle, _ctx(DT))
    _assert_step(vehicle.store, want)


def test_fpb_fmpb_from_gtvc_parm_xcg_thrust():
    vehicle, tvc = _ready(mtvc=2, delecx=DELECX, delrcx=0.5, gtvc=GTVC, thrust=THRUST, xcg=XCG)
    want = _expected(vehicle.store, DT)
    tvc.execute(vehicle, _ctx(DT))
    fpb = vehicle.store.get("FPB")
    fmpb = vehicle.store.get("FMPB")
    eta = vehicle.store.get("etas")
    zet = vehicle.store.get("zeta")
    assert _approx(fpb[0], cos(eta) * cos(zet) * THRUST)
    assert _approx(fpb[1], cos(eta) * sin(zet) * THRUST)
    assert _approx(fpb[2], -sin(eta) * THRUST)
    arm = PARM - XCG
    assert fmpb[0] == 0.0
    assert _approx(fmpb[1], arm * fpb[2])
    assert _approx(fmpb[2], -arm * fpb[1])
    assert _approx(vehicle.store.get("etacx"), GTVC * DELECX)
    np.testing.assert_allclose(fpb, want["FPB"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(fmpb, want["FMPB"], rtol=RTOL, atol=ATOL)


def test_command_converted_with_gtvc_and_rad():
    vehicle, tvc = _ready(mtvc=2, delecx=DELECX, delrcx=2.0, gtvc=0.5)
    tvc.execute(vehicle, _ctx(DT))
    assert _approx(vehicle.store.get("etacx"), 0.5 * DELECX)
    assert _approx(vehicle.store.get("zetcx"), 0.5 * 2.0)


def test_position_limit_zeros_rate_when_same_sign():
    vehicle, tvc = _ready(mtvc=2, delecx=DELECX, etas=0.3, detas=1.0)
    want = _expected(vehicle.store, DT)
    assert want["etas"] == TVCLIMX * RAD or abs(want["etas"]) <= TVCLIMX * RAD
    tvc.execute(vehicle, _ctx(DT))
    _assert_step(vehicle.store, want)
    assert abs(vehicle.store.get("etax")) <= TVCLIMX


def test_position_limit_keeps_rate_when_opposite_sign():
    vehicle, tvc = _ready(mtvc=2, delecx=DELECX, etas=0.3, detas=-1.0)
    want = _expected(vehicle.store, DT)
    tvc.execute(vehicle, _ctx(DT))
    _assert_step(vehicle.store, want)


def test_rate_limit_iflag_zeros_detasd_when_same_sign():
    vehicle, tvc = _ready(mtvc=2, delecx=100.0, detas=8.0)
    want = _expected(vehicle.store, DT)
    tvc.execute(vehicle, _ctx(DT))
    _assert_step(vehicle.store, want)


def test_rate_limit_keeps_detasd_when_opposite_sign():
    vehicle, tvc = _ready(mtvc=2, delecx=-100.0, detas=8.0)
    want = _expected(vehicle.store, DT)
    tvc.execute(vehicle, _ctx(DT))
    _assert_step(vehicle.store, want)


def test_cadac_sign_zero_is_plus_one_not_numpy_sign():
    # |etas=0| > tvclimx*RAD when tvclimx is negative, so the limiter fires.
    # CADAC sign(0)=+1 → clamp to tvclimx*RAD; np.sign(0)=0 would clamp to 0.
    vehicle, tvc = _ready(mtvc=2, delecx=DELECX, etas=0.0, tvclimx=-5.0, detas=1.0)
    want = _expected(vehicle.store, DT)
    cadac_clamped = -5.0 * RAD * _sign(0.0)
    numpy_clamped = -5.0 * RAD * float(np.sign(0.0))
    assert cadac_clamped != numpy_clamped
    assert _approx(want["etas"], integrate(1.0, 0.0, cadac_clamped, DT))
    assert want["etas"] != pytest.approx(integrate(1.0, 0.0, numpy_clamped, DT), rel=RTOL, abs=ATOL)
    tvc.execute(vehicle, _ctx(DT))
    _assert_step(vehicle.store, want)
    assert np.sign(0.0) == 0.0


def test_stored_slope_second_step():
    vehicle, tvc = _ready(mtvc=2, delecx=DELECX, delrcx=0.5)
    ctx = _ctx(DT)
    want1 = _expected(vehicle.store, DT)
    tvc.execute(vehicle, ctx)
    _assert_step(vehicle.store, want1)
    want2 = _expected(vehicle.store, DT)
    tvc.execute(vehicle, ctx)
    _assert_step(vehicle.store, want2)
    assert abs(vehicle.store.get("etax")) <= TVCLIMX
    assert vehicle.store.get("etas") != want1["etas"]
    assert np.isfinite(vehicle.store.get("FPB")[0])


def test_mtvc0_matches_expected_unchanged_outputs():
    vehicle, tvc = _ready(mtvc=0, thrust=THRUST, delecx=DELECX)
    want = _expected(vehicle.store, DT)
    tvc.execute(vehicle, _ctx(DT))
    _assert_step(vehicle.store, want)


def test_terminate_exists_and_is_pass():
    vehicle, tvc = _ready(mtvc=0, delecx=DELECX)
    assert tvc.terminate(vehicle, _ctx()) is None
