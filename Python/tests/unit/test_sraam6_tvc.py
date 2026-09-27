from math import cos, sin
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.sraam6.tvc import Sraam6Tvc

RTOL = 1e-12
ATOL = 1e-14

DT = 0.001
TVCLIMX = 28.0
DTVCLIMX = 600.0
WNTVC = 100.0
ZETTVC = 0.7

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
INT_FIELDS = ("mtvc",)
DATA_REALS = (
    "tvclimx",
    "dtvclimx",
    "wntvc",
    "zettvc",
    "factgtvc",
    "gtvc",
    "parm",
)
OUT_VEC = ("FPB", "FMPB")
DIA_PLOT = ("etax", "zetx")
DIA = ("etacx", "zetcx")
STATE = (
    "etasd",
    "zetad",
    "etas",
    "zeta",
    "detasd",
    "dzetad",
    "detas",
    "dzeta",
)
PLANE6 = ("delecx", "delrcx")


def _ctx(dt=DT):
    return SimContext(
        sim_time=0.0,
        int_step=dt,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _plant_inputs(store, *, thrust=100.0, xcg=0.0, dqcx=0.0, drcx=0.0, pdynmc=0.0):
    store.define(Field("thrust", thrust, "real", "out", "propulsion"))
    store.define(Field("xcg", xcg, "real", "diag", "propulsion"))
    store.define(Field("dqcx", dqcx, "real", "out", "control"))
    store.define(Field("drcx", drcx, "real", "out", "control"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))


def _ready(
    *,
    mtvc=0,
    gtvc=0.0,
    factgtvc=0.0,
    parm=0.0,
    tvclimx=TVCLIMX,
    dtvclimx=DTVCLIMX,
    wntvc=WNTVC,
    zettvc=ZETTVC,
    thrust=100.0,
    xcg=0.0,
    dqcx=0.0,
    drcx=0.0,
    pdynmc=0.0,
    plant=True,
):
    vehicle = SimpleNamespace(store=StateStore())
    tvc = Sraam6Tvc()
    tvc.define(vehicle)
    if plant:
        _plant_inputs(
            vehicle.store,
            thrust=thrust,
            xcg=xcg,
            dqcx=dqcx,
            drcx=drcx,
            pdynmc=pdynmc,
        )
    store = vehicle.store
    store.set("mtvc", mtvc)
    store.set("gtvc", gtvc)
    store.set("factgtvc", factgtvc)
    store.set("parm", parm)
    store.set("tvclimx", tvclimx)
    store.set("dtvclimx", dtvclimx)
    store.set("wntvc", wntvc)
    store.set("zettvc", zettvc)
    tvc.initialize(vehicle, _ctx())
    return vehicle, tvc


def _fpb(eta, zet, thrust):
    return np.array(
        [
            cos(eta) * cos(zet) * thrust,
            cos(eta) * sin(zet) * thrust,
            -sin(eta) * thrust,
        ],
        dtype=float,
    )


def test_name_is_tvc():
    assert Sraam6Tvc.name == "tvc"
    assert Sraam6Tvc().name == "tvc"


def test_define_registers_cpp_fields():
    vehicle = SimpleNamespace(store=StateStore())
    Sraam6Tvc().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    for name in DEFINED:
        assert store.field(name).module == "tvc"
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
    for name in OUT_VEC:
        np.testing.assert_allclose(store.get(name), np.zeros(3), rtol=RTOL, atol=ATOL)
        assert store.field(name).type == "vec"
        assert store.field(name).role == "out"
        assert store.field(name).outputs == ()
    for name in DIA_PLOT:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "diag"
        assert store.field(name).outputs == ("plot",)
    for name in DIA:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "diag"
        assert store.field(name).outputs == ()
    for name in STATE:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "state"
        assert store.field(name).outputs == ()
    for name in PLANE6 + ("time",):
        assert name not in store.names()


def test_mtvc0_leaves_fpb_at_define_zeros():
    vehicle, tvc = _ready(mtvc=0, plant=False)
    tvc.execute(vehicle, _ctx())
    np.testing.assert_allclose(vehicle.store.get("FPB"), np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("FMPB"), np.zeros(3), rtol=RTOL, atol=ATOL)
    assert vehicle.store.get("etax") == 0.0
    assert vehicle.store.get("zetx") == 0.0


def test_mtvc1_fpb_matches_sin_cos_formulas():
    vehicle, tvc = _ready(mtvc=1, gtvc=1.0, dqcx=1.0, thrust=100.0, parm=1.5, xcg=0.5)
    tvc.execute(vehicle, _ctx())
    store = vehicle.store
    eta = 1.0 * 1.0 * RAD
    zet = 0.0
    want = _fpb(eta, zet, 100.0)
    fpb = store.get("FPB")
    assert np.all(np.isfinite(fpb))
    np.testing.assert_allclose(fpb, want, rtol=RTOL, atol=ATOL)
    arm = 1.5 - 0.5
    np.testing.assert_allclose(
        store.get("FMPB"),
        np.array([0.0, arm * want[2], -arm * want[1]]),
        rtol=RTOL,
        atol=ATOL,
    )
    assert store.get("etax") == pytest.approx(eta * DEG, rel=RTOL, abs=ATOL)
    assert store.get("zetx") == pytest.approx(zet * DEG, rel=RTOL, abs=ATOL)
    assert store.get("etacx") == pytest.approx(store.get("etax"), rel=RTOL, abs=ATOL)
    assert store.get("zetcx") == pytest.approx(store.get("zetx"), rel=RTOL, abs=ATOL)


def test_mtvc4_raises():
    vehicle, tvc = _ready(mtvc=4, plant=False)
    with pytest.raises(ValueError):
        tvc.execute(vehicle, _ctx())


def test_mtvc_negative_raises():
    vehicle, tvc = _ready(mtvc=-1, plant=False)
    with pytest.raises(ValueError):
        tvc.execute(vehicle, _ctx())


def test_mtvc2_first_step_lags_and_writes_etacx_from_etax():
    vehicle, tvc = _ready(mtvc=2, gtvc=1.0, dqcx=1.0, thrust=100.0)
    tvc.execute(vehicle, _ctx(DT))
    store = vehicle.store
    etac = 1.0 * 1.0 * RAD
    detasd_new = WNTVC * WNTVC * etac
    detas = (detasd_new + 0.0) * DT / 2.0
    np.testing.assert_allclose(store.get("FPB"), _fpb(0.0, 0.0, 100.0), rtol=RTOL, atol=ATOL)
    assert store.get("etas") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert store.get("detasd") == pytest.approx(detasd_new, rel=RTOL, abs=ATOL)
    assert store.get("detas") == pytest.approx(detas, rel=RTOL, abs=ATOL)
    assert store.get("etax") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert store.get("etacx") == pytest.approx(store.get("etax"), rel=RTOL, abs=ATOL)
    assert store.get("etacx") != pytest.approx(etac * DEG, rel=1e-6)
    assert store.get("gtvc") == 1.0


def test_mtvc3_inline_gtvc_from_pdynmc_not_store():
    vehicle, tvc = _ready(
        mtvc=3,
        gtvc=1.0,
        factgtvc=0.0,
        pdynmc=0.0,
        dqcx=1.0,
        thrust=100.0,
    )
    tvc.execute(vehicle, _ctx(DT))
    store = vehicle.store
    gtvc_inline = (-5.0e-6 * 0.0 + 0.5) * (0.0 + 1)
    etac = gtvc_inline * 1.0 * RAD
    detasd_new = WNTVC * WNTVC * etac
    assert store.get("gtvc") == 1.0
    assert store.get("detasd") == pytest.approx(detasd_new, rel=RTOL, abs=ATOL)
    assert detasd_new != pytest.approx(WNTVC * WNTVC * 1.0 * RAD, rel=1e-9)


def test_mtvc3_high_q_zeros_gtvc():
    vehicle, tvc = _ready(
        mtvc=3,
        gtvc=1.0,
        factgtvc=0.0,
        pdynmc=2.0e5,
        dqcx=1.0,
        thrust=100.0,
    )
    tvc.execute(vehicle, _ctx(DT))
    assert vehicle.store.get("gtvc") == 1.0
    assert vehicle.store.get("detasd") == pytest.approx(0.0, rel=RTOL, abs=ATOL)


def test_cadac_sign_zero_is_plus_one_not_numpy_sign():
    vehicle, tvc = _ready(
        mtvc=2,
        gtvc=1.0,
        dqcx=1.0,
        thrust=100.0,
        tvclimx=-5.0,
    )
    tvc.execute(vehicle, _ctx())
    store = vehicle.store
    etas = -5.0 * RAD
    assert store.get("etas") == pytest.approx(etas, rel=RTOL, abs=ATOL)
    assert np.sign(0.0) == 0.0


def test_uses_dqcx_not_plane6_delecx():
    vehicle, tvc = _ready(mtvc=1, gtvc=1.0, dqcx=1.0, drcx=0.0, thrust=100.0)
    vehicle.store.define(Field("delecx", 40.0, "real", "out", "control"))
    vehicle.store.define(Field("delrcx", 40.0, "real", "out", "control"))
    tvc.execute(vehicle, _ctx())
    eta = 1.0 * RAD
    np.testing.assert_allclose(vehicle.store.get("FPB"), _fpb(eta, 0.0, 100.0), rtol=RTOL, atol=ATOL)


def test_does_not_require_time():
    vehicle, tvc = _ready(mtvc=1, gtvc=1.0, dqcx=1.0, thrust=100.0)
    assert "time" not in vehicle.store.names()
    tvc.execute(vehicle, _ctx())
    assert "time" not in vehicle.store.names()


def test_initialize_and_terminate_are_pass():
    vehicle, tvc = _ready(mtvc=0, plant=False)
    assert tvc.initialize(vehicle, _ctx()) is None
    assert tvc.terminate(vehicle, _ctx()) is None
    np.testing.assert_allclose(vehicle.store.get("FPB"), np.zeros(3), rtol=RTOL, atol=ATOL)
