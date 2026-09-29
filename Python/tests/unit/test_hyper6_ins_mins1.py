"""HYPER6 mins=1 instrumented INS (GPS/star updates optional when modules exist)."""

from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import RAD, WEII3
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat3tr
from cadac.math.wgs84 import GM, cad_in_geo84, cad_tdi84
from cadac.stoch import gauss, seed
from cadac.vehicles.round6.hyper6.ins import Hyper6Ins

RTOL = 1e-12
ATOL = 1e-14
ZEROS3 = np.zeros(3)

# C++ Hyper::def_ins gauss sigmas (g++ Variable::init args right-to-left).
_INS_DEFINE_SIGMAS = (
    (1.1e-4, 1.1e-4, 1.1e-4),
    (2.0e-5, 2.0e-5, 2.0e-5),
    (1.0e-6, 1.0e-6, 1.0e-6),
    (1.1e-4, 1.1e-4, 1.1e-4),
    (5.0e-4, 5.0e-4, 5.0e-4),
    (3.56e-3, 3.56e-3, 3.56e-3),
)


def _ctx(int_step=0.001):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _gauss3_rtl(sigs):
    third = gauss(0.0, sigs[2])
    second = gauss(0.0, sigs[1])
    first = gauss(0.0, sigs[0])
    return np.array([first, second, third], dtype=float)


def _plant_truth(store):
    time = 0.0
    lonx = 10.0
    latx = 10.0
    alt = 10000.0
    sbii = cad_in_geo84(lonx * RAD, latx * RAD, alt, time)
    tdi = cad_tdi84(lonx * RAD, latx * RAD, alt, time)
    tbd = mat3tr(0.0, 2.5 * RAD, 0.0)
    tbi = tbd @ tdi
    vbed = np.array([1000.0, 0.0, 0.0], dtype=float)
    veic = np.array([-WEII3 * sbii[1], WEII3 * sbii[0], 0.0], dtype=float)
    vbii = tdi.T @ vbed + veic
    wbib = np.array([0.01, 0.02, 0.03], dtype=float)
    wbii = tbi.T @ wbib
    fspb = np.array([1.0, -0.2, 9.5], dtype=float)
    dbi = float(np.linalg.norm(sbii))
    for name, value, ftype, role, module in (
        ("time", time, "real", "exec", "kinematics"),
        ("TBI", tbi, "mat", "state", "kinematics"),
        ("WBIB", wbib, "vec", "state", "euler"),
        ("WBII", wbii, "vec", "out", "euler"),
        ("SBII", sbii, "vec", "state", "newton"),
        ("VBII", vbii, "vec", "state", "newton"),
        ("FSPB", fspb, "vec", "out", "newton"),
        ("dbi", dbi, "real", "out", "newton"),
    ):
        if name not in store.names():
            store.define(Field(name, value, ftype, role, module))
        store.set(name, value)
    return {
        "time": time,
        "TBI": tbi,
        "WBIB": wbib,
        "WBII": wbii,
        "SBII": sbii,
        "VBII": vbii,
        "FSPB": fspb,
        "dbi": dbi,
    }


def _defined(mins=1):
    vehicle = SimpleNamespace(store=StateStore())
    ins = Hyper6Ins()
    ins.define(vehicle)
    vehicle.store.set("mins", mins)
    return vehicle, ins


def _ready(mins=1):
    vehicle, ins = _defined(mins=mins)
    truth = _plant_truth(vehicle.store)
    ins.initialize(vehicle, _ctx())
    if mins == 1:
        for name in (
            "ESBI",
            "EVBI",
            "RICI",
            "EMISG",
            "ESCALG",
            "EBIASG",
            "EMISA",
            "ESCALA",
            "EBIASA",
        ):
            vehicle.store.set(name, ZEROS3)
    return vehicle, ins, truth


def test_hyper6_mins_1_initialize_and_execute():
    seed(1234)
    vehicle, ins = _defined(mins=1)
    truth = _plant_truth(vehicle.store)
    vehicle.store.set("frax", 0.0)
    ctx = _ctx()
    ins.initialize(vehicle, ctx)
    store = vehicle.store
    assert store.get("mins") == 1
    assert not np.allclose(store.get("ESBI"), ZEROS3)
    assert not np.allclose(store.get("EVBI"), ZEROS3)
    assert not np.allclose(store.get("RICI"), ZEROS3)
    for name in ("EMISG", "ESCALG", "EBIASG", "EMISA", "ESCALA", "EBIASA"):
        assert not np.allclose(store.get(name), ZEROS3)

    # Replay execute with zero instruments so outputs stay deterministic.
    for name in (
        "ESBI",
        "EVBI",
        "RICI",
        "EMISG",
        "ESCALG",
        "EBIASG",
        "EMISA",
        "ESCALA",
        "EBIASA",
    ):
        store.set(name, ZEROS3)
    ins.execute(vehicle, ctx)
    np.testing.assert_allclose(store.get("SBIIC"), truth["SBII"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VBIIC"), truth["VBII"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("TBIC"), truth["TBI"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FSPCB"), truth["FSPB"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("WBICB"), truth["WBIB"], rtol=RTOL, atol=ATOL)
    assert store.get("dvbec") > 0.0


def test_initialize_mins_one_matches_hyper6_gauss_then_cholesky():
    seed(1234)
    vehicle, ins = _defined(mins=1)
    _plant_truth(vehicle.store)
    vehicle.store.set("frax", 0.0)
    ins.initialize(vehicle, _ctx())

    seed(1234)
    emisg = _gauss3_rtl(_INS_DEFINE_SIGMAS[0])
    escalg = _gauss3_rtl(_INS_DEFINE_SIGMAS[1])
    ebiasg = _gauss3_rtl(_INS_DEFINE_SIGMAS[2])
    emisa = _gauss3_rtl(_INS_DEFINE_SIGMAS[3])
    escala = _gauss3_rtl(_INS_DEFINE_SIGMAS[4])
    ebiasa = _gauss3_rtl(_INS_DEFINE_SIGMAS[5])
    draws = np.array([gauss(0.0, 1.0) for _ in range(9)], dtype=float)
    from cadac.vehicles.round6.hyper6.ins import _PP0, _cholesky

    xx = _cholesky(_PP0) @ draws
    store = vehicle.store
    np.testing.assert_allclose(store.get("EMISG"), emisg, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ESCALG"), escalg, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("EBIASG"), ebiasg, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("EMISA"), emisa, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ESCALA"), escala, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("EBIASA"), ebiasa, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ESBI"), xx[0:3], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("EVBI"), xx[3:6], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("RICI"), xx[6:9] * 0.001, rtol=RTOL, atol=ATOL)


def test_initialize_mins_one_frax_scales_error_states():
    seed(99)
    vehicle, ins = _defined(mins=1)
    _plant_truth(vehicle.store)
    vehicle.store.set("frax", 10.0)
    ins.initialize(vehicle, _ctx())

    seed(99)
    for sigs in _INS_DEFINE_SIGMAS:
        _gauss3_rtl(sigs)
    draws = np.array([gauss(0.0, 1.0) for _ in range(9)], dtype=float)
    from cadac.vehicles.round6.hyper6.ins import _PP0, _cholesky

    xx = _cholesky(_PP0) @ draws * 11.0
    store = vehicle.store
    np.testing.assert_allclose(store.get("ESBI"), xx[0:3], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("EVBI"), xx[3:6], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("RICI"), xx[6:9] * 0.001, rtol=RTOL, atol=ATOL
    )


def test_execute_mins_one_does_not_raise():
    vehicle, ins, _truth = _ready(mins=1)
    ins.execute(vehicle, _ctx())


def test_execute_mins_one_zero_instruments_sbiic_equals_sbii():
    vehicle, ins, truth = _ready(mins=1)
    ins.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        vehicle.store.get("SBIIC"), truth["SBII"], rtol=RTOL, atol=ATOL
    )


def test_ins_grav_matches_cpp_gm_formula():
    vehicle, ins = _defined(mins=1)
    esbi = np.array([10.0, -3.0, 5.0], dtype=float)
    sbiic = np.array([6.4e6, 1.0e5, 2.0e5], dtype=float)
    dbi = 6.378e6
    vehicle.store.define(Field("dbi", dbi, "real", "out", "newton"))
    dbic = float(np.linalg.norm(sbiic))
    ed = dbic - dbi
    dum = GM / dbic**3
    want = esbi * (-dum) - sbiic * (3.0 * ed * dum / dbic)
    got = ins.ins_grav(vehicle, esbi, sbiic)
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)


def test_execute_mins_one_skips_gps_without_modules():
    vehicle, ins, truth = _ready(mins=1)
    store = vehicle.store
    for name in ("mgps", "mstar", "SXH", "VXH", "URIC"):
        assert name not in store.names()
    ins.execute(vehicle, _ctx())
    np.testing.assert_allclose(store.get("SBIIC"), truth["SBII"], rtol=RTOL, atol=ATOL)


def test_execute_mins_one_gps_update_when_present():
    vehicle, ins, truth = _ready(mins=1)
    store = vehicle.store
    sxh = np.array([1.0, -2.0, 3.0], dtype=float)
    vxh = np.array([0.4, 0.5, -0.6], dtype=float)
    store.define(Field("mgps", 3, "int", "data", "gps"))
    store.define(Field("SXH", sxh, "vec", "out", "gps"))
    store.define(Field("VXH", vxh, "vec", "out", "gps"))
    ins.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        store.get("SBIIC"), truth["SBII"] - sxh, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        store.get("VBIIC"), truth["VBII"] - vxh, rtol=RTOL, atol=ATOL
    )
    assert store.get("mgps") == 2


def test_execute_mins_one_star_update_when_present():
    vehicle, ins, truth = _ready(mins=1)
    store = vehicle.store
    uric = np.array([0.001, -0.002, 0.003], dtype=float)
    store.define(Field("mstar", 3, "int", "data", "startrack"))
    store.define(Field("URIC", uric, "vec", "out", "startrack"))
    # Nonzero WBIB drives tilt integration; zero instruments => WBICB==WBIB.
    store.set("EWALKG", ZEROS3)
    store.set("EUNBG", ZEROS3)
    ins.execute(vehicle, _ctx())
    assert store.get("mstar") == 2
    # With zero gyro errors and zero prior RICI, star subtracts URIC from RICI.
    np.testing.assert_allclose(store.get("RICI"), -uric, rtol=RTOL, atol=ATOL)


def test_execute_mins_two_still_raises():
    vehicle, ins, truth = _ready(mins=0)
    vehicle.store.set("mins", 2)
    sentinel = np.array([9.0, 8.0, 7.0])
    vehicle.store.set("SBIIC", sentinel)
    with pytest.raises(ValueError, match="unknown mins"):
        ins.execute(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("SBIIC"), sentinel)
    np.testing.assert_array_equal(vehicle.store.get("SBII"), truth["SBII"])
