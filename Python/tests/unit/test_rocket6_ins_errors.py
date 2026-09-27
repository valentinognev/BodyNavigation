from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import RAD, WEII3
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat3tr
from cadac.math.wgs84 import GM, cad_in_geo84, cad_tdi84
from cadac.stoch import seed
from cadac.vehicles.round6.rocket6.ins import Rocket6Ins

RTOL = 1e-12
ATOL = 1e-14
ZEROS3 = np.zeros(3)


def _ctx(int_step=0.001):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


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


def _plant_truth(store):
    time = 0.0
    lonx = -120.49
    latx = 34.68
    alt = 100.0
    sbii = cad_in_geo84(lonx * RAD, latx * RAD, alt, time)
    tdi = cad_tdi84(lonx * RAD, latx * RAD, alt, time)
    tbd = mat3tr(0.0, 90.0 * RAD, 0.0)
    tbi = tbd @ tdi
    vbed = np.array([10.0, 0.0, 0.0], dtype=float)
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
    ins = Rocket6Ins()
    ins.define(vehicle)
    vehicle.store.set("mins", mins)
    return vehicle, ins


def _ready(mins=1):
    vehicle, ins = _defined(mins=mins)
    truth = _plant_truth(vehicle.store)
    ins.initialize(vehicle, _ctx())
    if mins == 1:
        # Execute-path tests plant their own error states; zero C++ init_ins draws.
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


def test_initialize_mins_one_writes_cholesky_error_states():
    seed(1234)
    vehicle, ins = _defined(mins=1)
    ins.initialize(vehicle, _ctx())
    store = vehicle.store
    np.testing.assert_allclose(
        store.get("ESBI"),
        np.array([1.00288005, -13.18049982, -9.24066121]),
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        store.get("EVBI"),
        np.array([0.03403813, -0.10710086, -0.02137323]),
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        store.get("RICI"),
        np.array([4.34377825e-05, 3.48318603e-05, 1.10402655e-04]),
        rtol=1e-8,
        atol=1e-12,
    )


def test_initialize_mins_zero_stays_noop():
    vehicle, ins = _defined(mins=0)
    truth = _plant_truth(vehicle.store)
    sentinel = np.array([1.0, 2.0, 3.0])
    vehicle.store.set("ESBI", sentinel)
    ins.initialize(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("ESBI"), sentinel)
    np.testing.assert_array_equal(vehicle.store.get("SBIIC"), ZEROS3)
    np.testing.assert_array_equal(vehicle.store.get("SBII"), truth["SBII"])


def test_initialize_mins_two_still_raises():
    vehicle, ins = _defined(mins=2)
    with pytest.raises(ValueError, match="unknown mins"):
        ins.initialize(vehicle, _ctx())


def test_execute_mins_one_does_not_raise():
    vehicle, ins, _truth = _ready(mins=1)
    ins.execute(vehicle, _ctx())


def test_execute_mins_one_zero_instruments_sbiic_equals_sbii():
    vehicle, ins, truth = _ready(mins=1)
    ins.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        vehicle.store.get("SBIIC"), truth["SBII"], rtol=RTOL, atol=ATOL
    )


def test_execute_mins_one_zero_instruments_wbicb_equals_wbib():
    vehicle, ins, truth = _ready(mins=1)
    ins.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        vehicle.store.get("WBICB"), truth["WBIB"], rtol=RTOL, atol=ATOL
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


def test_ins_gyro_zero_errors_wbicb_equals_wbib():
    vehicle, ins, truth = _ready(mins=1)
    ewbib, wbicb = ins.ins_gyro(vehicle, 0.001)
    np.testing.assert_allclose(ewbib, ZEROS3, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(wbicb, truth["WBIB"], rtol=RTOL, atol=ATOL)


def test_execute_mins_two_still_raises():
    vehicle, ins, truth = _ready(mins=0)
    vehicle.store.set("mins", 2)
    sentinel = np.array([9.0, 8.0, 7.0])
    vehicle.store.set("SBIIC", sentinel)
    with pytest.raises(ValueError, match="unknown mins"):
        ins.execute(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("SBIIC"), sentinel)
    np.testing.assert_array_equal(vehicle.store.get("SBII"), truth["SBII"])


def test_execute_mins_one_skips_gps_without_sxh_vxh():
    vehicle, ins, truth = _ready(mins=1)
    store = vehicle.store
    store.define(Field("mgps", 3, "int", "data", "gps"))
    ins.execute(vehicle, _ctx())
    assert store.get("mgps") == 3
    np.testing.assert_allclose(store.get("SBIIC"), truth["SBII"], rtol=RTOL, atol=ATOL)


def test_execute_mins_one_gps_update_then_recompute():
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
    np.testing.assert_allclose(store.get("ESBI"), -sxh, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("EVBI"), -vxh, rtol=RTOL, atol=ATOL)
    assert store.get("mgps") == 2


def test_execute_mins_one_skips_star_without_uric():
    vehicle, ins, truth = _ready(mins=1)
    store = vehicle.store
    store.define(Field("mstar", 3, "int", "data", "startrack"))
    ins.execute(vehicle, _ctx())
    assert store.get("mstar") == 3
    np.testing.assert_allclose(store.get("TBIC"), truth["TBI"], rtol=RTOL, atol=ATOL)


def test_execute_mins_one_star_update_applies_uric():
    vehicle, ins, truth = _ready(mins=1)
    store = vehicle.store
    uric = np.array([0.001, -0.002, 0.003], dtype=float)
    store.define(Field("mstar", 3, "int", "data", "startrack"))
    store.define(Field("URIC", uric, "vec", "out", "startrack"))
    ins.execute(vehicle, _ctx())
    rici = -uric
    tiic = np.eye(3) - _skew(rici)
    want_tbic = truth["TBI"] @ tiic
    np.testing.assert_allclose(store.get("RICI"), rici, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("TBIC"), want_tbic, rtol=RTOL, atol=ATOL)
    assert store.get("mstar") == 2
