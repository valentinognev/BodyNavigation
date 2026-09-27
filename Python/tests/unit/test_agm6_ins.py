import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import DEG, PI, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat3tr
from cadac.stoch import gauss, seed
from cadac.vehicles.flat6.agm6.ins import Agm6Ins, PP0, _cholesky

RTOL = 1e-12
ATOL = 1e-14
DT = 0.001
PLOT = ("plot",)
ZEROS3 = (0.0, 0.0, 0.0)
ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

# C++ Missile::def_ins order (empty last-arg → no plot/scrn).
FIELDS = {
    "mins": ("int", "data", 0, ()),
    "frax": ("real", "data", 0.0, ()),
    "hbem": ("real", "out", 0.0, ()),
    "VBELC": ("vec", "out", ZEROS3, ()),
    "SBELC": ("vec", "out", ZEROS3, ()),
    "WBECB": ("vec", "out", ZEROS3, ()),
    "EWALKG": ("vec", "data", ZEROS3, ()),
    "EUNBG": ("vec", "data", ZEROS3, ()),
    "EMISG": ("vec", "data", ZEROS3, ()),
    "ESCALG": ("vec", "data", ZEROS3, ()),
    "EBIASG": ("vec", "data", ZEROS3, ()),
    "biasal": ("real", "data", 0.0, ()),
    "randal": ("real", "data", 0.0, ()),
    "ehbe": ("real", "out", 0.0, ()),
    "TBLC": ("mat", "out", ZEROS33, ()),
    "EWALKA": ("vec", "data", ZEROS3, ()),
    "EMISA": ("vec", "data", ZEROS3, ()),
    "ESCALA": ("vec", "data", ZEROS3, ()),
    "EBIASA": ("vec", "data", ZEROS3, ()),
    "EUG": ("vec", "diag", ZEROS3, ()),
    "EWG": ("vec", "diag", ZEROS3, ()),
    "EWBEB": ("vec", "diag", ZEROS3, ()),
    "EFSPB": ("vec", "diag", ZEROS3, ()),
    "tanlat": ("real", "init", 0.0, ()),
    "thtblc": ("real", "out", 0.0, ()),
    "thtblcx": ("real", "out", 0.0, ()),
    "dvbec": ("real", "out", 0.0, ()),
    "thtvlc": ("real", "out", 0.0, ()),
    "thtvlcx": ("real", "out", 0.0, ()),
    "psivlcx": ("real", "out", 0.0, ()),
    "FSPCB": ("vec", "out", ZEROS3, ()),
    "phiblcx": ("real", "out", 0.0, ()),
    "RECED": ("vec", "state", ZEROS3, ()),
    "RECE": ("vec", "state", ZEROS3, ()),
    "EVBED": ("vec", "state", ZEROS3, ()),
    "EVBE": ("vec", "state", ZEROS3, ()),
    "ESTTCD": ("vec", "state", ZEROS3, ()),
    "ESTTC": ("vec", "state", ZEROS3, ()),
}
DEFINED = tuple(FIELDS)
GAUSS_IN_CPP = ("EMISG", "ESCALG", "EBIASG", "EMISA", "ESCALA", "EBIASA")
CONTROL_INS = ("WBECB", "FSPCB", "phiblcx")
ERROR_STATES = ("RECED", "RECE", "EVBED", "EVBE", "ESTTCD", "ESTTC")
TRUTH = (
    "TBL",
    "FSPB",
    "WBEB",
    "SBEL",
    "VBEL",
    "dvbe",
    "phiblx",
)
EXTERNALS = TRUTH + ("hbe", "TLB", "time")

PSIBLx = 10.0
THTBLx = 3.0
PHIBLx = 2.0
DVBE = 293.0
ALPHA0X = 3.0
HBE = 7000.0
SBEL = np.array([100.0, -50.0, -HBE], dtype=float)
FSPB = np.array([1.0, 0.2, -9.5], dtype=float)
WBEB = np.array([0.1, 0.05, -0.03], dtype=float)
BIASAL = 10.0
RANDAL = 2.0


def _cadac_sign(variable):
    if variable < 0.0:
        return -1
    return 1


def _ctx(int_step=DT):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _tbl():
    return mat3tr(PSIBLx * RAD, THTBLx * RAD, PHIBLx * RAD)


def _vbel(tbl):
    alpha = ALPHA0X * RAD
    vbeb = np.array(
        [math.cos(alpha) * DVBE, 0.0, math.sin(alpha) * DVBE],
        dtype=float,
    )
    return tbl.T @ vbeb


def _cpp_common(vbelc, tblc):
    vbelc1 = float(vbelc[0])
    vbelc2 = float(vbelc[1])
    vbelc3 = float(vbelc[2])
    if vbelc1 == 0.0 and vbelc2 == 0.0:
        psivlc = 0.0
        thtvlc = 0.0
    else:
        psivlc = math.atan2(vbelc2, vbelc1)
        thtvlc = math.atan2(
            -vbelc3, math.sqrt(vbelc1 * vbelc1 + vbelc2 * vbelc2)
        )
    tblc13 = float(tblc[0, 2])
    if math.fabs(tblc13) < 1.0:
        thtblc = math.asin(-tblc13)
    else:
        thtblc = PI / 2.0 * _cadac_sign(-tblc13)
    phiblcx = math.atan2(float(tblc[1, 2]), float(tblc[2, 2])) * DEG
    return {
        "psivlcx": psivlc * DEG,
        "thtvlc": thtvlc,
        "thtvlcx": thtvlc * DEG,
        "thtblc": thtblc,
        "thtblcx": thtblc * DEG,
        "phiblcx": phiblcx,
    }


def _plant_sbel_vbel(store, sbel=SBEL, vbel=None):
    tbl = _tbl()
    if vbel is None:
        vbel = _vbel(tbl)
    store.define(Field("SBEL", sbel, "vec", "state", "newton"))
    store.define(Field("VBEL", vbel, "vec", "out", "newton"))
    return {"SBEL": np.asarray(sbel, dtype=float), "VBEL": np.asarray(vbel, dtype=float)}


def _plant_truth(store, *, tlb=False):
    tbl = _tbl()
    vbel = _vbel(tbl)
    planted = {
        "TBL": tbl,
        "FSPB": np.asarray(FSPB, dtype=float),
        "WBEB": np.asarray(WBEB, dtype=float),
        "SBEL": np.asarray(SBEL, dtype=float),
        "VBEL": vbel,
        "dvbe": float(np.linalg.norm(vbel)),
        "phiblx": PHIBLx,
        "hbe": HBE,
    }
    specs = (
        ("TBL", planted["TBL"], "mat", "state", "kinematics"),
        ("FSPB", planted["FSPB"], "vec", "out", "newton"),
        ("WBEB", planted["WBEB"], "vec", "diag", "euler"),
        ("SBEL", planted["SBEL"], "vec", "state", "newton"),
        ("VBEL", planted["VBEL"], "vec", "out", "newton"),
        ("dvbe", planted["dvbe"], "real", "out", "newton"),
        ("phiblx", planted["phiblx"], "real", "out", "kinematics"),
        ("hbe", planted["hbe"], "real", "out", "newton"),
    )
    for name, value, ftype, role, module in specs:
        store.define(Field(name, value, ftype, role, module))
    if tlb:
        planted["TLB"] = tbl.T.copy()
        store.define(Field("TLB", planted["TLB"], "mat", "state", "kinematics"))
    return planted


def _defined(mins=0):
    vehicle = SimpleNamespace(store=StateStore())
    ins = Agm6Ins()
    ins.define(vehicle)
    vehicle.store.set("mins", mins)
    return vehicle, ins


def _zero_live_ins_errors(store, truth=None):
    for name in GAUSS_IN_CPP:
        store.set(name, (0.0, 0.0, 0.0))
    for name in ERROR_STATES:
        store.set(name, (0.0, 0.0, 0.0))
    if truth is not None:
        store.set("SBELC", np.asarray(truth["SBEL"], dtype=float).copy())
        store.set("VBELC", np.asarray(truth["VBEL"], dtype=float).copy())


def _ready(mins=0, *, tlb=False):
    vehicle, ins = _defined(mins=mins)
    truth = _plant_truth(vehicle.store, tlb=tlb)
    ins.initialize(vehicle, _ctx())
    _zero_live_ins_errors(vehicle.store, truth)
    return vehicle, ins, truth


def test_name_is_ins():
    assert Agm6Ins.name == "ins"
    assert Agm6Ins().name == "ins"


def test_define_registers_cpp_def_ins_fields():
    vehicle = SimpleNamespace(store=StateStore())
    Agm6Ins().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    zeros3 = np.zeros(3)
    zeros33 = np.zeros((3, 3))
    for name, (ftype, role, default, outputs) in FIELDS.items():
        field = store.field(name)
        assert field.module == "ins"
        assert field.type == ftype
        assert field.role == role
        assert field.outputs == outputs
        if ftype == "int":
            assert store.get(name) == default
            assert type(store.get(name)) is int
        elif ftype == "real":
            assert store.get(name) == default
        elif ftype == "vec":
            if name not in GAUSS_IN_CPP:
                np.testing.assert_array_equal(store.get(name), zeros3)
            assert store.get(name).shape == (3,)
        else:
            np.testing.assert_array_equal(store.get(name), zeros33)
            assert store.get(name).shape == (3, 3)
    for name in EXTERNALS:
        assert name not in store.names()


def _gauss3_rtl(sigs):
    # g++ evaluates Variable::init(v1,v2,v3) arguments right-to-left.
    third = gauss(0.0, sigs[2])
    second = gauss(0.0, sigs[1])
    first = gauss(0.0, sigs[0])
    return (first, second, third)


def _cpp_def_ins_gauss_vectors():
    return {
        "EMISG": _gauss3_rtl((1.1e-4, 1.1e-4, 1.1e-4)),
        "ESCALG": _gauss3_rtl((2e-5, 2.5e-5, 2.5e-5)),
        "EBIASG": _gauss3_rtl((1e-5, 3.2e-6, 3.2e-6)),
        "EMISA": _gauss3_rtl((1.1e-4, 1.1e-4, 1.1e-4)),
        "ESCALA": _gauss3_rtl((5e-4, 5e-4, 5e-4)),
        "EBIASA": _gauss3_rtl((3.56e-3, 3.56e-3, 3.56e-3)),
    }


def test_define_error_data_vectors_match_cpp_gauss_order():
    seed(12345)
    vehicle = SimpleNamespace(store=StateStore())
    Agm6Ins().define(vehicle)
    seed(12345)
    want = _cpp_def_ins_gauss_vectors()
    store = vehicle.store
    for name in GAUSS_IN_CPP:
        np.testing.assert_allclose(store.get(name), want[name], rtol=RTOL, atol=ATOL)
        assert store.field(name).role == "data"


def test_initialize_mins_one_applies_cholesky_gauss_draws():
    seed(12345)
    vehicle, ins = _defined(mins=1)
    planted = _plant_sbel_vbel(vehicle.store)
    vehicle.store.set("frax", 0.0)
    ins.initialize(vehicle, _ctx())
    seed(12345)
    _cpp_def_ins_gauss_vectors()
    draws = np.array([gauss(0.0, 1.0) for _ in range(9)])
    xx_init = _cholesky(PP0) @ draws
    esttc = xx_init[0:3]
    evbe = xx_init[3:6]
    rece = xx_init[6:9] * 0.001
    np.testing.assert_allclose(vehicle.store.get("ESTTC"), esttc, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("EVBE"), evbe, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("RECE"), rece, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        vehicle.store.get("SBELC"), esttc + planted["SBEL"], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("VBELC"), evbe + planted["VBEL"], rtol=RTOL, atol=ATOL
    )


def test_define_does_not_register_kinematics_newton_truth_names():
    vehicle = SimpleNamespace(store=StateStore())
    Agm6Ins().define(vehicle)
    store = vehicle.store
    for name in TRUTH:
        with pytest.raises(KeyError):
            store.get(name)
        assert name not in store.names()
    for name in CONTROL_INS:
        store.get(name)


def test_define_control_names_wbecb_fspcb_phiblcx():
    vehicle = SimpleNamespace(store=StateStore())
    Agm6Ins().define(vehicle)
    store = vehicle.store
    for name in CONTROL_INS:
        field = store.field(name)
        assert field.module == "ins"
        assert name in store.names()


def test_initialize_mins_zero_copies_sbel_vbel():
    vehicle, ins = _defined(mins=0)
    planted = _plant_sbel_vbel(vehicle.store)
    ins.initialize(vehicle, _ctx())
    np.testing.assert_allclose(
        vehicle.store.get("SBELC"), planted["SBEL"], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("VBELC"), planted["VBEL"], rtol=RTOL, atol=ATOL
    )
    for name in ERROR_STATES:
        np.testing.assert_array_equal(vehicle.store.get(name), np.zeros(3))


def test_initialize_mins_one_frax_scales_cholesky_draws():
    seed(12345)
    vehicle, ins = _defined(mins=1)
    planted = _plant_sbel_vbel(vehicle.store)
    vehicle.store.set("frax", 10.0)
    ins.initialize(vehicle, _ctx())
    seed(12345)
    _cpp_def_ins_gauss_vectors()
    draws = np.array([gauss(0.0, 1.0) for _ in range(9)])
    xx_init = _cholesky(PP0) @ draws * 11.0
    np.testing.assert_allclose(
        vehicle.store.get("ESTTC"), xx_init[0:3], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("SBELC"),
        xx_init[0:3] + planted["SBEL"],
        rtol=RTOL,
        atol=ATOL,
    )


def test_initialize_mins_two_raises():
    vehicle, ins = _defined(mins=2)
    _plant_sbel_vbel(vehicle.store)
    with pytest.raises(ValueError, match="unknown mins"):
        ins.initialize(vehicle, _ctx())


def test_terminate_exists_and_is_pass():
    vehicle, ins, _truth = _ready(mins=0)
    sentinel = np.array([9.0, 8.0, 7.0])
    vehicle.store.set("SBELC", sentinel)
    assert ins.terminate(vehicle, _ctx()) is None
    np.testing.assert_array_equal(vehicle.store.get("SBELC"), sentinel)


def test_execute_mins_zero_copies_sbel_to_sbelc():
    vehicle, ins, truth = _ready(mins=0)
    ins.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        vehicle.store.get("SBELC"), truth["SBEL"], rtol=RTOL, atol=ATOL
    )


def test_execute_mins_zero_copies_truth_and_common_angles():
    vehicle, ins, truth = _ready(mins=0)
    ins.execute(vehicle, _ctx())
    store = vehicle.store
    np.testing.assert_allclose(store.get("TBLC"), truth["TBL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FSPCB"), truth["FSPB"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("WBECB"), truth["WBEB"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBELC"), truth["SBEL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VBELC"), truth["VBEL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("dvbec"), truth["dvbe"], rtol=RTOL, atol=ATOL)
    want = _cpp_common(store.get("VBELC"), store.get("TBLC"))
    for name, value in want.items():
        np.testing.assert_allclose(store.get(name), value, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(store.get("SBEL"), truth["SBEL"])
    np.testing.assert_array_equal(store.get("TBL"), truth["TBL"])
    np.testing.assert_array_equal(store.get("EWBEB"), np.zeros(3))
    np.testing.assert_array_equal(store.get("EFSPB"), np.zeros(3))
    assert "TLB" not in store.names()


def test_execute_mins_zero_writes_control_names():
    vehicle, ins, truth = _ready(mins=0)
    ins.execute(vehicle, _ctx())
    store = vehicle.store
    want = _cpp_common(truth["VBEL"], truth["TBL"])
    np.testing.assert_allclose(store.get("WBECB"), truth["WBEB"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FSPCB"), truth["FSPB"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("phiblcx"), want["phiblcx"], rtol=RTOL, atol=ATOL)
    for name in CONTROL_INS:
        assert name in store.names()


def test_execute_mins_zero_skips_gyro_accl_helpers():
    vehicle, ins, truth = _ready(mins=0)
    store = vehicle.store
    store.set("EWALKG", (1.0, 1.0, 1.0))
    store.set("EBIASA", (4.0, 0.0, 0.0))
    ins.execute(vehicle, _ctx())
    np.testing.assert_array_equal(store.get("EWG"), np.zeros(3))
    np.testing.assert_array_equal(store.get("EUG"), np.zeros(3))
    np.testing.assert_array_equal(store.get("EWBEB"), np.zeros(3))
    np.testing.assert_array_equal(store.get("EFSPB"), np.zeros(3))
    np.testing.assert_allclose(store.get("FSPCB"), truth["FSPB"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("WBECB"), truth["WBEB"], rtol=RTOL, atol=ATOL)


def test_execute_mins_one_zero_errors_sbelc_approx_sbel():
    vehicle, ins, truth = _ready(mins=1, tlb=True)
    ins.execute(vehicle, _ctx())
    store = vehicle.store
    np.testing.assert_allclose(store.get("SBELC"), truth["SBEL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VBELC"), truth["VBEL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("TBLC"), truth["TBL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FSPCB"), truth["FSPB"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("WBECB"), truth["WBEB"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("dvbec"), np.linalg.norm(truth["VBEL"]), rtol=RTOL, atol=ATOL
    )
    want = _cpp_common(store.get("VBELC"), store.get("TBLC"))
    for name, value in want.items():
        np.testing.assert_allclose(store.get(name), value, rtol=RTOL, atol=ATOL)
    for name in CONTROL_INS:
        store.get(name)


def test_execute_mins_one_planted_evbe_integrates_position():
    vehicle, ins, truth = _ready(mins=1, tlb=True)
    vehicle.store.set("FSPB", (0.0, 0.0, 0.0))
    vehicle.store.set("EVBE", (1.0, 0.0, 0.0))
    ins.execute(vehicle, _ctx(DT))
    want_esttc0 = DT / 2.0
    np.testing.assert_allclose(
        vehicle.store.get("ESTTC")[0], want_esttc0, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("SBELC")[0],
        truth["SBEL"][0] + want_esttc0,
        rtol=RTOL,
        atol=ATOL,
    )
    assert abs(vehicle.store.get("SBELC")[0] - truth["SBEL"][0]) > ATOL


def test_ins_alt_hbem_is_hbe_plus_bias_plus_rand():
    vehicle, ins, _truth = _ready(mins=0)
    store = vehicle.store
    store.set("biasal", BIASAL)
    store.set("randal", RANDAL)
    store.set("hbe", HBE)
    ins.execute(vehicle, _ctx())
    np.testing.assert_allclose(store.get("ehbe"), BIASAL + RANDAL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("hbem"), HBE + BIASAL + RANDAL, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(store.get("hbem"), 7012.0, rtol=RTOL, atol=ATOL)


def test_ins_alt_reads_hbe_not_minus_sbel2():
    vehicle, ins, _truth = _ready(mins=0)
    store = vehicle.store
    store.set("hbe", 5000.0)
    store.set("biasal", 0.0)
    store.set("randal", 0.0)
    ins.execute(vehicle, _ctx())
    np.testing.assert_allclose(store.get("hbem"), 5000.0, rtol=RTOL, atol=ATOL)
    assert store.get("hbem") != -store.get("SBEL")[2]


@pytest.mark.parametrize("mins", (2, -1, 99))
def test_execute_unknown_mins_raises(mins):
    vehicle, ins, _truth = _ready(mins=0)
    vehicle.store.set("mins", mins)
    sentinel = np.array([9.0, 8.0, 7.0])
    vehicle.store.set("SBELC", sentinel)
    with pytest.raises(ValueError, match="unknown mins"):
        ins.execute(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("SBELC"), sentinel)


def test_module_does_not_import_plane6():
    import cadac.vehicles.flat6.agm6.ins as ins_mod

    src = Path(ins_mod.__file__).read_text()
    assert "plane6" not in src.lower()
    assert "Plane6" not in src
    assert "gauss(" not in src
    assert "np.random" not in src
    assert "random." not in src
