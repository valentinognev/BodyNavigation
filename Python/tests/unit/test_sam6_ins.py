from math import acos, asin, atan2, cos, fabs, sqrt
from pathlib import Path

import numpy as np
import pytest

from cadac.constants import AGRAV, DEG, EPS, PI, RAD, REARTH
from cadac.eom.flat6 import Flat6Environment
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat3tr
from cadac.stoch import gauss, seed, uniform
from cadac.vehicles.sam6.ins import Sam6Ins, _PP0, _cholesky, _gauss

# C++ gauss(0,1) after srand(0) on Linux glibc (CADAC unituni Box-Muller).
_CPP_GAUSS01_SRAND0 = (
    -0.34532367182326629,
    1.1122716058967226,
    0.6410062294340324,
    0.60805637857480721,
    1.1591500561471559,
    -0.71208231924892507,
    0.41746552617877442,
    -1.718945215304998,
)

RTOL = 1e-12
ATOL = 1e-14
PLOT = ("plot",)
ZEROS3 = (0.0, 0.0, 0.0)
ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
DT = 0.001

# C++ Missile::def_ins plus init_ins ASpec slots (gauss draws stored as zeros)
# and EWALKG (C++ missile[307], read by ins_gyro, never inited → 0).
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
    "thtvlcx": ("real", "out", 0.0, PLOT),
    "psivlcx": ("real", "out", 0.0, PLOT),
    "FSPCB": ("vec", "out", ZEROS3, ()),
    "phiblcx": ("real", "out", 0.0, PLOT),
    "RECED": ("vec", "state", ZEROS3, ()),
    "RECE": ("vec", "state", ZEROS3, ()),
    "EVBED": ("vec", "state", ZEROS3, ()),
    "EVBE": ("vec", "state", ZEROS3, ()),
    "ESTTCD": ("vec", "state", ZEROS3, ()),
    "ESTTC": ("vec", "state", ZEROS3, ()),
}
DEFINED = tuple(FIELDS)
EXTERNALS = (
    "TBL",
    "TLB",
    "WBEB",
    "SBEL",
    "FSPB",
    "VBEL",
    "dvbe",
    "alt",
    "time",
)
CONTROL_INS = (
    "TBLC",
    "FSPCB",
    "WBECB",
    "thtblcx",
    "phiblcx",
    "SBELC",
    "VBELC",
    "dvbec",
)
ERROR_STATES = ("RECED", "RECE", "EVBED", "EVBE", "ESTTCD", "ESTTC")
ERROR_DATA = (
    "EWALKG",
    "EUNBG",
    "EMISG",
    "ESCALG",
    "EBIASG",
    "EWALKA",
    "EMISA",
    "ESCALA",
    "EBIASA",
)

SBEL = np.array([10.0, -20.0, -1000.0], dtype=float)
VBEL = np.array([16.0, 4.0, -2.0], dtype=float)
FSPB = np.array([1.0, -0.2, 9.5], dtype=float)
WBEB = np.array([0.1, 0.05, -0.02], dtype=float)
DVBE = float(np.linalg.norm(VBEL))
ALT = 1000.0
THT = 10.0 * RAD
PHI = 5.0 * RAD
PSI = 20.0 * RAD


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


def _sign(variable):
    if variable < 0:
        return -1
    return 1


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


def _diamat(vec):
    return np.diag(np.asarray(vec, dtype=float))


def _plant_truth(store, *, tbl=None, tlb=None):
    if tbl is None:
        tbl = mat3tr(PSI, THT, PHI)
    if tlb is None:
        tlb = tbl.T.copy()
    for name, value, ftype, role, module in (
        ("TBL", tbl, "mat", "out", "kinematics"),
        ("TLB", tlb, "mat", "diag", "kinematics"),
        ("WBEB", WBEB, "vec", "diag", "euler"),
        ("SBEL", SBEL, "vec", "state", "newton"),
        ("FSPB", FSPB, "vec", "out", "newton"),
        ("VBEL", VBEL, "vec", "out", "newton"),
        ("dvbe", DVBE, "real", "in/out", "newton"),
        ("alt", ALT, "real", "out", "newton"),
    ):
        if name not in store.names():
            store.define(Field(name, value, ftype, role, module))
        store.set(name, value)
    return {
        "TBL": np.asarray(tbl, dtype=float).copy(),
        "TLB": np.asarray(tlb, dtype=float).copy(),
        "WBEB": np.asarray(WBEB, dtype=float).copy(),
        "SBEL": np.asarray(SBEL, dtype=float).copy(),
        "FSPB": np.asarray(FSPB, dtype=float).copy(),
        "VBEL": np.asarray(VBEL, dtype=float).copy(),
        "dvbe": DVBE,
        "alt": ALT,
    }


def _defined(mins=0):
    vehicle = _Vehicle()
    ins = Sam6Ins()
    ins.define(vehicle)
    vehicle.store.set("mins", mins)
    return vehicle, ins


def _ready(mins=0, **plant):
    seed(0)
    vehicle, ins = _defined(mins=mins)
    truth = _plant_truth(vehicle.store, **plant)
    ins.initialize(vehicle, _ctx())
    return vehicle, ins, truth


def _cpp_euler_fpa(tblc, vbelc):
    vbelc1 = float(vbelc[0])
    vbelc2 = float(vbelc[1])
    vbelc3 = float(vbelc[2])
    if vbelc1 == 0 and vbelc2 == 0:
        psivlc = 0.0
        thtvlc = 0.0
    else:
        psivlc = atan2(vbelc2, vbelc1)
        thtvlc = atan2(-vbelc3, sqrt(vbelc1 * vbelc1 + vbelc2 * vbelc2))
    tblc13 = float(tblc[0, 2])
    if fabs(tblc13) < 1:
        thtblc = asin(-tblc13)
        cthtblc = cos(thtblc)
    else:
        thtblc = PI / 2 * _sign(-tblc13)
        cthtblc = EPS
    tblc23 = float(tblc[1, 2])
    tblc33 = float(tblc[2, 2])
    cphic = tblc33 / cthtblc
    if fabs(cphic) >= 1:
        cphic = (1 - EPS) * _sign(cphic)
    phiblc = acos(cphic) * _sign(tblc23)
    return {
        "psivlcx": psivlc * DEG,
        "thtvlc": thtvlc,
        "thtvlcx": thtvlc * DEG,
        "thtblc": thtblc,
        "thtblcx": thtblc * DEG,
        "phiblcx": phiblc * DEG,
    }


def _cpp_ins_gyro(wbeb, fspb, ewalkg, eunbg, emisg, escalg, ebiasg, int_step):
    egb = _diamat(escalg) + _skew(emisg)
    emiscg = egb @ wbeb
    emsbg = ebiasg + emiscg
    eug = eunbg * fspb
    ewg = ewalkg * (1.0 / sqrt(int_step))
    ewbeb = emsbg + eug + ewg
    wbecb = wbeb + ewbeb
    return ewbeb, wbecb, eug, ewg


def _cpp_ins_accl(fspb, emisa, escala, ebiasa):
    eab = _diamat(escala) + _skew(emisa)
    return ebiasa + eab @ fspb


def _cpp_ins_mins1(truth, store, int_step):
    reced = np.asarray(store.get("RECED"), dtype=float).copy()
    rece = np.asarray(store.get("RECE"), dtype=float).copy()
    evbed = np.asarray(store.get("EVBED"), dtype=float).copy()
    evbe = np.asarray(store.get("EVBE"), dtype=float).copy()
    esttcd = np.asarray(store.get("ESTTCD"), dtype=float).copy()
    esttc = np.asarray(store.get("ESTTC"), dtype=float).copy()
    tanlat = store.get("tanlat")
    ewalka = np.asarray(store.get("EWALKA"), dtype=float).copy()
    tbl = truth["TBL"]
    tlb = truth["TLB"]
    wbeb = truth["WBEB"]
    sbel = truth["SBEL"]
    fspb = truth["FSPB"]
    vbel = truth["VBEL"]
    efspb = _cpp_ins_accl(
        fspb,
        store.get("EMISA"),
        store.get("ESCALA"),
        store.get("EBIASA"),
    )
    ewbeb, wbecb, eug, ewg = _cpp_ins_gyro(
        wbeb,
        fspb,
        store.get("EWALKG"),
        store.get("EUNBG"),
        store.get("EMISG"),
        store.get("ESCALG"),
        store.get("EBIASG"),
        int_step,
    )
    ewbel = tlb @ ewbeb
    reced_new = np.array(
        [
            ewbel[0] + evbe[1] / REARTH,
            ewbel[1] - evbe[0] / REARTH,
            ewbel[2] - evbe[1] * tanlat / REARTH,
        ],
        dtype=float,
    )
    rece = integrate(reced_new, reced, rece, int_step)
    reced = reced_new
    rere = _skew(rece)
    tllc = rere + np.eye(3)
    tblc = tbl @ tllc
    tlcb = tblc.T
    walka3 = uniform(-ewalka[2], ewalka[2])
    walka2 = uniform(-ewalka[1], ewalka[1])
    walka1 = uniform(-ewalka[0], ewalka[0])
    walka = np.array([walka1, walka2, walka3], dtype=float)
    fspcb = walka + efspb + fspb
    ef = tlcb @ efspb - rere @ tlcb @ fspcb
    evbed_new = np.array(
        [ef[0], ef[1], ef[2] + 2.0 * AGRAV * esttc[2] / REARTH],
        dtype=float,
    )
    evbe = integrate(evbed_new, evbed, evbe, int_step)
    evbed = evbed_new
    esttcd_new = evbe
    esttc = integrate(esttcd_new, esttcd, esttc, int_step)
    esttcd = esttcd_new
    sbelc = esttc + sbel
    vbelc = evbe + vbel
    dvbec = float(np.linalg.norm(vbelc))
    angles = _cpp_euler_fpa(tblc, vbelc)
    return {
        "RECED": reced,
        "RECE": rece,
        "EVBED": evbed,
        "EVBE": evbe,
        "ESTTCD": esttcd,
        "ESTTC": esttc,
        "TBLC": tblc,
        "FSPCB": fspcb,
        "WBECB": wbecb,
        "SBELC": sbelc,
        "VBELC": vbelc,
        "dvbec": dvbec,
        "EWBEB": ewbeb,
        "EFSPB": efspb,
        "EUG": eug,
        "EWG": ewg,
        "hbem": truth["alt"],
        "ehbe": 0.0,
        **angles,
    }


def _gauss3_rtl(sig):
    third = gauss(0.0, sig)
    second = gauss(0.0, sig)
    first = gauss(0.0, sig)
    return np.array([first, second, third], dtype=float)


def _cpp_init_aspec_cholesky(sbel, vbel, frax=0.0):
    errors = {
        "EUNBG": np.zeros(3),
        "EMISG": _gauss3_rtl(1.1e-4),
        "ESCALG": _gauss3_rtl(2.5e-5),
        "EBIASG": _gauss3_rtl(3.2e-6),
        "EWALKA": _gauss3_rtl(8.35e-4),
        "EMISA": _gauss3_rtl(1.1e-4),
        "ESCALA": _gauss3_rtl(5e-4),
        "EBIASA": _gauss3_rtl(3.56e-3),
        "EWALKG": np.zeros(3),
    }
    draws = np.array([gauss(0.0, 1.0) for _ in range(9)], dtype=float)
    xx_init = _cholesky(_PP0) @ draws * (1.0 + frax)
    esttc = xx_init[0:3]
    evbe = xx_init[3:6]
    rece = xx_init[6:9] * 0.001
    errors["ESTTC"] = esttc
    errors["EVBE"] = evbe
    errors["RECE"] = rece
    errors["SBELC"] = esttc + sbel
    errors["VBELC"] = evbe + vbel
    return errors


def test_gauss_after_srand0_matches_cpp_box_muller():
    seed(0)
    got = [_gauss(0.0, 1.0) for _ in _CPP_GAUSS01_SRAND0]
    np.testing.assert_allclose(got, _CPP_GAUSS01_SRAND0, rtol=0.0, atol=1e-15)


def test_name_is_ins():
    assert Sam6Ins().name == "ins"


def test_define_registers_cpp_def_ins_fields():
    vehicle = _Vehicle()
    Sam6Ins().define(vehicle)
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
            np.testing.assert_array_equal(store.get(name), zeros3)
            assert store.get(name).shape == (3,)
        else:
            np.testing.assert_array_equal(store.get(name), zeros33)
            assert store.get(name).shape == (3, 3)
    for name in EXTERNALS:
        assert name not in store.names()


def test_define_does_not_register_kinematics_newton_euler_names():
    vehicle = _Vehicle()
    Sam6Ins().define(vehicle)
    for name in EXTERNALS:
        with pytest.raises(KeyError):
            vehicle.store.get(name)


def test_mins0_execute_copies_sbel_to_sbelc_and_fspb_to_fspcb():
    vehicle, ins, truth = _ready(mins=0)
    ins.execute(vehicle, _ctx())
    store = vehicle.store
    np.testing.assert_allclose(store.get("SBELC"), truth["SBEL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FSPCB"), truth["FSPB"], rtol=RTOL, atol=ATOL)


def test_mins0_execute_copies_truth_then_euler_fpa():
    vehicle, ins, truth = _ready(mins=0)
    ins.execute(vehicle, _ctx())
    store = vehicle.store
    np.testing.assert_allclose(store.get("TBLC"), truth["TBL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("WBECB"), truth["WBEB"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VBELC"), truth["VBEL"], rtol=RTOL, atol=ATOL)
    assert _approx(store.get("dvbec"), truth["dvbe"])
    want = _cpp_euler_fpa(truth["TBL"], truth["VBEL"])
    for name, value in want.items():
        assert _approx(store.get(name), value)
    np.testing.assert_array_equal(store.get("EWBEB"), np.zeros(3))
    np.testing.assert_array_equal(store.get("EFSPB"), np.zeros(3))


def test_mins1_init_applies_cholesky_after_aspec_gauss():
    vehicle, ins, truth = _ready(mins=1)
    store = vehicle.store
    seed(0)
    want = _cpp_init_aspec_cholesky(truth["SBEL"], truth["VBEL"])
    np.testing.assert_allclose(store.get("SBELC"), want["SBELC"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VBELC"), want["VBELC"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ESTTC"), want["ESTTC"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("EVBE"), want["EVBE"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("RECE"), want["RECE"], rtol=RTOL, atol=ATOL)
    for name in ("EMISG", "ESCALG", "EBIASG", "EWALKA", "EMISA", "ESCALA", "EBIASA"):
        np.testing.assert_allclose(store.get(name), want[name], rtol=RTOL, atol=ATOL)
    assert abs(store.get("EVBE")[0]) > 1e-6


def test_mins1_one_execute_wbecb_includes_gyro_errors():
    vehicle, ins, truth = _ready(mins=1)
    store = vehicle.store
    ewbeb, wbecb, eug, ewg = _cpp_ins_gyro(
        truth["WBEB"],
        truth["FSPB"],
        store.get("EWALKG"),
        store.get("EUNBG"),
        store.get("EMISG"),
        store.get("ESCALG"),
        store.get("EBIASG"),
        DT,
    )
    ins.execute(vehicle, _ctx())
    np.testing.assert_allclose(store.get("WBECB"), wbecb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("EWBEB"), ewbeb, rtol=RTOL, atol=ATOL)
    assert np.linalg.norm(store.get("WBECB") - truth["WBEB"]) > 1e-12


def test_mins1_one_execute_matches_cadac_error_odes():
    vehicle, ins, truth = _ready(mins=1)
    ctx = _ctx()
    ins.execute(vehicle, ctx)
    store = vehicle.store
    seed(0)
    replay, replay_ins, replay_truth = _ready(mins=1)
    want = _cpp_ins_mins1(replay_truth, replay.store, ctx.int_step)
    for name, value in want.items():
        got = store.get(name)
        if isinstance(value, np.ndarray):
            np.testing.assert_allclose(got, value, rtol=RTOL, atol=ATOL)
        else:
            assert _approx(got, value)


def test_mins2_initialize_raises():
    vehicle, ins = _defined(mins=2)
    with pytest.raises(ValueError, match="unknown mins"):
        ins.initialize(vehicle, _ctx())


def test_mins2_execute_raises():
    vehicle, ins = _defined(mins=2)
    with pytest.raises(ValueError, match="unknown mins"):
        ins.execute(vehicle, _ctx())


@pytest.mark.parametrize("mins", [-1, 3, 99])
def test_mins_not_0_or_1_raises(mins):
    vehicle, ins = _defined(mins=mins)
    with pytest.raises(ValueError, match="unknown mins"):
        ins.initialize(vehicle, _ctx())
    with pytest.raises(ValueError, match="unknown mins"):
        ins.execute(vehicle, _ctx())


def test_ins_alt_zero_bias_and_noise():
    vehicle, ins, truth = _ready(mins=0)
    ins.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("biasal") == 0.0
    assert store.get("randal") == 0.0
    assert _approx(store.get("ehbe"), 0.0)
    assert _approx(store.get("hbem"), truth["alt"])


def test_execute_writes_control_ins_names():
    vehicle, ins, truth = _ready(mins=1)
    ins.execute(vehicle, _ctx())
    store = vehicle.store
    for name in CONTROL_INS:
        value = store.get(name)
        if isinstance(value, np.ndarray):
            assert np.all(np.isfinite(value))
        else:
            assert np.isfinite(value)
    assert np.all(np.isfinite(store.get("FSPCB")))
    assert np.all(np.isfinite(store.get("WBECB")))
    assert np.isfinite(store.get("thtblcx"))
    assert np.isfinite(store.get("phiblcx"))


def test_euler_singularity_uses_cadac_sign():
    tbl = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    vehicle, ins, truth = _ready(mins=0, tbl=tbl, tlb=tbl.T.copy())
    ins.execute(vehicle, _ctx())
    want = _cpp_euler_fpa(truth["TBL"], truth["VBEL"])
    assert _approx(vehicle.store.get("thtblc"), want["thtblc"])
    assert _approx(vehicle.store.get("phiblcx"), want["phiblcx"])


def test_cadac_sign_zero_tblc23_is_plus_one():
    tbl = np.eye(3)
    vbel = np.array([16.0, 0.0, 0.0], dtype=float)
    vehicle, ins = _defined(mins=0)
    truth = _plant_truth(vehicle.store, tbl=tbl, tlb=tbl.T.copy())
    vehicle.store.set("VBEL", vbel)
    ins.initialize(vehicle, _ctx())
    ins.execute(vehicle, _ctx())
    want = _cpp_euler_fpa(truth["TBL"], vbel)
    got = vehicle.store.get("phiblcx")
    assert _approx(got, want["phiblcx"])
    assert got > 0.0


def test_terminate_exists_and_is_pass():
    vehicle, ins, _truth = _ready(mins=0)
    store = vehicle.store
    sentinel = np.array([9.0, 8.0, 7.0])
    store.set("SBELC", sentinel)
    assert ins.terminate(vehicle, _ctx()) is None
    np.testing.assert_array_equal(store.get("SBELC"), sentinel)


def test_not_a_subclass_of_flat6_and_no_hyper_plane_ins():
    assert not issubclass(Sam6Ins, Flat6Environment)
    import cadac.vehicles.sam6.ins as insmod

    assert "Flat6Environment" not in dir(insmod)
    assert "Hyper6Ins" not in dir(insmod)


def test_no_flat6_or_plane_or_hyper_ins_imports():
    import cadac.vehicles.sam6.ins as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6" not in src
    assert "from cadac.eom.flat6 import" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src
    assert "Hyper6Ins" not in src
    assert "Plane6" not in src
