import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import DEG, EPS, PI, RAD, WEII3
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import cadac_matmul, mat3tr
from cadac.math.wgs84 import cad_geo84_in, cad_in_geo84, cad_tdi84
from cadac.vehicles.rocket6.ins import Rocket6Ins, _geodetic_euler_from_tbd

RTOL = 1e-12
ATOL = 1e-14
PLOT = ("plot",)
SCRN_PLOT = ("scrn", "plot")
ZEROS3 = (0.0, 0.0, 0.0)
ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

# C++ Hyper::def_ins order (ROCKET6 ins.cpp); unused slots 302, 314, 323 omitted.
# Instrument gauss() vectors default to 0 (Monte Carlo out of scope).
FIELDS = {
    "mins": ("int", "data", 0, ()),
    "frax_algnmnt": ("real", "data", 0.0, ()),
    "VBIIC": ("vec", "out", ZEROS3, ()),
    "SBIIC": ("vec", "out", ZEROS3, ()),
    "WBICI": ("vec", "out", ZEROS3, ()),
    "WBICB": ("vec", "out", ZEROS3, ()),
    "EWALKG": ("vec", "data", ZEROS3, ()),
    "EUNBG": ("vec", "data", ZEROS3, ()),
    "EMISG": ("vec", "data", ZEROS3, ()),
    "ESCALG": ("vec", "data", ZEROS3, ()),
    "EBIASG": ("vec", "data", ZEROS3, ()),
    "EUG": ("vec", "diag", ZEROS3, ()),
    "EWG": ("vec", "diag", ZEROS3, ()),
    "TBIC": ("mat", "out", ZEROS33, ()),
    "EWALKA": ("vec", "data", ZEROS3, ()),
    "EMISA": ("vec", "data", ZEROS3, ()),
    "ESCALA": ("vec", "data", ZEROS3, ()),
    "EBIASA": ("vec", "data", ZEROS3, ()),
    "ppcx": ("real", "out", 0.0, ()),
    "qqcx": ("real", "out", 0.0, ()),
    "rrcx": ("real", "out", 0.0, ()),
    "EWBIB": ("vec", "diag", ZEROS3, ()),
    "EFSPB": ("vec", "diag", ZEROS3, ()),
    "loncx": ("real", "out", 0.0, ()),
    "latcx": ("real", "out", 0.0, ()),
    "altc": ("real", "out", 0.0, ()),
    "VBECD": ("vec", "out", ZEROS3, ()),
    "dvbec": ("real", "out", 0.0, ()),
    "TDCI": ("mat", "out", ZEROS33, ()),
    "thtvdcx": ("real", "out", 0.0, ()),
    "psivdcx": ("real", "out", 0.0, ()),
    "FSPCB": ("vec", "out", ZEROS3, ()),
    "dbic": ("real", "out", 0.0, ()),
    "alphacx": ("real", "out", 0.0, PLOT),
    "betacx": ("real", "out", 0.0, PLOT),
    "phibdcx": ("real", "out", 0.0, PLOT),
    "thtbdcx": ("real", "out", 0.0, PLOT),
    "psibdcx": ("real", "out", 0.0, PLOT),
    "alppcx": ("real", "out", 0.0, ()),
    "phipcx": ("real", "diag", 0.0, ()),
    "RICID": ("vec", "state", ZEROS3, ()),
    "RICI": ("vec", "state", ZEROS3, PLOT),
    "EVBID": ("vec", "state", ZEROS3, ()),
    "EVBI": ("vec", "state", ZEROS3, PLOT),
    "ESBID": ("vec", "state", ZEROS3, ()),
    "ESBI": ("vec", "state", ZEROS3, PLOT),
    "ins_pos_err": ("real", "diag", 0.0, SCRN_PLOT),
    "ins_vel_err": ("real", "diag", 0.0, SCRN_PLOT),
    "ins_tilt_err": ("real", "diag", 0.0, SCRN_PLOT),
    "frax_transfer": ("real", "data", 0.0, ()),
    "eunbg": ("real", "data", 0.0, ()),
}
DEFINED = tuple(FIELDS)
GAUSS_INSTRUMENTS = ("EMISG", "ESCALG", "EBIASG", "EMISA", "ESCALA", "EBIASA")
EXTERNALS = (
    "time",
    "TBI",
    "WBIB",
    "WBII",
    "SBII",
    "VBII",
    "FSPB",
    "GRAVG",
    "mroll",
    "mgps",
    "mstar",
    "SXH",
    "VXH",
    "URIC",
)
GPS_STAR = ("mgps", "mstar", "SXH", "VXH", "URIC")
ERROR_STATES = ("RICID", "RICI", "EVBID", "EVBI", "ESBID", "ESBI")
HYPER6_ONLY = ("frax",)


def _ctx(int_step=0.001):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _cadac_sign(variable):
    if variable < 0.0:
        return -1
    return 1


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
    for name, value, ftype, role, module in (
        ("time", time, "real", "exec", "kinematics"),
        ("TBI", tbi, "mat", "state", "kinematics"),
        ("WBIB", wbib, "vec", "state", "euler"),
        ("WBII", wbii, "vec", "out", "euler"),
        ("SBII", sbii, "vec", "state", "newton"),
        ("VBII", vbii, "vec", "state", "newton"),
        ("FSPB", fspb, "vec", "out", "newton"),
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
    }


def _cpp_ins_mins0(tbi, fspb, wbib, wbii, sbii, vbii, time, mroll=0):
    tbic = np.asarray(tbi, dtype=float).copy()
    fspcb = np.asarray(fspb, dtype=float).copy()
    wbici = np.asarray(wbii, dtype=float).copy()
    wbicb = np.asarray(wbib, dtype=float).copy()
    sbiic = np.asarray(sbii, dtype=float).copy()
    vbiic = np.asarray(vbii, dtype=float).copy()
    dbic = float(np.linalg.norm(sbiic))

    veic = np.array(
        [-WEII3 * sbiic[1], WEII3 * sbiic[0], 0.0],
        dtype=float,
    )
    vbeic = vbiic - veic
    vbecb = cadac_matmul(tbic, vbeic)
    dvbec = float(np.linalg.norm(vbecb))

    ppcx = wbicb[0] * DEG
    qqcx = wbicb[1] * DEG
    rrcx = wbicb[2] * DEG

    alphac = math.atan2(vbecb[2], vbecb[0])
    betac = math.asin(vbecb[1] / dvbec)
    alphacx = alphac * DEG
    betacx = betac * DEG

    dum = vbecb[0] / dvbec
    if math.fabs(dum) > 1.0:
        dum = 1.0 * _cadac_sign(dum)
    alppc = math.acos(dum)
    if vbecb[1] == 0.0 and vbecb[2] == 0.0:
        phipc = 0.0
    elif math.fabs(vbecb[1]) < EPS:
        phipc = 0.0
        if vbecb[2] > 0.0:
            phipc = 0.0
        if vbecb[2] < 0.0:
            phipc = PI
    else:
        phipc = math.atan2(vbecb[1], vbecb[2])
    alppcx = alppc * DEG
    phipcx = phipc * DEG

    lonc, latc, altc = cad_geo84_in(sbiic, time)
    tdci = cad_tdi84(lonc, latc, altc, time)
    loncx = lonc * DEG
    latcx = latc * DEG
    vbecd = cadac_matmul(tdci, vbeic)

    if vbecd[0] == 0.0 and vbecd[1] == 0.0:
        psivdc = 0.0
        thtvdc = 0.0
    else:
        psivdc = math.atan2(vbecd[1], vbecd[0])
        thtvdc = math.atan2(
            -vbecd[2], math.sqrt(vbecd[0] * vbecd[0] + vbecd[1] * vbecd[1])
        )
    psivdcx = psivdc * DEG
    thtvdcx = thtvdc * DEG

    tbd = cadac_matmul(tbic, tdci.T.copy())
    psibdc, thtbdc, phibdc = _geodetic_euler_from_tbd(tbd, mroll)
    return {
        "TBIC": tbic,
        "FSPCB": fspcb,
        "WBICI": wbici,
        "WBICB": wbicb,
        "SBIIC": sbiic,
        "VBIIC": vbiic,
        "dbic": dbic,
        "ppcx": ppcx,
        "qqcx": qqcx,
        "rrcx": rrcx,
        "alphacx": alphacx,
        "betacx": betacx,
        "alppcx": alppcx,
        "phipcx": phipcx,
        "loncx": loncx,
        "latcx": latcx,
        "altc": altc,
        "TDCI": tdci,
        "VBECD": vbecd,
        "dvbec": dvbec,
        "psivdcx": psivdcx,
        "thtvdcx": thtvdcx,
        "psibdcx": DEG * psibdc,
        "thtbdcx": DEG * thtbdc,
        "phibdcx": DEG * phibdc,
    }


def _defined(mins=0):
    vehicle = SimpleNamespace(store=StateStore())
    ins = Rocket6Ins()
    ins.define(vehicle)
    vehicle.store.set("mins", mins)
    return vehicle, ins


def _ready(mins=0):
    vehicle, ins = _defined(mins=mins)
    truth = _plant_truth(vehicle.store)
    ins.initialize(vehicle, _ctx())
    return vehicle, ins, truth


def test_name_is_ins():
    assert Rocket6Ins().name == "ins"


def test_define_registers_cpp_def_ins_fields():
    vehicle = SimpleNamespace(store=StateStore())
    Rocket6Ins().define(vehicle)
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
    for name in GAUSS_INSTRUMENTS:
        np.testing.assert_array_equal(store.get(name), zeros3)
    for name in EXTERNALS:
        assert name not in store.names()
    for name in HYPER6_ONLY:
        assert name not in store.names()


def test_define_does_not_register_kinematics_newton_euler_names():
    vehicle = SimpleNamespace(store=StateStore())
    Rocket6Ins().define(vehicle)
    store = vehicle.store
    for name in ("TBI", "FSPB", "SBII", "VBII", "WBIB", "WBII", "time"):
        with pytest.raises(KeyError):
            store.get(name)


def test_initialize_mins_zero_is_noop():
    vehicle, ins = _defined(mins=0)
    truth = _plant_truth(vehicle.store)
    ins.initialize(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("SBIIC"), np.zeros(3))
    np.testing.assert_array_equal(vehicle.store.get("VBIIC"), np.zeros(3))
    np.testing.assert_array_equal(vehicle.store.get("TBIC"), np.zeros((3, 3)))
    np.testing.assert_array_equal(vehicle.store.get("FSPCB"), np.zeros(3))
    np.testing.assert_array_equal(vehicle.store.get("WBICB"), np.zeros(3))
    np.testing.assert_array_equal(vehicle.store.get("SBII"), truth["SBII"])
    assert vehicle.store.get("mins") == 0
    assert vehicle.store.get("frax_algnmnt") == 0.0
    assert vehicle.store.get("dbic") == 0.0
    for name in ERROR_STATES:
        np.testing.assert_array_equal(vehicle.store.get(name), np.zeros(3))


def test_initialize_mins_two_raises():
    vehicle, ins = _defined(mins=2)
    with pytest.raises(ValueError, match="unknown mins"):
        ins.initialize(vehicle, _ctx())
    vehicle.store.set("mins", -1)
    with pytest.raises(ValueError, match="unknown mins"):
        ins.initialize(vehicle, _ctx())


def test_terminate_exists_and_is_pass():
    vehicle, ins, _truth = _ready()
    store = vehicle.store
    store.set("mins", 0)
    sentinel = np.array([9.0, 8.0, 7.0])
    store.set("SBIIC", sentinel)
    ins.terminate(vehicle, _ctx())
    assert store.get("mins") == 0
    np.testing.assert_array_equal(store.get("SBIIC"), sentinel)


def test_module_does_not_import_hyper6():
    import cadac.vehicles.rocket6.ins as ins_mod

    text = Path(ins_mod.__file__).read_text(encoding="utf-8")
    assert "hyper6" not in text.lower()
    assert "Hyper6Ins" not in text


def test_execute_mins_zero_copies_sbii_to_sbiic():
    vehicle, ins, truth = _ready(mins=0)
    ins.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        vehicle.store.get("SBIIC"), truth["SBII"], rtol=RTOL, atol=ATOL
    )


def test_execute_mins_zero_copies_tbi_to_tbic():
    vehicle, ins, truth = _ready(mins=0)
    ins.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        vehicle.store.get("TBIC"), truth["TBI"], rtol=RTOL, atol=ATOL
    )


def test_execute_mins_zero_copies_truth_into_computed():
    vehicle, ins, truth = _ready(mins=0)
    ins.execute(vehicle, _ctx())
    store = vehicle.store
    np.testing.assert_allclose(store.get("TBIC"), truth["TBI"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FSPCB"), truth["FSPB"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBIIC"), truth["SBII"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VBIIC"), truth["VBII"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("WBICB"), truth["WBIB"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("WBICI"), truth["WBII"], rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(store.get("SBII"), truth["SBII"])
    np.testing.assert_array_equal(store.get("TBI"), truth["TBI"])


def test_execute_mins_zero_writes_lon_lat_alt_euler_flight_path():
    vehicle, ins, truth = _ready(mins=0)
    want = _cpp_ins_mins0(
        truth["TBI"],
        truth["FSPB"],
        truth["WBIB"],
        truth["WBII"],
        truth["SBII"],
        truth["VBII"],
        truth["time"],
    )
    ins.execute(vehicle, _ctx())
    store = vehicle.store
    for name, value in want.items():
        np.testing.assert_allclose(store.get(name), value, rtol=RTOL, atol=ATOL)


def test_execute_mins_zero_does_not_require_gps_star_or_mroll():
    vehicle, ins, _truth = _ready(mins=0)
    store = vehicle.store
    for name in GPS_STAR + ("mroll",):
        assert name not in store.names()
    ins.execute(vehicle, _ctx())
    for name in GPS_STAR + ("mroll",):
        assert name not in store.names()
    np.testing.assert_array_equal(store.get("EWBIB"), np.zeros(3))
    np.testing.assert_array_equal(store.get("EFSPB"), np.zeros(3))
    assert store.get("ins_pos_err") == 0.0
    assert store.get("ins_vel_err") == 0.0
    assert store.get("ins_tilt_err") == 0.0
    for name in ERROR_STATES:
        np.testing.assert_array_equal(store.get(name), np.zeros(3))


def test_execute_mins_zero_leaves_planted_gps_star_unchanged():
    vehicle, ins, _truth = _ready(mins=0)
    store = vehicle.store
    store.define(Field("mgps", 3, "int", "data", "gps"))
    store.define(Field("mstar", 3, "int", "data", "startrack"))
    store.define(Field("SXH", (1.0, 2.0, 3.0), "vec", "out", "gps"))
    store.define(Field("VXH", (4.0, 5.0, 6.0), "vec", "out", "gps"))
    store.define(Field("URIC", (0.1, 0.2, 0.3), "vec", "out", "startrack"))
    ins.execute(vehicle, _ctx())
    assert store.get("mgps") == 3
    assert store.get("mstar") == 3
    np.testing.assert_array_equal(store.get("SXH"), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(store.get("VXH"), np.array([4.0, 5.0, 6.0]))
    np.testing.assert_array_equal(store.get("URIC"), np.array([0.1, 0.2, 0.3]))


def test_execute_mins_two_raises():
    vehicle, ins, truth = _ready(mins=0)
    vehicle.store.set("mins", 2)
    sentinel = np.array([9.0, 8.0, 7.0])
    vehicle.store.set("SBIIC", sentinel)
    with pytest.raises(ValueError, match="unknown mins"):
        ins.execute(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("SBIIC"), sentinel)
    np.testing.assert_array_equal(vehicle.store.get("SBII"), truth["SBII"])


@pytest.mark.parametrize("mins", (2, -1, 99))
def test_execute_unknown_mins_raises(mins):
    vehicle, ins, _truth = _ready(mins=0)
    vehicle.store.set("mins", mins)
    with pytest.raises(ValueError, match="unknown mins"):
        ins.execute(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("SBIIC"), np.zeros(3))
