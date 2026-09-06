from math import acos, atan2, cos, fabs, sin, tan
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import DEG
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr, polar_from_cart
from cadac.vehicles.agm6.sensor import Agm6Sensor

RTOL = 1e-12
ATOL = 1e-14
SMALL = 1.0e-7
PLOT = ("plot",)
SCRN = ("scrn",)
COM = ("com",)
ZEROS3 = (0.0, 0.0, 0.0)
ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

# C++ Missile::def_sensor order, then undeclared execute slots timeac/dbtk.
FIELDS = {
    "tgt_num": ("int", "data", 0, ()),
    "STEL": ("vec", "", ZEROS3, ()),
    "VTEL": ("vec", "", ZEROS3, ()),
    "tgt_com_slot": ("int", "out", 0, ()),
    "mseek": ("int", "data/diag", 0, COM),
    "skr_dyn": ("int", "data", 0, ()),
    "isets1": ("int", "init", 0, ()),
    "epchac": ("real", "init", 0.0, ()),
    "ibreak": ("int", "init", 0, ()),
    "dblind": ("real", "data", 0.0, ()),
    "racq": ("real", "data", 0.0, ()),
    "dtimac": ("real", "data", 0.0, ()),
    "gk": ("real", "data", 0.0, ()),
    "zetak": ("real", "data", 0.0, ()),
    "wnk": ("real", "data", 0.0, ()),
    "biast": ("real", "data", 0.0, ()),
    "randt": ("real", "data", 0.0, ()),
    "biasp": ("real", "data", 0.0, ()),
    "randp": ("real", "data", 0.0, ()),
    "wlq1d": ("real", "state", 0.0, ()),
    "wlq1": ("real", "state", 0.0, ()),
    "wlqd": ("real", "state", 0.0, ()),
    "wlq": ("real", "state", 0.0, ()),
    "wlr1d": ("real", "state", 0.0, ()),
    "wlr1": ("real", "state", 0.0, ()),
    "wlrd": ("real", "state", 0.0, ()),
    "wlr": ("real", "state", 0.0, ()),
    "wlq2d": ("real", "state", 0.0, ()),
    "wlq2": ("real", "state", 0.0, ()),
    "wlr2d": ("real", "state", 0.0, ()),
    "wlr2": ("real", "state", 0.0, ()),
    "fovyaw": ("real", "data", 0.0, ()),
    "fovpitch": ("real", "data", 0.0, ()),
    "dba": ("real", "diag", 0.0, ()),
    "daim": ("real", "data", 0.0, ()),
    "BIASAI": ("vec", "data", ZEROS3, ()),
    "BIASSC": ("vec", "data", ZEROS3, ()),
    "RANDSC": ("vec", "data", ZEROS3, ()),
    "epy": ("real", "diag", 0.0, PLOT),
    "epz": ("real", "diag", 0.0, PLOT),
    "thtpb": ("real", "out", 0.0, PLOT),
    "psipb": ("real", "out", 0.0, PLOT),
    "ththb": ("real", "diag", 0.0, PLOT),
    "phihb": ("real", "diag", 0.0, PLOT),
    "TPB": ("mat", "init", ZEROS33, ()),
    "THB": ("mat", "init", ZEROS33, ()),
    "dvbtc": ("real", "diag", 0.0, ()),
    "EAHH": ("vec", "diag", ZEROS3, ()),
    "EPHH": ("vec", "diag", ZEROS3, ()),
    "EAPH": ("vec", "diag", ZEROS3, ()),
    "thtpbx": ("real", "diag", 0.0, ()),
    "psipbx": ("real", "diag", 0.0, ()),
    "sigdpy": ("real", "out", 0.0, PLOT),
    "sigdpz": ("real", "out", 0.0, PLOT),
    "biaseh": ("real", "data", 0.0, ()),
    "randeh": ("real", "data", 0.0, ()),
    "SBTL": ("vec", "diag", ZEROS3, SCRN),
    "epaz_saved": ("real", "save", 0.0, ()),
    "epel_saved": ("real", "save", 0.0, ()),
    "range_saved": ("real", "save", 0.0, ()),
    "rate_saved": ("real", "save", 0.0, ()),
    "timeac": ("real", "save", 0.0, ()),
    "dbtk": ("real", "diag", 0.0, ()),
}
DEFINED = tuple(FIELDS)
EXTERNALS = (
    "time",
    "SBEL",
    "VBEL",
    "TBL",
    "trcond",
    "mguid",
    "fovlimx",
    "racq_irs",
    "fovyawx_irs",
    "fovpitchx_irs",
)

RACQ = 7000.0
DTIMAC = 0.3
SBEL = np.array([0.0, 0.0, -7000.0], dtype=float)
VBEL = np.array([250.0, 0.0, 0.0], dtype=float)
SAEL = np.array([3000.0, 4000.0, -7000.0], dtype=float)
VAEL = np.array([0.0, -5.0, 0.0], dtype=float)
SAEL_FAR = np.array([20000.0, 0.0, -7000.0], dtype=float)
SAEL_TWO = np.array([1000.0, 2000.0, -6500.0], dtype=float)
VAEL_TWO = np.array([1.0, 2.0, 3.0], dtype=float)
TBL = np.eye(3)


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


def _sensor_ir_kin(sbtl, vtel, dbtk, tbl, vbel):
    stbl = -sbtl
    stbb = tbl @ stbl
    utbl = stbl / dbtk
    vtbl = vtel - vbel
    dvbtc = abs(float(utbl @ vtbl))
    woeb = tbl @ _skew(utbl) @ vtbl / dbtk
    polar = polar_from_cart(stbb)
    psipb = float(polar[1])
    thtpb = float(polar[2])
    tpb = mat2tr(psipb, thtpb)
    woep = tpb @ woeb
    return thtpb, psipb, float(woep[1]), float(woep[2]), dvbtc


def _sensor_ir_uthpb(psipb, thtpb):
    ththb = acos(cos(thtpb) * cos(psipb))
    sinpsi = sin(psipb)
    tantht = tan(thtpb)
    if fabs(sinpsi) and fabs(tantht) < SMALL:
        phihb = 0.0
    else:
        phihb = atan2(sinpsi, tantht)
    return ththb, phihb


def _sensor_ir_thb(tht, phi):
    thb = np.zeros((3, 3))
    thb[0, 0] = cos(tht)
    thb[2, 0] = sin(tht)
    thb[1, 1] = cos(phi)
    thb[1, 2] = sin(phi)
    thb[0, 1] = thb[2, 0] * thb[1, 2]
    thb[0, 2] = (-thb[2, 0]) * thb[1, 1]
    thb[2, 1] = (-thb[0, 0]) * thb[1, 2]
    thb[2, 2] = thb[0, 0] * thb[1, 1]
    thb[1, 0] = 0.0
    return thb


def _ctx(combus, sim_time=0.0, int_step=0.001):
    return SimContext(
        sim_time=sim_time,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=0,
    )


def _packet(name, ptype, **vars_):
    return Packet(name=name, type=ptype, status=1, vars=dict(vars_))


def _combus(target_vars=None, extra_targets=()):
    if target_vars is None:
        target_vars = {"SAEL": SAEL.copy(), "VAEL": VAEL.copy()}
    missile = _packet("m1", "MISSILE6")
    aircraft = _packet("blue", "AIRCRAFT3", SAEL=np.zeros(3), VAEL=np.zeros(3))
    target = _packet("decoy", "TARGET3", **target_vars)
    packets = [missile, aircraft, target]
    packets.extend(extra_targets)
    return packets


def _plant_truth(store, *, sbel=SBEL, vbel=VBEL, tbl=TBL, time=0.0):
    store.define(Field("time", time, "real", "exec", "kinematics"))
    store.define(Field("SBEL", sbel, "vec", "out", "newton"))
    store.define(Field("VBEL", vbel, "vec", "out", "newton"))
    store.define(Field("TBL", tbl, "mat", "out", "kinematics"))


def _ready(
    *,
    mseek=2,
    skr_dyn=0,
    tgt_num=1,
    racq=RACQ,
    dtimac=DTIMAC,
    combus=None,
    sim_time=0.0,
    sbel=SBEL,
    vbel=VBEL,
    tbl=TBL,
):
    vehicle = SimpleNamespace(store=StateStore())
    sensor = Agm6Sensor()
    sensor.define(vehicle)
    store = vehicle.store
    _plant_truth(store, sbel=sbel, vbel=vbel, tbl=tbl, time=sim_time)
    store.set("mseek", mseek)
    store.set("skr_dyn", skr_dyn)
    store.set("tgt_num", tgt_num)
    store.set("racq", racq)
    store.set("dtimac", dtimac)
    if combus is None:
        combus = _combus()
    sensor.initialize(vehicle, _ctx(combus, sim_time=sim_time))
    return vehicle, sensor, combus


def test_name_is_sensor():
    assert Agm6Sensor.name == "sensor"
    assert Agm6Sensor().name == "sensor"


def test_define_registers_cpp_def_sensor_plus_undeclared():
    vehicle = SimpleNamespace(store=StateStore())
    Agm6Sensor().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    zeros3 = np.zeros(3)
    zeros33 = np.zeros((3, 3))
    for name, (ftype, role, default, outputs) in FIELDS.items():
        field = store.field(name)
        assert field.module == "sensor"
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
        assert name not in DEFINED


def test_acquire_lock_hold_and_off_modes():
    vehicle, sensor, combus = _ready(mseek=2)
    store = vehicle.store
    sbtl = SBEL - SAEL
    dbtk = float(np.linalg.norm(sbtl))
    assert dbtk < RACQ

    sensor.execute(vehicle, _ctx(combus, sim_time=0.0))
    np.testing.assert_allclose(store.get("STEL"), SAEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VTEL"), VAEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("dbtk"), dbtk, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBTL"), sbtl, rtol=RTOL, atol=ATOL)
    assert store.get("mseek") == 3
    assert store.get("tgt_com_slot") == 2
    assert store.get("isets1") == 0
    np.testing.assert_allclose(store.get("epchac"), 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("timeac"), 0.0, rtol=RTOL, atol=ATOL)
    assert store.get("sigdpy") == 0.0
    assert store.get("sigdpz") == 0.0

    store.set("time", 0.31)
    sensor.execute(vehicle, _ctx(combus, sim_time=0.31))
    assert store.get("mseek") == 4
    np.testing.assert_allclose(store.get("timeac"), 0.31, rtol=RTOL, atol=ATOL)
    thtpb, psipb, sigdy, sigdz, dvbtc = _sensor_ir_kin(
        store.get("SBTL"), store.get("VTEL"), store.get("dbtk"), TBL, VBEL
    )
    np.testing.assert_allclose(store.get("thtpb"), thtpb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("psipb"), psipb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("sigdpy"), sigdy, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("sigdpz"), sigdz, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("dvbtc"), dvbtc, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("thtpbx"), thtpb * DEG, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("psipbx"), psipb * DEG, rtol=RTOL, atol=ATOL)

    vehicle0, sensor0, combus0 = _ready(mseek=0)
    store0 = vehicle0.store
    sensor0.execute(vehicle0, _ctx(combus0))
    assert store0.get("mseek") == 0
    np.testing.assert_allclose(store0.get("STEL"), SAEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store0.get("VTEL"), VAEL, rtol=RTOL, atol=ATOL)

    vehicle5, sensor5, combus5 = _ready(mseek=5)
    store5 = vehicle5.store
    sensor5.execute(vehicle5, _ctx(combus5))
    assert store5.get("mseek") == 5
    np.testing.assert_allclose(store5.get("STEL"), SAEL, rtol=RTOL, atol=ATOL)

    vehicle1, sensor1, combus1 = _ready(mseek=1)
    with pytest.raises(ValueError):
        sensor1.execute(vehicle1, _ctx(combus1))


def test_mseek_2_outside_racq_stays_enabled():
    combus = _combus({"SAEL": SAEL_FAR.copy(), "VAEL": VAEL.copy()})
    vehicle, sensor, _ = _ready(mseek=2, combus=combus)
    store = vehicle.store
    dbtk = float(np.linalg.norm(SBEL - SAEL_FAR))
    assert dbtk > RACQ
    sensor.execute(vehicle, _ctx(combus))
    assert store.get("mseek") == 2
    assert store.get("isets1") == 1
    np.testing.assert_allclose(store.get("dbtk"), dbtk, rtol=RTOL, atol=ATOL)


def test_skr_dyn_1_raises_until_dynamic_iir():
    vehicle, sensor, combus = _ready(mseek=2, skr_dyn=1)
    with pytest.raises(ValueError):
        sensor.execute(vehicle, _ctx(combus))


def test_tgt_num_selects_target3_by_combus_order_not_cadac_id():
    first = _packet("t9", "TARGET3", SAEL=SAEL.copy(), VAEL=VAEL.copy())
    second = _packet("t1", "TARGET3", SAEL=SAEL_TWO.copy(), VAEL=VAEL_TWO.copy())
    combus = [_packet("m1", "MISSILE6"), first, _packet("blue", "AIRCRAFT3"), second]
    vehicle, sensor, _ = _ready(mseek=0, tgt_num=2, combus=combus)
    store = vehicle.store
    sensor.execute(vehicle, _ctx(combus))
    np.testing.assert_allclose(store.get("STEL"), SAEL_TWO, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VTEL"), VAEL_TWO, rtol=RTOL, atol=ATOL)
    assert store.get("tgt_com_slot") == 3
    assert store.get("mseek") == 0


def test_sbel_vbel_fallback_when_sael_vael_missing():
    combus = _combus({"SBEL": SAEL.copy(), "VBEL": VAEL.copy()})
    vehicle, sensor, _ = _ready(mseek=0, combus=combus)
    store = vehicle.store
    sensor.execute(vehicle, _ctx(combus))
    np.testing.assert_allclose(store.get("STEL"), SAEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VTEL"), VAEL, rtol=RTOL, atol=ATOL)


def test_absent_fovlimx_and_irs_names_are_skipped():
    vehicle, sensor, combus = _ready(mseek=2)
    store = vehicle.store
    for name in ("fovlimx", "racq_irs", "fovyawx_irs", "fovpitchx_irs"):
        assert name not in store.names()
    sensor.execute(vehicle, _ctx(combus))
    assert store.get("mseek") == 3


def test_acquire_init_writes_thb_from_kinematic_pointing():
    vehicle, sensor, combus = _ready(mseek=2)
    store = vehicle.store
    sensor.execute(vehicle, _ctx(combus, sim_time=0.0))
    sbtl = store.get("SBTL")
    dbtk = store.get("dbtk")
    thtpb, psipb, _, _, _ = _sensor_ir_kin(sbtl, VAEL, dbtk, TBL, VBEL)
    ththb, phihb = _sensor_ir_uthpb(psipb, thtpb)
    want_thb = _sensor_ir_thb(ththb, phihb)
    np.testing.assert_allclose(store.get("THB"), want_thb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("thtpb"), thtpb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("psipb"), psipb, rtol=RTOL, atol=ATOL)
