from math import acos, atan2, cos, fabs, sin, tan
from pathlib import Path

import numpy as np
import pytest

from cadac.constants import DEG
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr, polar_from_cart
from cadac.vehicles.sam6.sensor import KBOLTZ, SMALL, Sam6Sensor

RTOL = 1e-12
ATOL = 1e-14
DT = 0.001
RACQ_RF = 7000.0
RACQ_IR = 7000.0
ZEROS3 = (0.0, 0.0, 0.0)
ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

SBEL = np.array([0.0, 0.0, 0.0], dtype=float)
VBEL = np.array([16.0, 0.0, 0.0], dtype=float)
SAEL = np.array([1000.0, 0.0, 0.0], dtype=float)
VAEL = np.array([0.0, 250.0, 0.0], dtype=float)
WBECB = np.array([0.0, 0.0, 0.0], dtype=float)

DEFINED = (
    "STEL",
    "VTEL",
    "tgt_slot",
    "mseek",
    "skr_dyn",
    "isets1",
    "epchac",
    "fst_tgt_slot",
    "mtarget",
    "dbtk",
    "thtpb",
    "psipb",
    "SBTL",
    "thtpbx",
    "psipbx",
    "dblind",
    "ibreak",
    "racq_ir",
    "dtimac_ir",
    "ehz",
    "ehy",
    "trtht",
    "trthtd",
    "trphid",
    "trate",
    "gk",
    "zetak",
    "wnk",
    "biast",
    "randt",
    "biasp",
    "randp",
    "wlq1d",
    "wlq1",
    "wlqd",
    "wlq",
    "wlr1d",
    "wlr1",
    "wlrd",
    "wlr",
    "wlq2d",
    "wlq2",
    "wlr2d",
    "wlr2",
    "fovyaw_ir",
    "fovpitch_ir",
    "daim",
    "BIASAI",
    "BIASSC",
    "RANDSC",
    "epy",
    "dta",
    "epz",
    "ththb",
    "phihb",
    "TPB",
    "THB",
    "dvbtc",
    "EAHH",
    "EPHH",
    "EAPH",
    "sigdy",
    "sigdz",
    "biaseh",
    "randeh",
    "timeac",
    "racq_rf",
    "dtimac_rf",
    "temp_resx",
    "biasaz",
    "biasel",
    "freqghz",
    "rngegw",
    "thta_3db",
    "powrs",
    "gainsdb",
    "gainmdb",
    "tgt_rcs",
    "rlatmodb",
    "rltotldb",
    "dwltm",
    "rnoisfgdb",
    "plc5",
    "plc4",
    "plc3",
    "plc2",
    "plc1",
    "plc0",
    "biasgl1",
    "biasgl2",
    "biasgl3",
    "randgl1",
    "randgl2",
    "randgl3",
    "fovlim_rfx",
    "forlim_rfx",
    "gain_rf",
    "aztbx",
    "eltbx",
    "dab",
    "ddab",
    "pwr_loss_db",
    "snr_db",
    "onax",
    "epaz_rf_saved",
    "epel_rf_saved",
    "range_rf_saved",
    "rate_rf_saved",
    "epsic",
    "ethtc",
    "mepsit",
    "methtt",
    "psisb",
    "psisbd",
    "thtsb",
    "thtsbd",
    "lamdqb",
    "lamdrb",
)
ROLES = {
    "STEL": "out",
    "VTEL": "out",
    "tgt_slot": "out",
    "mseek": "data",
    "skr_dyn": "data",
    "isets1": "init",
    "epchac": "init",
    "fst_tgt_slot": "save",
    "mtarget": "data",
    "dbtk": "diag",
    "thtpb": "out",
    "psipb": "out",
    "SBTL": "diag",
    "thtpbx": "diag",
    "psipbx": "diag",
    "dblind": "data",
    "ibreak": "init",
    "racq_ir": "data",
    "dtimac_ir": "data",
    "ehz": "diag",
    "ehy": "diag",
    "trtht": "data",
    "trthtd": "data",
    "trphid": "data",
    "trate": "data",
    "gk": "data",
    "zetak": "data",
    "wnk": "data",
    "biast": "data",
    "randt": "data",
    "biasp": "data",
    "randp": "data",
    "wlq1d": "state",
    "wlq1": "state",
    "wlqd": "state",
    "wlq": "state",
    "wlr1d": "state",
    "wlr1": "state",
    "wlrd": "state",
    "wlr": "state",
    "wlq2d": "state",
    "wlq2": "state",
    "wlr2d": "state",
    "wlr2": "state",
    "fovyaw_ir": "data",
    "fovpitch_ir": "data",
    "daim": "data",
    "BIASAI": "data",
    "BIASSC": "data",
    "RANDSC": "data",
    "epy": "diag",
    "dta": "out",
    "epz": "diag",
    "ththb": "diag",
    "phihb": "diag",
    "TPB": "init",
    "THB": "init",
    "dvbtc": "diag",
    "EAHH": "diag",
    "EPHH": "diag",
    "EAPH": "diag",
    "sigdy": "out",
    "sigdz": "out",
    "biaseh": "data",
    "randeh": "data",
    "timeac": "save",
    "racq_rf": "data",
    "dtimac_rf": "data",
    "temp_resx": "data",
    "biasaz": "data",
    "biasel": "data",
    "freqghz": "data",
    "rngegw": "data",
    "thta_3db": "data",
    "powrs": "data",
    "gainsdb": "data",
    "gainmdb": "data",
    "tgt_rcs": "data",
    "rlatmodb": "data",
    "rltotldb": "data",
    "dwltm": "data",
    "rnoisfgdb": "data",
    "plc5": "data",
    "plc4": "data",
    "plc3": "data",
    "plc2": "data",
    "plc1": "data",
    "plc0": "data",
    "biasgl1": "data",
    "biasgl2": "data",
    "biasgl3": "data",
    "randgl1": "data",
    "randgl2": "data",
    "randgl3": "data",
    "fovlim_rfx": "data",
    "forlim_rfx": "data",
    "gain_rf": "data",
    "aztbx": "diag",
    "eltbx": "diag",
    "dab": "out",
    "ddab": "out",
    "pwr_loss_db": "diag",
    "snr_db": "diag",
    "onax": "diag",
    "epaz_rf_saved": "save",
    "epel_rf_saved": "save",
    "range_rf_saved": "save",
    "rate_rf_saved": "save",
    "epsic": "diag",
    "ethtc": "diag",
    "mepsit": "diag",
    "methtt": "diag",
    "psisb": "state",
    "psisbd": "state",
    "thtsb": "state",
    "thtsbd": "state",
    "lamdqb": "out",
    "lamdrb": "out",
}
OUTPUTS = {
    "dbtk": ("plot", "scrn"),
    "thtpbx": ("plot",),
    "psipbx": ("plot",),
    "epy": ("plot",),
    "dta": ("plot",),
    "epz": ("plot",),
    "sigdy": ("plot",),
    "sigdz": ("plot",),
    "aztbx": ("plot",),
    "eltbx": ("plot",),
    "dab": ("plot",),
    "ddab": ("plot",),
    "snr_db": ("plot",),
    "onax": ("plot",),
    "epsic": ("plot",),
    "ethtc": ("plot",),
    "psisb": ("plot",),
    "thtsb": ("plot",),
    "lamdqb": ("plot",),
    "lamdrb": ("plot",),
}
INT_FIELDS = (
    "tgt_slot",
    "mseek",
    "skr_dyn",
    "isets1",
    "fst_tgt_slot",
    "mtarget",
    "ibreak",
)
VEC_FIELDS = (
    "STEL",
    "VTEL",
    "SBTL",
    "BIASAI",
    "BIASSC",
    "RANDSC",
    "EAHH",
    "EPHH",
    "EAPH",
)
MAT_FIELDS = ("TPB", "THB")
NOT_DEFINED = (
    "time",
    "SBEL",
    "TBL",
    "VBEL",
    "WBECB",
    "trcond",
    "mguide",
    "stop",
    "TTL",
)

RF_RADIO = {
    "freqghz": 16.0,
    "rngegw": 15.0,
    "thta_3db": 7.0,
    "powrs": 500.0,
    "gainsdb": 26.0,
    "gainmdb": 26.0,
    "tgt_rcs": 2.0,
    "rltotldb": 7.0,
    "dwltm": 0.005,
    "plc5": 1.0,
    "plc4": 4.7,
    "plc3": 8.2,
    "plc2": 6.9,
    "plc1": 3.0,
    "plc0": 1.0,
    "gain_rf": 5.0,
    "forlim_rfx": 40.0,
    "fovlim_rfx": 10.0,
    "dtimac_rf": 0.1,
}


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


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


def _ctx(int_step=DT, combus=None, vehicle_slot=0, sim_time=0.0):
    return SimContext(
        sim_time=sim_time,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=vehicle_slot,
    )


def _missile_packet(name="SAM1"):
    return Packet(name=name, type="MISSILE6", status=1, vars={})


def _aircraft_packet(sael=SAEL, vael=VAEL, dta=0.0, name="AC1"):
    return Packet(
        name=name,
        type="AIRCRAFT3",
        status=1,
        vars={"SAEL": np.asarray(sael, dtype=float), "VAEL": np.asarray(vael, dtype=float), "dta": dta},
    )


def _rocket_packet(sael=SAEL, vael=VAEL, dta=0.0, name="SRBM"):
    return Packet(
        name=name,
        type="ROCKET5",
        status=1,
        vars={"SAEL": np.asarray(sael, dtype=float), "VAEL": np.asarray(vael, dtype=float), "dta": dta},
    )


def _plant_missile(store, **overrides):
    tbl = np.array(IDENTITY, dtype=float)
    for name, value, ftype, role, module in (
        ("time", 0.0, "real", "exec", "kinematics"),
        ("SBEL", SBEL, "vec", "state", "newton"),
        ("TBL", tbl, "mat", "out", "kinematics"),
        ("VBEL", VBEL, "vec", "out", "newton"),
        ("WBECB", WBECB, "vec", "out", "ins"),
        ("trcond", 0, "int", "diag", "aerodynamics"),
        ("mguide", 0, "int", "data", "guidance"),
    ):
        if name not in store.names():
            store.define(Field(name, value, ftype, role, module))
        store.set(name, value)
    for name, value in overrides.items():
        store.set(name, value)


def _defined():
    vehicle = _Vehicle()
    sensor = Sam6Sensor()
    sensor.define(vehicle)
    return vehicle, sensor


def _ready(mseek=12, mtarget=2, skr_dyn=0, combus=None, vehicle_slot=0, **plant):
    vehicle, sensor = _defined()
    store = vehicle.store
    store.set("mseek", mseek)
    store.set("mtarget", mtarget)
    store.set("skr_dyn", skr_dyn)
    store.set("racq_rf", RACQ_RF)
    store.set("racq_ir", RACQ_IR)
    _plant_missile(store, **plant)
    if combus is None:
        combus = [_missile_packet(), _aircraft_packet()]
    return vehicle, sensor, _ctx(combus=combus, vehicle_slot=vehicle_slot)


def _sensor_kin(sbtl, vtel, dbtk, tbl, vbel):
    stbl = sbtl * (-1.0)
    stbb = tbl @ stbl
    utbl = stbl / dbtk
    vtbl = vtel - vbel
    ddab = -fabs(float(utbl @ vtbl))
    woeb = tbl @ _skew(utbl) @ vtbl / dbtk
    lamdqb = float(woeb[1])
    lamdrb = float(woeb[2])
    polar = polar_from_cart(stbb)
    psipb = float(polar[1])
    thtpb = float(polar[2])
    tpb = mat2tr(psipb, thtpb)
    woep = tpb @ woeb
    return thtpb, psipb, float(woep[1]), float(woep[2]), lamdrb, lamdqb, ddab


def _uthpb(psipb, thtpb):
    ththb = acos(cos(thtpb) * cos(psipb))
    sinpsi = sin(psipb)
    tantht = tan(thtpb)
    if fabs(sinpsi) and fabs(tantht) < SMALL:
        phihb = 0.0
    else:
        phihb = atan2(sinpsi, tantht)
    return ththb, phihb


def test_name_is_sensor():
    assert Sam6Sensor.name == "sensor"


def test_define_registers_cpp_fields():
    vehicle, _ = _defined()
    names = vehicle.store.names()
    for name in DEFINED:
        assert name in names
    for name in DEFINED:
        field = vehicle.store.field(name)
        assert field.role == ROLES[name]
        assert field.module == "sensor"
        assert field.outputs == OUTPUTS.get(name, ())
        if name in INT_FIELDS:
            assert field.type == "int"
        elif name in VEC_FIELDS:
            assert field.type == "vec"
        elif name in MAT_FIELDS:
            assert field.type == "mat"
        else:
            assert field.type == "real"
    assert _approx(vehicle.store.get("temp_resx"), 290.0)


def test_define_does_not_register_kinematics_ins_or_guidance():
    vehicle, _ = _defined()
    for name in NOT_DEFINED:
        assert name not in vehicle.store.names()


def test_mseek_0_returns_without_raise_or_write():
    vehicle, sensor, ctx = _ready(mseek=0, mtarget=0)
    store = vehicle.store
    store.set("dbtk", 999.0)
    store.set("mseek", 0)
    sensor.execute(vehicle, ctx)
    assert store.get("mseek") == 0
    assert _approx(store.get("dbtk"), 999.0)


def test_mtarget_0_raises():
    vehicle, sensor, ctx = _ready(mseek=12, mtarget=0)
    with pytest.raises(ValueError, match="mtarget"):
        sensor.execute(vehicle, ctx)


def test_mseek_32_skr_type_3_raises():
    vehicle, sensor, ctx = _ready(mseek=32, mtarget=2)
    with pytest.raises(ValueError, match="skr_type"):
        sensor.execute(vehicle, ctx)


def test_rf_mode2_acquire_inside_racq_rf():
    vehicle, sensor, ctx = _ready(mseek=12, mtarget=2, skr_dyn=0)
    sensor.execute(vehicle, ctx)
    mseek = vehicle.store.get("mseek")
    assert mseek // 10 == 1
    assert mseek % 10 in (3, 4)
    assert _approx(vehicle.store.get("dbtk"), 1000.0)
    np.testing.assert_allclose(vehicle.store.get("STEL"), SAEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("VTEL"), VAEL, rtol=RTOL, atol=ATOL)
    assert vehicle.store.get("tgt_slot") == 1


def test_rf_kinematic_vs_cpp_replica():
    vehicle, sensor, ctx = _ready(mseek=12, mtarget=2, skr_dyn=0)
    sensor.execute(vehicle, ctx)
    sbtl = SBEL - SAEL
    dbtk = float(np.linalg.norm(sbtl))
    thtpb, psipb, sigdy, sigdz, lamdrb, lamdqb, ddab = _sensor_kin(
        sbtl, VAEL, dbtk, np.array(IDENTITY, dtype=float), VBEL
    )
    store = vehicle.store
    assert store.get("mseek") == 13
    assert _approx(store.get("dbtk"), dbtk)
    assert _approx(store.get("thtpb"), thtpb)
    assert _approx(store.get("psipb"), psipb)
    assert _approx(store.get("sigdy"), sigdy)
    assert _approx(store.get("sigdz"), sigdz)
    assert _approx(store.get("lamdqb"), lamdqb)
    assert _approx(store.get("lamdrb"), lamdrb)
    assert _approx(store.get("ddab"), ddab)
    assert _approx(store.get("thtpbx"), thtpb * DEG)
    assert _approx(store.get("psipbx"), psipb * DEG)
    np.testing.assert_allclose(store.get("SBTL"), sbtl, rtol=RTOL, atol=ATOL)
    assert _approx(store.get("psisb"), psipb)
    assert _approx(store.get("thtsb"), thtpb)


def test_rf_lock_after_dtimac_rf():
    vehicle, sensor, ctx = _ready(mseek=12, mtarget=2, skr_dyn=0)
    vehicle.store.set("dtimac_rf", 0.1)
    sensor.execute(vehicle, ctx)
    assert vehicle.store.get("mseek") % 10 == 3
    vehicle.store.set("time", 0.2)
    sensor.execute(vehicle, _ctx(combus=ctx.combus, vehicle_slot=0, sim_time=0.2))
    assert vehicle.store.get("mseek") == 14


def test_mtarget_1_pairs_rocket5():
    combus = [_missile_packet(), _rocket_packet(dta=42.0)]
    vehicle, sensor, ctx = _ready(mseek=12, mtarget=1, skr_dyn=0, combus=combus)
    sensor.execute(vehicle, ctx)
    assert vehicle.store.get("mseek") % 10 in (3, 4)
    np.testing.assert_allclose(vehicle.store.get("STEL"), SAEL, rtol=RTOL, atol=ATOL)
    assert _approx(vehicle.store.get("dta"), 42.0)
    assert vehicle.store.get("tgt_slot") == 1


def test_kth_missile_pairs_kth_target_not_combus_slot():
    far = np.array([20000.0, 0.0, 0.0], dtype=float)
    near = np.array([500.0, 0.0, 0.0], dtype=float)
    combus = [
        _aircraft_packet(sael=far, name="AC_far"),
        _missile_packet("SAM1"),
        _aircraft_packet(sael=near, name="AC_near"),
        _missile_packet("SAM2"),
    ]
    vehicle, sensor, ctx = _ready(
        mseek=12, mtarget=2, skr_dyn=0, combus=combus, vehicle_slot=3
    )
    sensor.execute(vehicle, ctx)
    np.testing.assert_allclose(vehicle.store.get("STEL"), near, rtol=RTOL, atol=ATOL)
    assert vehicle.store.get("tgt_slot") == 2
    assert _approx(vehicle.store.get("dbtk"), 500.0)


def test_pairs_by_packet_type_not_cpp_id():
    combus = [
        Packet(
            name="r1",
            type="AIRCRAFT3",
            status=1,
            vars={"SAEL": SAEL, "VAEL": VAEL, "dta": 0.0},
        ),
        Packet(name="a1", type="MISSILE6", status=1, vars={}),
    ]
    vehicle, sensor, ctx = _ready(
        mseek=12, mtarget=2, skr_dyn=0, combus=combus, vehicle_slot=1
    )
    sensor.execute(vehicle, ctx)
    np.testing.assert_allclose(vehicle.store.get("STEL"), SAEL, rtol=RTOL, atol=ATOL)
    assert vehicle.store.get("tgt_slot") == 0


def test_ir_mode2_acquire_inside_racq_ir():
    combus = [_missile_packet(), _rocket_packet()]
    vehicle, sensor, ctx = _ready(mseek=22, mtarget=1, skr_dyn=0, combus=combus)
    sensor.execute(vehicle, ctx)
    mseek = vehicle.store.get("mseek")
    assert mseek // 10 == 2
    assert mseek % 10 in (3, 4)


def _ir_thb(tht, phi):
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


def test_ir_kinematic_thb_vs_cpp():
    combus = [_missile_packet(), _rocket_packet()]
    vehicle, sensor, ctx = _ready(mseek=22, mtarget=1, skr_dyn=0, combus=combus)
    sensor.execute(vehicle, ctx)
    sbtl = SBEL - SAEL
    dbtk = float(np.linalg.norm(sbtl))
    thtpb, psipb, *_ = _sensor_kin(
        sbtl, VAEL, dbtk, np.array(IDENTITY, dtype=float), VBEL
    )
    ththb, phihb = _uthpb(psipb, thtpb)
    store = vehicle.store
    assert store.get("mseek") == 23
    assert _approx(store.get("thtpb"), thtpb)
    assert _approx(store.get("psipb"), psipb)
    np.testing.assert_allclose(
        store.get("THB"), _ir_thb(ththb, phihb), rtol=RTOL, atol=ATOL
    )


def test_skr_dyn_1_rf_bias_random_glint_zero_acquires():
    vehicle, sensor, ctx = _ready(mseek=12, mtarget=2, skr_dyn=1)
    store = vehicle.store
    for name, value in RF_RADIO.items():
        store.set(name, value)
    store.set("biasaz", 0.0)
    store.set("biasel", 0.0)
    store.set("biasgl1", 0.0)
    store.set("randgl1", 0.0)
    sensor.execute(vehicle, ctx)
    assert store.get("mseek") % 10 in (3, 4)
    assert np.isfinite(store.get("lamdqb"))
    assert np.isfinite(store.get("lamdrb"))
    assert np.isfinite(store.get("ddab"))


def test_skr_dyn_1_rf_lock_when_in_for():
    vehicle, sensor, ctx = _ready(mseek=12, mtarget=2, skr_dyn=1)
    store = vehicle.store
    for name, value in RF_RADIO.items():
        store.set(name, value)
    sensor.execute(vehicle, ctx)
    assert store.get("mseek") % 10 == 3
    store.set("time", 0.2)
    sensor.execute(vehicle, _ctx(combus=ctx.combus, vehicle_slot=0, sim_time=0.2))
    assert store.get("mseek") == 14
    assert fabs(store.get("aztbx")) <= store.get("forlim_rfx")
    assert fabs(store.get("eltbx")) <= store.get("forlim_rfx")


def test_rf_dyn_glint_zero_range_matches_dbtk():
    vehicle, sensor, ctx = _ready(mseek=12, mtarget=2, skr_dyn=1)
    store = vehicle.store
    for name, value in RF_RADIO.items():
        store.set(name, value)
    sensor.execute(vehicle, ctx)
    dbtk = store.get("dbtk")
    assert _approx(store.get("dab"), dbtk)


def test_cadac_sign_zero_is_plus_one():
    from cadac.vehicles.sam6.sensor import _sign

    assert _sign(0.0) == 1
    assert _sign(-0.1) == -1
    assert _sign(0.1) == 1


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.sam6.sensor as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "Plane5" not in src
    assert "Plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src


def test_initialize_and_terminate_are_pass():
    vehicle, sensor, ctx = _ready()
    assert sensor.initialize(vehicle, ctx) is None
    assert sensor.terminate(vehicle, ctx) is None
    sensor.execute(vehicle, ctx)
    assert vehicle.store.get("mseek") % 10 in (3, 4)


def test_kboltz_and_small_match_cadac():
    assert _approx(KBOLTZ, 1.38e-23)
    assert _approx(SMALL, 1e-7)
