from math import acos, atan2, cos, sin, sqrt, tan

import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr, polar_from_cart
from cadac.vehicles.flat6.sraam6.seeker import SMALL, Sraam6Seeker

RTOL = 1e-12
ATOL = 1e-14

SBEL = np.array([0.0, 0.0, -5000.0])
SAEL = np.array([10000.0, 500.0, -2000.0])
DVAE = 250.0
PSIALX = 180.0
THTALX = 0.0
DVBE = 250.0
DTIMAC = 0.250

COM_FIELDS = (
    "tgt_num",
    "STEL",
    "VTEL",
    "TTL",
    "tgt_com_slot",
    "sht_num",
    "SSEL",
    "dts",
    "STSL",
)
SEEKER_FIELDS = (
    "mseek",
    "ms1dyn",
    "isets1",
    "epchac",
    "ibreak",
    "dblind",
    "racq",
    "dtimac",
    "dbt",
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
    "fovyaw",
    "fovpitch",
    "dba",
    "daim",
    "BIASAI",
    "BIASSC",
    "RANDSC",
    "epy",
    "epz",
    "thtpb",
    "psipb",
    "ththb",
    "phihb",
    "ththbx",
    "phihbx",
    "TPB",
    "THB",
    "timeac",
    "psiot1",
    "thtot1",
    "dvbtc",
    "EAHH",
    "EPHH",
    "EAPH",
    "thtpbx",
    "psipbx",
    "sigdpy",
    "sigdpz",
    "biaseh",
    "randeh",
    "psihlx",
    "ththlx",
    "phihlx",
    "SBTL",
)
# S2 AI radar fields registered via Sraam6Seeker.define → Sraam6AiRadar.define
AI_RADAR_FIELDS = (
    "ntag",
    "dtimtu",
    "dtimup",
    "biastd",
    "randtd",
    "biasta",
    "randta",
    "biaste",
    "randte",
    "EVT1EL",
    "ST1CEL",
    "VT1CEL",
    "ai_iset1",
    "ai_iset2",
    "epchtai",
    "epchup",
    "mnav",
)
DEFINED = COM_FIELDS + SEEKER_FIELDS + AI_RADAR_FIELDS
INT_COM = ("tgt_num", "tgt_com_slot", "sht_num")
INT_SEEKER = ("mseek", "ms1dyn", "isets1", "ibreak")
VEC_COM = ("STEL", "VTEL", "SSEL", "STSL")
MAT_ZERO = ("TPB", "THB")
BIASAI_DEFAULT = (1.0, 0.5, 0.2)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _vael(dvae=DVAE, psialx=PSIALX, thtalx=THTALX):
    psial = psialx * RAD
    thtal = thtalx * RAD
    return np.array(
        [
            dvae * cos(thtal) * cos(psial),
            dvae * cos(thtal) * sin(psial),
            -dvae * sin(thtal),
        ]
    )


def _vbel():
    return np.array([DVBE, 0.0, 0.0])


def _dbt():
    sbtl = SBEL - SAEL
    return float(sqrt(float(sbtl @ sbtl)))


def _missile_packet():
    return Packet(name="m1", type="MISSILE6", status=1, vars={"SBEL": SBEL})


def _target_packet(sael=SAEL, vael=None, name="t1"):
    if vael is None:
        vael = _vael()
    return Packet(
        name=name,
        type="TARGET3",
        status=1,
        vars={"SAEL": np.asarray(sael, dtype=float), "VAEL": np.asarray(vael, dtype=float)},
    )


def _combus_1v1():
    return [_missile_packet(), _target_packet()]


def _ctx(combus=None, sim_time=0.0, int_step=0.001, vehicle_slot=0):
    if combus is None:
        combus = _combus_1v1()
    return SimContext(
        sim_time=sim_time,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=vehicle_slot,
    )


def _plant_kinematics(store, sbel=SBEL, vbel=None, tbl=None):
    if vbel is None:
        vbel = _vbel()
    if tbl is None:
        tbl = np.eye(3)
    store.define(Field("SBEL", sbel, "vec", "state", "newton"))
    store.define(Field("VBEL", vbel, "vec", "state", "newton"))
    store.define(Field("TBL", tbl, "mat", "out", "kinematics"))
    store.define(Field("WBEB", (0.0, 0.0, 0.0), "vec", "diag", "euler"))
    store.define(Field("mguid", 0, "int", "data", "guidance"))
    store.define(Field("trcond", 0, "int", "diag", "aerodynamics"))
    store.define(Field("trtht", 1.0, "real", "data", "aerodynamics"))
    store.define(Field("trthtd", 10.0, "real", "data", "aerodynamics"))
    store.define(Field("trphid", 14.0, "real", "data", "aerodynamics"))
    store.define(Field("trate", 1.0, "real", "data", "aerodynamics"))


def _ready(
    *,
    tgt_num=1,
    sht_num=0,
    mseek=2,
    ms1dyn=0,
    racq=20000.0,
    dtimac=DTIMAC,
    isets1=0,
    epchac=0.0,
    fovyaw=0.03140,
    fovpitch=0.03140,
    mguid=0,
    combus=None,
    sim_time=0.0,
    plant=True,
):
    vehicle = _Vehicle()
    seeker = Sraam6Seeker()
    seeker.define(vehicle)
    if plant:
        _plant_kinematics(vehicle.store)
        vehicle.store.set("mguid", mguid)
    store = vehicle.store
    store.set("tgt_num", tgt_num)
    store.set("sht_num", sht_num)
    store.set("mseek", mseek)
    store.set("ms1dyn", ms1dyn)
    store.set("racq", racq)
    store.set("dtimac", dtimac)
    store.set("isets1", isets1)
    store.set("epchac", epchac)
    store.set("fovyaw", fovyaw)
    store.set("fovpitch", fovpitch)
    return vehicle, seeker, _ctx(combus, sim_time=sim_time)


def test_name_is_seeker():
    assert Sraam6Seeker().name == "seeker"


def test_small_is_module_level_1e7():
    assert SMALL == 1e-7


def test_define_registers_def_seeker_fields():
    vehicle = _Vehicle()
    Sraam6Seeker().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    for name in INT_COM:
        assert store.field(name).module == "combus"
        assert store.field(name).type == "int"
        assert store.get(name) == 0
    assert store.field("tgt_num").role == "data"
    assert store.field("sht_num").role == "data"
    for name in VEC_COM:
        np.testing.assert_allclose(store.get(name), np.zeros(3), rtol=RTOL, atol=ATOL)
        assert store.field(name).type == "vec"
        assert store.field(name).module == "combus"
    np.testing.assert_allclose(store.get("TTL"), np.eye(3), rtol=RTOL, atol=ATOL)
    assert store.field("TTL").type == "mat"
    assert store.field("TTL").module == "combus"
    assert store.field("dts").outputs == ("scrn", "plot")
    assert store.field("mseek").module == "seeker"
    assert store.field("mseek").role == "data/diag"
    assert store.field("mseek").outputs == ("com",)
    for name in INT_SEEKER:
        assert store.get(name) == 0
        assert store.field(name).type == "int"
    np.testing.assert_allclose(store.get("BIASAI"), BIASAI_DEFAULT, rtol=RTOL, atol=ATOL)
    for name in MAT_ZERO:
        np.testing.assert_allclose(store.get(name), np.zeros((3, 3)), rtol=RTOL, atol=ATOL)
        assert store.field(name).type == "mat"
        assert store.field(name).role == "init"
    assert store.field("ththbx").outputs == ("scrn",)
    assert store.field("thtpbx").outputs == ("scrn", "plot")
    assert store.field("dbt").outputs == ("scrn", "plot")
    for name in ("mguid", "SBEL", "VBEL", "TBL", "time"):
        assert name not in store.names()


def test_1v1_racq_20000_enable_enters_acquisition():
    dbt = _dbt()
    assert dbt > 7000.0
    assert dbt < 20000.0
    vehicle, seeker, ctx = _ready(tgt_num=1, mseek=2, racq=20000.0, ms1dyn=0)
    seeker.execute(vehicle, ctx)
    store = vehicle.store
    assert store.get("mseek") == 3
    np.testing.assert_allclose(store.get("STEL"), SAEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VTEL"), _vael(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("dbt"), dbt, rtol=RTOL, atol=ATOL)
    assert store.get("tgt_com_slot") == 1
    assert np.isfinite(store.get("thtpb"))
    assert np.isfinite(store.get("psipb"))


def test_racq_1_stays_enabled():
    vehicle, seeker, ctx = _ready(tgt_num=1, mseek=2, racq=1.0, ms1dyn=0)
    seeker.execute(vehicle, ctx)
    assert vehicle.store.get("mseek") == 2
    np.testing.assert_allclose(vehicle.store.get("STEL"), SAEL, rtol=RTOL, atol=ATOL)
    assert vehicle.store.get("dbt") > 1.0


def test_mseek_0_geometry_only_does_not_set_3():
    vehicle, seeker, ctx = _ready(tgt_num=1, mseek=0, racq=20000.0)
    seeker.execute(vehicle, ctx)
    store = vehicle.store
    assert store.get("mseek") == 0
    np.testing.assert_allclose(store.get("STEL"), SAEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VTEL"), _vael(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("dbt"), _dbt(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("thtpb"), 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("sigdpy"), 0.0, rtol=RTOL, atol=ATOL)


def test_mseek_1_raises():
    vehicle, seeker, ctx = _ready(mseek=1, racq=20000.0)
    with pytest.raises(ValueError):
        seeker.execute(vehicle, ctx)


def test_unknown_mseek_raises():
    vehicle, seeker, ctx = _ready(mseek=9, racq=20000.0)
    with pytest.raises(ValueError):
        seeker.execute(vehicle, ctx)


def test_seeker_kin_sigdy_thtpb_finite():
    vehicle, seeker, ctx = _ready(tgt_num=1, mseek=2, racq=20000.0, ms1dyn=0)
    seeker.execute(vehicle, ctx)
    store = vehicle.store
    sbtl = store.get("SBTL")
    vtel = store.get("VTEL")
    dbt = store.get("dbt")
    thtpb, psipb, sigdy, sigdz = seeker.seeker_kin(vehicle, sbtl, vtel, dbt)
    assert np.isfinite(sigdy)
    assert np.isfinite(thtpb)
    assert np.isfinite(psipb)
    assert np.isfinite(sigdz)
    assert np.isfinite(store.get("dvbtc"))
    sbel = store.get("SBEL")
    stel = store.get("STEL")
    stbl = stel - sbel
    stbb = store.get("TBL") @ stbl
    polar = polar_from_cart(stbb)
    assert thtpb == pytest.approx(float(polar[2]), rel=RTOL, abs=ATOL)
    assert psipb == pytest.approx(float(polar[1]), rel=RTOL, abs=ATOL)


def test_tgt_num_2_with_one_target_leaves_stel_zeros():
    vehicle, seeker, ctx = _ready(tgt_num=2, mseek=0, racq=20000.0)
    seeker.execute(vehicle, ctx)
    np.testing.assert_allclose(vehicle.store.get("STEL"), np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("VTEL"), np.zeros(3), rtol=RTOL, atol=ATOL)
    assert vehicle.store.get("tgt_com_slot") == 0


def test_identifies_target_by_type_not_name():
    combus = [
        Packet(name="t1", type="MISSILE6", status=1, vars={"SAEL": np.zeros(3), "VAEL": np.zeros(3)}),
        _target_packet(name="not-t1"),
    ]
    vehicle, seeker, ctx = _ready(tgt_num=1, mseek=2, racq=20000.0, combus=combus)
    seeker.execute(vehicle, ctx)
    np.testing.assert_allclose(vehicle.store.get("STEL"), SAEL, rtol=RTOL, atol=ATOL)
    assert vehicle.store.get("tgt_com_slot") == 1


def test_sht_num_0_leaves_ssel_zeros():
    vehicle, seeker, ctx = _ready(tgt_num=1, sht_num=0, mseek=0)
    seeker.execute(vehicle, ctx)
    np.testing.assert_allclose(vehicle.store.get("SSEL"), np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("STSL"), SAEL, rtol=RTOL, atol=ATOL)
    dts = float(np.linalg.norm(SAEL))
    assert vehicle.store.get("dts") == pytest.approx(dts, rel=RTOL, abs=ATOL)


def test_sht_num_1_downloads_shooter_sael():
    shooter = np.array([1.0, 2.0, -3.0])
    combus = [
        _missile_packet(),
        _target_packet(),
        _target_packet(sael=shooter, vael=np.zeros(3), name="t2"),
    ]
    vehicle, seeker, ctx = _ready(tgt_num=1, sht_num=2, mseek=0, combus=combus)
    seeker.execute(vehicle, ctx)
    np.testing.assert_allclose(vehicle.store.get("SSEL"), shooter, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("STEL"), SAEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("STSL"), SAEL - shooter, rtol=RTOL, atol=ATOL)


def test_ttl_stays_identity():
    vehicle, seeker, ctx = _ready(mseek=0)
    seeker.execute(vehicle, ctx)
    np.testing.assert_allclose(vehicle.store.get("TTL"), np.eye(3), rtol=RTOL, atol=ATOL)


def test_ms1dyn_invalid_raises():
    vehicle, seeker, ctx = _ready(mseek=3, ms1dyn=2, racq=20000.0, isets1=1)
    with pytest.raises(ValueError):
        seeker.execute(vehicle, ctx)


def test_mseek_3_timeac_beyond_dtimac_locks_and_sets_mguid_6():
    vehicle, seeker, ctx = _ready(
        mseek=3,
        ms1dyn=0,
        racq=20000.0,
        isets1=0,
        epchac=0.0,
        dtimac=DTIMAC,
        sim_time=0.251,
    )
    seeker.execute(vehicle, ctx)
    store = vehicle.store
    assert store.get("mseek") == 4
    assert store.get("mguid") == 6
    assert store.get("timeac") == pytest.approx(0.251, rel=RTOL, abs=ATOL)
    assert np.isfinite(store.get("sigdpy"))
    assert np.isfinite(store.get("sigdpz"))
    assert np.isfinite(store.get("thtpb"))


def test_ms1dyn_1_stub_dyn_locks_when_timeac_elapsed():
    vehicle, seeker, ctx = _ready(
        mseek=3,
        ms1dyn=1,
        racq=20000.0,
        isets1=0,
        epchac=0.0,
        dtimac=DTIMAC,
        sim_time=0.251,
    )
    seeker.execute(vehicle, ctx)
    assert vehicle.store.get("mseek") == 4
    assert vehicle.store.get("mguid") == 6


def test_ms1dyn_1_dyn_uses_this_frame_vael_download():
    vael = _vael()
    vehicle, seeker, ctx = _ready(
        mseek=3,
        ms1dyn=1,
        racq=20000.0,
        isets1=0,
        epchac=0.0,
        dtimac=DTIMAC,
        sim_time=0.251,
    )
    vehicle.store.set("VTEL", np.array([999.0, 888.0, 777.0]))
    seeker.execute(vehicle, ctx)
    store = vehicle.store
    sbtl = SBEL - SAEL
    dbt = _dbt()
    _thtpb, _psipb, sigdy, _sigdz = seeker.seeker_kin(vehicle, sbtl, vael, dbt)
    stale = seeker.seeker_kin(
        vehicle, sbtl, np.array([999.0, 888.0, 777.0]), dbt
    )
    assert stale[2] != pytest.approx(sigdy, rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(store.get("VTEL"), vael, rtol=RTOL, atol=ATOL)
    assert np.isfinite(store.get("thtpb"))
    assert np.isfinite(store.get("sigdpy"))
    assert store.get("sigdpy") == pytest.approx(store.get("wlq1"), rel=RTOL, abs=ATOL)


def test_seeker_uthpb_copies_cpp_and_precedence():
    seeker = Sraam6Seeker()
    ththb, phihb = seeker.seeker_uthpb(0.0, 0.0)
    assert ththb == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert phihb == pytest.approx(0.0, rel=RTOL, abs=ATOL)

    psipb = 0.2
    thtpb = 1e-9
    ththb, phihb = seeker.seeker_uthpb(psipb, thtpb)
    assert phihb == 0.0
    two_abs = abs(sin(psipb)) < SMALL and abs(tan(thtpb)) < SMALL
    assert two_abs is False
    not_cpp = atan2(sin(psipb), tan(thtpb))
    assert abs(not_cpp) > 1.0
    assert ththb == pytest.approx(acos(cos(thtpb) * cos(psipb)), rel=RTOL, abs=ATOL)


def test_seeker_thb_matches_cpp_entries():
    seeker = Sraam6Seeker()
    tht = 0.3
    phi = 0.4
    thb = seeker.seeker_thb(tht, phi)
    want = np.zeros((3, 3))
    want[0, 0] = cos(tht)
    want[2, 0] = sin(tht)
    want[1, 1] = cos(phi)
    want[1, 2] = sin(phi)
    want[0, 1] = want[2, 0] * want[1, 2]
    want[0, 2] = -want[2, 0] * want[1, 1]
    want[2, 1] = -want[0, 0] * want[1, 2]
    want[2, 2] = want[0, 0] * want[1, 1]
    want[1, 0] = 0.0
    np.testing.assert_allclose(thb, want, rtol=RTOL, atol=ATOL)


def test_seeker_kin_woep_uses_mat2tr_and_local_skew():
    vehicle, seeker, _ctx_unused = _ready(mseek=0)
    sbtl = SBEL - SAEL
    vtel = _vael()
    dbt = float(np.linalg.norm(sbtl))
    thtpb, psipb, sigdy, sigdz = seeker.seeker_kin(vehicle, sbtl, vtel, dbt)
    stbl = -sbtl
    dum1 = 1.0 / dbt
    utbl = stbl * dum1
    vtbl = vtel - vehicle.store.get("VBEL")
    woeb = vehicle.store.get("TBL") @ (_skew_ref(utbl) @ vtbl) * dum1
    tpb = mat2tr(psipb, thtpb)
    woep = tpb @ woeb
    assert sigdy == pytest.approx(float(woep[1]), rel=RTOL, abs=ATOL)
    assert sigdz == pytest.approx(float(woep[2]), rel=RTOL, abs=ATOL)


def test_initialize_and_terminate_are_pass():
    vehicle, seeker, ctx = _ready(mseek=0)
    assert seeker.initialize(vehicle, ctx) is None
    assert seeker.terminate(vehicle, ctx) is None
    np.testing.assert_allclose(vehicle.store.get("STEL"), np.zeros(3), rtol=RTOL, atol=ATOL)


def _skew_ref(vec):
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )
