from math import atan2, cos, sin, sqrt, tan
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import AGRAV, DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat3tr, polar_from_cart
from cadac.vehicles.agm6.guidance import Agm6Guidance

RTOL = 1e-12
ATOL = 1e-14
DT = 0.001
SMALL = 1.e-7
PLOT = ("plot",)
SCRN_PLOT = ("scrn", "plot")
ZEROS3 = (0.0, 0.0, 0.0)

# C++ Missile::def_guidance order.
FIELDS = {
    "mguid": ("int", "data", 0, ()),
    "gnav": ("real", "data", 0.0, ()),
    "ancomx": ("real", "out", 0.0, SCRN_PLOT),
    "alcomx": ("real", "out", 0.0, SCRN_PLOT),
    "gn": ("real", "diag", 0.0, ()),
    "grav_bias": ("real", "data", 0.0, ()),
    "apny": ("real", "diag", 0.0, ()),
    "apnz": ("real", "diag", 0.0, ()),
    "adely": ("real", "diag", 0.0, ()),
    "adelz": ("real", "diag", 0.0, ()),
    "all": ("real", "diag", 0.0, PLOT),
    "ann": ("real", "diag", 0.0, PLOT),
    "epchta": ("real", "save", 0.0, ()),
    "tgo_tgt_acrft": ("real", "diag", 0.0, ()),
    "line_gain": ("real", "data", 0.0, ()),
    "nl_gain_fact": ("real", "data", 0.0, ()),
    "decrement": ("real", "data", 0.0, ()),
    "dtac": ("real", "diag", 0.0, ()),
    "VBEO": ("vec", "diag", ZEROS3, ()),
    "VBEF": ("vec", "diag", ZEROS3, ()),
    "init_guide_line": ("int", "init", 1, ()),
    "thtflx": ("real", "data", 0.0, ()),
    "SBTO": ("vec", "diag", ZEROS3, ()),
    "quad_pos": ("real", "data", 0.0, ()),
    "quad_vel": ("real", "data", 0.0, ()),
    "w1": ("real", "data", 0.0, ()),
    "w3": ("real", "data", 0.0, ()),
    "WOELC": ("vec", "out", ZEROS3, ()),
    "tgoc": ("real", "diag", 0.0, ()),
    "dtbc": ("real", "diag", 0.0, ()),
    "dvtbc": ("real", "diag", 0.0, ()),
    "psiobcx": ("real", "diag", 0.0, PLOT),
    "thtobcx": ("real", "diag", 0.0, PLOT),
    "UTBLC": ("vec", "out", ZEROS3, ()),
    "STELC": ("vec", "diag", ZEROS3, ()),
    "STBLC": ("vec", "diag", ZEROS3, ()),
    "STELM": ("vec", "save", ZEROS3, ()),
    "VTELC": ("vec", "save", ZEROS3, ()),
}
DEFINED = tuple(FIELDS)

EXTERNALS = (
    "mnav",
    "STCEL",
    "VTCEL",
    "SAEL",
    "VAEL",
    "TBLC",
    "VBELC",
    "SBELC",
    "FSPCB",
    "STEL",
    "VTEL",
    "psipb",
    "thtpb",
    "sigdpy",
    "sigdpz",
    "gmax",
    "grav",
    "launch_time",
    "SBEL",
    "VBEL",
    "psiblx",
    "tgt_num",
)

ANCOMX_SENTINEL = 123.0
ALCOMX_SENTINEL = 456.0
GNAV = 3.0
GRAV_BIAS = 1.5
GRAV = AGRAV
GMAX = 20.0
PSIBLx = 10.0
THTBLx = 3.0
PHIBLx = 2.0
SBELC = np.array([100.0, -50.0, -7000.0], dtype=float)
VBELC = np.array([250.0, 10.0, 5.0], dtype=float)
STELM = np.array([33000.0, 10000.0, -100.0], dtype=float)
VTELC = np.array([0.0, -5.0, 0.0], dtype=float)
STCEL_OTHER = np.array([99999.0, 0.0, 0.0], dtype=float)
VTCEL_OTHER = np.array([9.0, 9.0, 9.0], dtype=float)
STEL = np.array([34000.0, 11000.0, -200.0], dtype=float)
VTEL = np.array([0.0, -5.0, 0.0], dtype=float)
SBEL = np.array([120.0, -40.0, -6980.0], dtype=float)
VBEL = np.array([248.0, 12.0, 4.0], dtype=float)
FSPCB = np.array([12.0, 0.5, -9.8], dtype=float)
PSIPB = 0.05
THTPB = -0.03
SIGDPY = 0.02
SIGDPZ = -0.01


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


def _tblc():
    return mat3tr(PSIBLx * RAD, THTBLx * RAD, PHIBLx * RAD)


def _ctx(sim_time=0.0, int_step=DT):
    return SimContext(
        sim_time=sim_time,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _cpp_mid_pronav(stblc, vtelc, tblc, vbelc, gnav, grav_bias, grav):
    dtbc = float(np.linalg.norm(stblc))
    utblc = stblc * (1.0 / dtbc)
    utbbc = tblc @ utblc
    polar = polar_from_cart(utbbc)
    psiobcx = float(polar[1]) * DEG
    thtobcx = float(polar[2]) * DEG
    vtblc = vtelc - vbelc
    dvtbc = abs(float(utblc @ vtblc))
    tgoc = dtbc / dvtbc
    woelc = _skew(utblc) @ vtblc * (1.0 / dtbc)
    grav_comp = np.array([0.0, 0.0, grav_bias * grav], dtype=float)
    acbx = tblc @ ((_skew(woelc) @ utblc * gnav * dvtbc - grav_comp) * (1.0 / AGRAV))
    return acbx, woelc, utblc, tgoc, dtbc, dvtbc, psiobcx, thtobcx


def _cpp_term_comp(sbel, stel, vbel, vtel, fspcb, tblc, gnav, psipb, thtpb, sigdpy, sigdpz):
    sbtl = sbel - stel
    dbt = float(np.linalg.norm(sbtl))
    dum = float(sbtl @ (vbel - vtel))
    dcvel = abs(dum / dbt)
    fspcb1 = float(fspcb[0])
    adely = fspcb1 * tan(psipb) / AGRAV
    adelz = fspcb1 * tan(thtpb) / (cos(psipb) * AGRAV)
    gravl = np.array([0.0, 0.0, 1.0], dtype=float)
    gravb = tblc @ gravl
    gn = gnav * dcvel
    apny = gn * sigdpz / (cos(psipb) * AGRAV)
    apnz = gn * (sigdpz * tan(thtpb) * tan(psipb) + sigdpy / cos(thtpb)) / AGRAV
    all_ = apny + adely - float(gravb[1])
    ann = apnz + adelz + float(gravb[2])
    acbx = np.array([0.0, all_, -ann], dtype=float)
    return acbx, gn, apny, apnz, adely, adelz


def _limit_commands(acbx, gmax):
    all_ = float(acbx[1])
    ann = -float(acbx[2])
    aa = sqrt(all_ * all_ + ann * ann)
    if aa > gmax:
        aa = gmax
    if abs(ann) < SMALL and abs(all_) < SMALL:
        phi = 0.0
    else:
        phi = atan2(ann, all_)
    return aa * cos(phi), aa * sin(phi)


def _plant(
    store,
    *,
    mnav=0,
    stcel=None,
    vtcel=None,
    sbelc=None,
    vbelc=None,
    tblc=None,
    fspcb=None,
    stel=None,
    vtel=None,
    sbel=None,
    vbel=None,
    gmax=GMAX,
    grav=GRAV,
    psipb=PSIPB,
    thtpb=THTPB,
    sigdpy=SIGDPY,
    sigdpz=SIGDPZ,
    launch_time=None,
):
    if stcel is None:
        stcel = STCEL_OTHER
    if vtcel is None:
        vtcel = VTCEL_OTHER
    if sbelc is None:
        sbelc = SBELC
    if vbelc is None:
        vbelc = VBELC
    if tblc is None:
        tblc = _tblc()
    if fspcb is None:
        fspcb = FSPCB
    if stel is None:
        stel = STEL
    if vtel is None:
        vtel = VTEL
    if sbel is None:
        sbel = SBEL
    if vbel is None:
        vbel = VBEL
    specs = (
        ("mnav", mnav, "int", "out", "datalink"),
        ("STCEL", stcel, "vec", "out", "datalink"),
        ("VTCEL", vtcel, "vec", "out", "datalink"),
        ("SBELC", sbelc, "vec", "out", "ins"),
        ("VBELC", vbelc, "vec", "out", "ins"),
        ("TBLC", tblc, "mat", "out", "ins"),
        ("FSPCB", fspcb, "vec", "out", "ins"),
        ("STEL", stel, "vec", "out", "sensor"),
        ("VTEL", vtel, "vec", "out", "sensor"),
        ("psipb", psipb, "real", "out", "sensor"),
        ("thtpb", thtpb, "real", "out", "sensor"),
        ("sigdpy", sigdpy, "real", "out", "sensor"),
        ("sigdpz", sigdpz, "real", "out", "sensor"),
        ("SBEL", sbel, "vec", "state", "newton"),
        ("VBEL", vbel, "vec", "out", "newton"),
        ("gmax", gmax, "real", "diag", "aerodynamics"),
        ("grav", grav, "real", "out", "environment"),
    )
    for name, value, ftype, role, module in specs:
        store.define(Field(name, value, ftype, role, module))
    if launch_time is not None:
        store.define(Field("launch_time", launch_time, "real", "save", "newton"))


def _ready(
    *,
    mguid=0,
    mnav=0,
    gnav=GNAV,
    grav_bias=GRAV_BIAS,
    stelm=None,
    vtelc=None,
    epchta=0.0,
    ancomx=ANCOMX_SENTINEL,
    alcomx=ALCOMX_SENTINEL,
    launch_time=None,
    **plant_kw,
):
    vehicle = SimpleNamespace(store=StateStore())
    guid = Agm6Guidance()
    guid.define(vehicle)
    _plant(vehicle.store, mnav=mnav, launch_time=launch_time, **plant_kw)
    store = vehicle.store
    store.set("mguid", mguid)
    store.set("gnav", gnav)
    store.set("grav_bias", grav_bias)
    store.set("epchta", epchta)
    store.set("ancomx", ancomx)
    store.set("alcomx", alcomx)
    if stelm is None:
        stelm = STELM
    if vtelc is None:
        vtelc = VTELC
    store.set("STELM", stelm)
    store.set("VTELC", vtelc)
    guid.initialize(vehicle, _ctx())
    return vehicle, guid


def test_name_is_guidance():
    assert Agm6Guidance.name == "guidance"
    assert Agm6Guidance().name == "guidance"


def test_define_registers_cpp_def_guidance_fields():
    vehicle = SimpleNamespace(store=StateStore())
    Agm6Guidance().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    zeros3 = np.zeros(3)
    for name, (ftype, role, default, outputs) in FIELDS.items():
        field = store.field(name)
        assert field.module == "guidance"
        assert field.type == ftype
        assert field.role == role
        assert field.outputs == outputs
        if ftype == "int":
            assert store.get(name) == default
            assert type(store.get(name)) is int
        elif ftype == "real":
            assert store.get(name) == default
        else:
            np.testing.assert_array_equal(store.get(name), zeros3)
            assert store.get(name).shape == (3,)
    for name in EXTERNALS:
        assert name not in store.names()


def test_mguid0_mnav3_latches_stelm_without_command_write():
    stcel = np.array([33000.0, 10000.0, -100.0], dtype=float)
    vtcel = np.array([0.0, -5.0, 0.0], dtype=float)
    sim_time = 4.2
    vehicle, guid = _ready(
        mguid=0,
        mnav=3,
        stcel=stcel,
        vtcel=vtcel,
        stelm=np.zeros(3),
        vtelc=np.zeros(3),
        epchta=0.0,
    )
    store = vehicle.store
    guid.execute(vehicle, _ctx(sim_time=sim_time))
    np.testing.assert_allclose(store.get("STELM"), stcel, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VTELC"), vtcel, rtol=RTOL, atol=ATOL)
    assert store.get("epchta") == pytest.approx(sim_time, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == ANCOMX_SENTINEL
    assert store.get("alcomx") == ALCOMX_SENTINEL
    np.testing.assert_allclose(store.get("STELC"), stcel, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("STBLC"), stcel - SBELC, rtol=RTOL, atol=ATOL
    )


def test_mguid0_mnav0_no_raise_no_command_write():
    vehicle, guid = _ready(mguid=0, mnav=0)
    store = vehicle.store
    guid.execute(vehicle, _ctx())
    assert store.get("ancomx") == ANCOMX_SENTINEL
    assert store.get("alcomx") == ALCOMX_SENTINEL
    np.testing.assert_allclose(store.get("STELM"), STELM, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VTELC"), VTELC, rtol=RTOL, atol=ATOL)


def test_mguid0_still_extrapolates_stelc():
    stelm = np.array([100.0, 200.0, 300.0], dtype=float)
    vtelc = np.array([1.0, 2.0, 3.0], dtype=float)
    epchta = 1.0
    launch_time = 5.0
    vehicle, guid = _ready(
        mguid=0,
        mnav=0,
        stelm=stelm,
        vtelc=vtelc,
        epchta=epchta,
        launch_time=launch_time,
    )
    guid.execute(vehicle, _ctx(sim_time=99.0))
    want = stelm + vtelc * (launch_time - epchta)
    np.testing.assert_allclose(vehicle.store.get("STELC"), want, rtol=RTOL, atol=ATOL)


def test_mnav3_does_not_write_mnav_onto_grav_bias():
    vehicle, guid = _ready(mguid=0, mnav=3, grav_bias=GRAV_BIAS)
    guid.execute(vehicle, _ctx(sim_time=2.0))
    assert vehicle.store.get("grav_bias") == pytest.approx(GRAV_BIAS, rel=RTOL, abs=ATOL)
    assert vehicle.store.get("mnav") == 3


def test_mguid30_pronav_acbx_then_limiter_vs_cpp():
    tblc = _tblc()
    stblc = STELM - SBELC
    vehicle, guid = _ready(mguid=30, mnav=0, epchta=0.0)
    store = vehicle.store
    acbx = guid.guidance_mid_pronav(vehicle, stblc, VTELC)
    want_acbx, *_ = _cpp_mid_pronav(
        stblc, VTELC, tblc, VBELC, GNAV, GRAV_BIAS, GRAV
    )
    np.testing.assert_allclose(acbx, want_acbx, rtol=RTOL, atol=ATOL)
    assert acbx.shape == (3,)

    guid.execute(vehicle, _ctx())
    alcomx, ancomx = _limit_commands(want_acbx, GMAX)
    assert store.get("alcomx") == pytest.approx(alcomx, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(ancomx, rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(store.get("STBLC"), stblc, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("STELM"), STELM, rtol=RTOL, atol=ATOL)
    assert store.get("grav_bias") == pytest.approx(GRAV_BIAS, rel=RTOL, abs=ATOL)


def test_mguid6_term_comp_vs_cpp():
    tblc = _tblc()
    vehicle, guid = _ready(mguid=6, mnav=0)
    store = vehicle.store
    acbx = guid.guidance_term_comp(vehicle)
    want_acbx, *_ = _cpp_term_comp(
        SBEL, STEL, VBEL, VTEL, FSPCB, tblc, GNAV, PSIPB, THTPB, SIGDPY, SIGDPZ
    )
    np.testing.assert_allclose(acbx, want_acbx, rtol=RTOL, atol=ATOL)
    assert acbx.shape == (3,)

    guid.execute(vehicle, _ctx())
    alcomx, ancomx = _limit_commands(want_acbx, GMAX)
    assert store.get("alcomx") == pytest.approx(alcomx, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(ancomx, rel=RTOL, abs=ATOL)


def test_mguid40_uses_stel_minus_sbelc():
    tblc = _tblc()
    stblc_true = STEL - SBELC
    stblc_link = STELM - SBELC
    assert not np.allclose(stblc_true, stblc_link, rtol=RTOL, atol=ATOL)
    vehicle, guid = _ready(mguid=40, mnav=0, epchta=0.0)
    store = vehicle.store
    guid.execute(vehicle, _ctx())
    want_acbx, *_ = _cpp_mid_pronav(
        stblc_true, VTELC, tblc, VBELC, GNAV, GRAV_BIAS, GRAV
    )
    other_acbx, *_ = _cpp_mid_pronav(
        stblc_link, VTELC, tblc, VBELC, GNAV, GRAV_BIAS, GRAV
    )
    alcomx, ancomx = _limit_commands(want_acbx, GMAX)
    other_al, other_an = _limit_commands(other_acbx, GMAX)
    assert store.get("alcomx") == pytest.approx(alcomx, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(ancomx, rel=RTOL, abs=ATOL)
    assert store.get("alcomx") != pytest.approx(other_al, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") != pytest.approx(other_an, rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(store.get("STBLC"), stblc_true, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("mguid", [20, 5, 2])
def test_unsupported_mguid_raises(mguid):
    vehicle, guid = _ready(mguid=mguid, mnav=0)
    with pytest.raises(ValueError):
        guid.execute(vehicle, _ctx())
