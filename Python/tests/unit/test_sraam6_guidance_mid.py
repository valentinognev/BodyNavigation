from math import cos, sin

import numpy as np
import pytest

from cadac.constants import AGRAV, DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import polar_from_cart
from cadac.vehicles.sraam6.guidance import Sraam6Guidance

RTOL = 1e-12
ATOL = 1e-14

SBEL = np.array([0.0, 0.0, -5000.0])
STEL = np.array([10000.0, 500.0, -2000.0])
DVAE = 250.0
PSIALX = 180.0
THTALX = 0.0
DVBE = 250.0
GNAV = 3.75
TBL = np.eye(3)

DEFINED = (
    "mguid",
    "gnav",
    "ancomx",
    "alcomx",
    "gn",
    "mnav",
    "apny",
    "apnz",
    "adely",
    "adelz",
    "all",
    "ann",
    "epchta",
    "WOELC",
    "tgoc",
    "dtbc",
    "dvtbc",
    "psiobcx",
    "thtobcx",
    "UTBLC",
    "STELC",
    "STBLC",
    "STELM",
    "VTELC",
    "range",
    "azimuthx",
    "elevationx",
)
INT_DATA = ("mguid", "mnav")
OUT_SCRN = ("ancomx", "alcomx")
DIAG_PLOT = (
    "all",
    "ann",
    "dtbc",
    "dvtbc",
    "psiobcx",
    "thtobcx",
    "range",
    "azimuthx",
    "elevationx",
)
VEC_FIELDS = ("WOELC", "UTBLC", "STELC", "STBLC", "STELM", "VTELC")
EXTERNALS = ("STEL", "VTEL", "SBEL", "VBEL", "TBL", "time")


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _vtel():
    psial = PSIALX * RAD
    thtal = THTALX * RAD
    return np.array(
        [
            DVAE * cos(thtal) * cos(psial),
            DVAE * cos(thtal) * sin(psial),
            -DVAE * sin(thtal),
        ]
    )


def _vbel():
    return np.array([DVBE, 0.0, 0.0])


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


def _cpp_aapnb(stblc, vtelc, vbel, tbl, gnav):
    dtbc = float(np.sqrt(float(stblc @ stblc)))
    utblc = stblc * (1.0 / dtbc)
    vtblc = vtelc - vbel
    dvtbc = abs(float(utblc @ vtblc))
    woelc = _skew(utblc) @ vtblc * (1.0 / dtbc)
    aapnb = tbl @ (_skew(woelc) @ utblc) * gnav * dvtbc
    ancomx = -float(aapnb[2]) / AGRAV
    alcomx = float(aapnb[1]) / AGRAV
    return aapnb, ancomx, alcomx


def _ctx(sim_time=0.0):
    return SimContext(
        sim_time=sim_time,
        int_step=0.001,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _plant_externals(
    store,
    *,
    sbel=SBEL,
    stel=STEL,
    vtel=None,
    vbel=None,
    tbl=None,
    time=0.0,
    plant_tbl=True,
    plant_vbel=True,
):
    if vtel is None:
        vtel = _vtel()
    store.define(Field("STEL", stel, "vec", "out", "seeker"))
    store.define(Field("VTEL", vtel, "vec", "out", "seeker"))
    store.define(Field("SBEL", sbel, "vec", "state", "newton"))
    store.define(Field("time", time, "real", "exec", "kinematics"))
    if plant_vbel:
        if vbel is None:
            vbel = _vbel()
        store.define(Field("VBEL", vbel, "vec", "state", "newton"))
    if plant_tbl:
        if tbl is None:
            tbl = TBL
        store.define(Field("TBL", tbl, "mat", "out", "kinematics"))


def _ready(
    *,
    mguid=3,
    mnav=3,
    gnav=GNAV,
    sbel=SBEL,
    stel=STEL,
    vtel=None,
    vbel=None,
    tbl=None,
    time=0.0,
    plant=True,
    plant_tbl=True,
    plant_vbel=True,
):
    vehicle = _Vehicle()
    guidance = Sraam6Guidance()
    guidance.define(vehicle)
    if plant:
        _plant_externals(
            vehicle.store,
            sbel=sbel,
            stel=stel,
            vtel=vtel,
            vbel=vbel,
            tbl=tbl,
            time=time,
            plant_tbl=plant_tbl,
            plant_vbel=plant_vbel,
        )
    store = vehicle.store
    store.set("mguid", mguid)
    store.set("mnav", mnav)
    store.set("gnav", gnav)
    return vehicle, guidance, _ctx(sim_time=time)


def test_name_is_guidance():
    assert Sraam6Guidance().name == "guidance"


def test_define_registers_def_guidance_fields():
    vehicle = _Vehicle()
    Sraam6Guidance().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    for name in INT_DATA:
        assert store.field(name).type == "int"
        assert store.field(name).role == "data"
        assert store.field(name).module == "guidance"
        assert store.get(name) == 0
    assert store.field("gnav").type == "real"
    assert store.field("gnav").role == "data"
    assert store.get("gnav") == 0.0
    for name in OUT_SCRN:
        assert store.field(name).role == "out"
        assert store.field(name).outputs == ("scrn",)
        assert store.get(name) == 0.0
    for name in DIAG_PLOT:
        assert store.field(name).role == "diag"
        assert store.field(name).outputs == ("plot",)
    for name in VEC_FIELDS:
        np.testing.assert_allclose(store.get(name), np.zeros(3), rtol=RTOL, atol=ATOL)
        assert store.field(name).type == "vec"
    assert store.field("STELM").role == "save"
    assert store.field("VTELC").role == "save"
    assert store.field("epchta").role == "save"
    for name in EXTERNALS:
        assert name not in store.names()


def test_1v1_midcourse_ancomx_alcomx_match_cpp_aapnb():
    vehicle, guidance, ctx = _ready(mguid=3, mnav=3)
    stel = STEL
    sbel = SBEL
    vtel = _vtel()
    vbel = _vbel()
    stblc = stel - sbel
    aapnb, ancomx, alcomx = _cpp_aapnb(stblc, vtel, vbel, TBL, GNAV)
    assert np.all(np.isfinite(aapnb))
    guidance.execute(vehicle, ctx)
    store = vehicle.store
    assert store.get("mnav") == 0
    np.testing.assert_allclose(store.get("ancomx"), ancomx, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("alcomx"), alcomx, rtol=RTOL, atol=ATOL)
    polar = polar_from_cart(stel - sbel)
    np.testing.assert_allclose(store.get("range"), polar[0], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("azimuthx"), polar[1] * DEG, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("elevationx"), polar[2] * DEG, rtol=RTOL, atol=ATOL)


def test_mguid_0_does_not_require_los():
    vehicle, guidance, ctx = _ready(
        mguid=0,
        mnav=0,
        stel=SBEL,
        sbel=SBEL,
        plant_tbl=False,
        plant_vbel=False,
    )
    store = vehicle.store
    store.set("ancomx", 12.0)
    store.set("alcomx", -4.0)
    guidance.execute(vehicle, ctx)
    assert store.get("mguid") == 0
    assert store.get("mnav") == 0
    assert store.get("ancomx") == 12.0
    assert store.get("alcomx") == -4.0
    assert "TBL" not in store.names()
    assert "VBEL" not in store.names()


def test_mguid_1_raises():
    vehicle, guidance, ctx = _ready(mguid=1, mnav=3)
    with pytest.raises(ValueError, match="mguid"):
        guidance.execute(vehicle, ctx)


def test_mnav_1_raises():
    vehicle, guidance, ctx = _ready(mguid=3, mnav=1)
    with pytest.raises(ValueError, match="mnav"):
        guidance.execute(vehicle, ctx)


def test_mnav0_keeps_snapshot_and_extrapolates():
    vehicle, guidance, ctx0 = _ready(mguid=3, mnav=3, time=0.0)
    guidance.execute(vehicle, ctx0)
    store = vehicle.store
    assert store.get("mnav") == 0
    snapshot_stel = store.get("STELM").copy()
    snapshot_vtel = store.get("VTELC").copy()
    np.testing.assert_allclose(snapshot_stel, STEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(snapshot_vtel, _vtel(), rtol=RTOL, atol=ATOL)

    live_stel = STEL + np.array([111.0, -22.0, 33.0])
    store.set("STEL", live_stel)
    store.set("time", 1.0)
    guidance.execute(vehicle, _ctx(sim_time=1.0))
    assert store.get("mnav") == 0
    np.testing.assert_allclose(store.get("STELM"), snapshot_stel, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VTELC"), snapshot_vtel, rtol=RTOL, atol=ATOL)
    stelc = snapshot_stel + snapshot_vtel * 1.0
    stblc = stelc - SBEL
    np.testing.assert_allclose(store.get("STELC"), stelc, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("STBLC"), stblc, rtol=RTOL, atol=ATOL)
    aapnb, ancomx, alcomx = _cpp_aapnb(stblc, snapshot_vtel, _vbel(), TBL, GNAV)
    np.testing.assert_allclose(store.get("ancomx"), ancomx, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("alcomx"), alcomx, rtol=RTOL, atol=ATOL)
    polar = polar_from_cart(live_stel - SBEL)
    np.testing.assert_allclose(store.get("range"), polar[0], rtol=RTOL, atol=ATOL)


def test_mguid_6_calls_guidance_term():
    vehicle, guidance, ctx = _ready(mguid=6, mnav=0)
    store = vehicle.store
    store.define(Field("thtpb", 0.0, "real", "out", "seeker"))
    store.define(Field("psipb", 0.0, "real", "out", "seeker"))
    store.define(Field("sigdpy", 0.0, "real", "out", "seeker"))
    store.define(Field("sigdpz", 0.0, "real", "out", "seeker"))
    store.define(Field("FSPB", (0.0, 0.0, 0.0), "vec", "out", "newton"))
    store.define(Field("gmax", 50.0, "real", "diag", "aerodynamics"))
    store.define(Field("trcond", 0, "int", "diag", "aerodynamics"))
    store.define(Field("trcvel", 10e-4, "real", "data", "aerodynamics"))
    store.set("ancomx", 99.0)
    store.set("alcomx", 99.0)
    guidance.execute(vehicle, ctx)
    assert store.get("mguid") == 6
    np.testing.assert_allclose(store.get("alcomx"), 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ancomx"), 1.0, rtol=RTOL, atol=ATOL)
