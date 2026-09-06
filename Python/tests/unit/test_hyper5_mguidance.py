import math

import numpy as np
import pytest

from cadac.constants import DEG, EPS, RAD, REARTH
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.earth import cadine, cadtei, cadtge
from cadac.math.frames import cadtbv
from cadac.vehicles.hyper5.guidance import Hyper5Guidance

# Demo 4.7 Terminal pro-nav
PRONAV_GAIN = 3.5
BIAS = 5.0
GRAV = 9.81
CLOSING_SPEED = 240.0
WOEB = np.array([0.02, -0.05, 0.01])
UTBB = np.array([0.95, 0.22, -0.22])
TBG = np.array(
    [
        [0.8, 0.0, 0.6],
        [0.0, 1.0, 0.0],
        [-0.6, 0.0, 0.8],
    ]
)
ALLIMX = 1.0
ANPOSLIMX = 2.0
ANNEGLIMX = -2.0

# Demo 4.6 point guidance
POINT_LONX = -106.3
POINT_LATX = 32.0
POINT_ALT = 20000.0
POINT_PSIVGX = 0.0
POINT_THTVGX = 0.0
POINT_DVBE = 1181.0
WP_LONX_46 = -106.28
WP_LATX_46 = 33.4
WP_ALT_46 = 1200.0
POINT_GAIN = 0.04
PHILIMX_46 = 70.0

# Demo 5.12 arc midcourse
ARC_LONX = -120.6
ARC_LATX = 34.7
ARC_ALT = 20000.0
ARC_PSIVGX = 90.0
ARC_THTVGX = 0.0
ARC_DVBE = 1475.0
WP_LONX_VEGAS = -115.14
WP_LATX_VEGAS = 36.17
WP_ALT_ARC = 0.0
PHILIMX_ARC = 60.0
ALPHAX_ARC = 0.0
PHIMVX_ARC = 0.0
FSPV_ARC = np.array([2.0, 1.0, -12.0])
TIME = 0.0

RTOL = 1e-12
ATOL = 1e-14


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.05,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _cadac_sign(variable):
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


def _angle(vec1, vec2):
    scalar = float(vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2])
    abs1 = math.sqrt(float(vec1[0] ** 2 + vec1[1] ** 2 + vec1[2] ** 2))
    abs2 = math.sqrt(float(vec2[0] ** 2 + vec2[1] ** 2 + vec2[2] ** 2))
    dum = abs1 * abs2
    if abs1 * abs2 > EPS:
        argument = scalar / dum
    else:
        argument = 1.0
    if argument > 1.0:
        argument = 1.0
    if argument < -1.0:
        argument = -1.0
    return math.acos(argument)


def _round3_kinematics(lonx, latx, alt, psivgx, thtvgx, dvbe, time):
    sbig = np.array([0.0, 0.0, -(alt + REARTH)])
    tge = cadtge(lonx * RAD, latx * RAD)
    teg = tge.T
    sbie = teg @ sbig
    tei = cadtei(time)
    sbii = tei.T @ sbie
    psivg = psivgx * RAD
    thtvg = thtvgx * RAD
    vbeg = np.array(
        [
            dvbe * np.cos(thtvg) * np.cos(psivg),
            dvbe * np.cos(thtvg) * np.sin(psivg),
            dvbe * (-np.sin(thtvg)),
        ]
    )
    tig = tei.T @ teg
    return tig, sbii, vbeg


def _expected_pronav(woeb, utbb, pronav_gain, closing_speed, tbg, grav, bias):
    grav_g = np.array([0.0, 0.0, grav + bias])
    return _skew(woeb) @ utbb * (pronav_gain * closing_speed) - tbg @ grav_g


def _clip_commands(alcomx, ancomx, allimx, anposlimx, anneglimx):
    if ancomx > anposlimx:
        ancomx = anposlimx
    if ancomx < anneglimx:
        ancomx = anneglimx
    if alcomx > allimx:
        alcomx = allimx
    if alcomx < -allimx:
        alcomx = -allimx
    return alcomx, ancomx


def _expected_arc(
    wp_lonx,
    wp_latx,
    wp_alt,
    time,
    fspv,
    grav,
    tig,
    dvbe,
    vbeg,
    sbii,
    alphax,
    phimvx,
    philimx,
):
    swii = cadine(wp_lonx * RAD, wp_latx * RAD, wp_alt, time)
    swbg = tig.T @ (swii - sbii)
    swbg1 = float(swbg[0])
    swbg2 = float(swbg[1])
    sh = np.array([swbg1, swbg2, 0.0])
    dwbh = math.sqrt(swbg1 * swbg1 + swbg2 * swbg2)
    vbeg1 = float(vbeg[0])
    vbeg2 = float(vbeg[1])
    vh = np.array([vbeg1, vbeg2, 0.0])
    uv = _skew(vh) @ sh
    psiwvx = DEG * _angle(vh, sh)
    zz = np.array([0.0, 0.0, 1.0])
    psiwvx = psiwvx * _cadac_sign(float(uv @ zz))
    tbv = cadtbv(phimvx * RAD, alphax * RAD)
    fspb = tbv @ fspv
    fspb3 = float(fspb[2])
    argument = 0.0
    if abs(psiwvx) < 90:
        num = -2 * dvbe * dvbe * math.sin(psiwvx * RAD)
        denom = fspb3 * dwbh
        if denom != 0:
            argument = num / denom
        if abs(argument) <= 1.0 and abs(math.asin(argument)) < philimx * RAD:
            phicx = DEG * math.asin(argument)
        else:
            phicx = philimx * _cadac_sign(argument)
    else:
        phicx = philimx * _cadac_sign(psiwvx)
    rad_min = dvbe * dvbe / (grav * math.tan(philimx * RAD))
    if dwbh < 0.2 * rad_min:
        wp_flag = _cadac_sign(float(vh @ sh))
    else:
        wp_flag = 0
    return phicx, {
        "SWBG": swbg,
        "wp_grdrange": dwbh,
        "rad_min": rad_min,
        "wp_flag": wp_flag,
    }


def _plant_control(
    store,
    phicx=12.0,
    ancomx=1.5,
    alcomx=0.5,
    anposlimx=ANPOSLIMX,
    anneglimx=ANNEGLIMX,
    allimx=ALLIMX,
    philimx=PHILIMX_46,
    alphax=0.0,
    phimvx=0.0,
):
    store.define(Field("phicx", phicx, "real", "data", "control", ("scrn", "plot")))
    store.define(Field("ancomx", ancomx, "real", "data", "control", ("scrn", "plot")))
    store.define(Field("alcomx", alcomx, "real", "data", "control", ("scrn", "plot")))
    store.define(Field("anposlimx", anposlimx, "real", "data", "control"))
    store.define(Field("anneglimx", anneglimx, "real", "data", "control"))
    store.define(Field("allimx", allimx, "real", "data", "control"))
    store.define(Field("philimx", philimx, "real", "data", "control"))
    store.define(Field("alphax", alphax, "real", "out", "control", ("scrn", "plot")))
    store.define(Field("phimvx", phimvx, "real", "out", "control", ("scrn", "plot")))


def _ready_pronav(
    mguidance=66,
    pronav_gain=PRONAV_GAIN,
    bias=BIAS,
    grav=GRAV,
    closing_speed=CLOSING_SPEED,
    woeb=None,
    utbb=None,
    tbg=None,
    plant_control=True,
    anposlimx=ANPOSLIMX,
    anneglimx=ANNEGLIMX,
    allimx=ALLIMX,
    phicx=12.0,
    ancomx=1.5,
    alcomx=0.5,
):
    vehicle = _Vehicle()
    guidance = Hyper5Guidance()
    guidance.define(vehicle)
    store = vehicle.store
    if woeb is None:
        woeb = WOEB
    if utbb is None:
        utbb = UTBB
    if tbg is None:
        tbg = TBG
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("TBG", tbg, "mat", "out", "control"))
    store.define(Field("WOEB", woeb, "vec", "out", "seeker"))
    store.define(Field("UTBB", utbb, "vec", "out", "seeker"))
    store.define(Field("closing_speed", closing_speed, "real", "out", "seeker"))
    store.set("mguidance", mguidance)
    store.set("pronav_gain", pronav_gain)
    store.set("bias", bias)
    if plant_control:
        _plant_control(
            store,
            phicx=phicx,
            ancomx=ancomx,
            alcomx=alcomx,
            anposlimx=anposlimx,
            anneglimx=anneglimx,
            allimx=allimx,
        )
    return vehicle, guidance


def _ready_point(
    mguidance=44,
    plant_control=True,
    anposlimx=ANPOSLIMX,
    anneglimx=ANNEGLIMX,
    allimx=ALLIMX,
    phicx=12.0,
    ancomx=1.5,
    alcomx=0.5,
):
    vehicle = _Vehicle()
    guidance = Hyper5Guidance()
    guidance.define(vehicle)
    store = vehicle.store
    tig, sbii, vbeg = _round3_kinematics(
        POINT_LONX,
        POINT_LATX,
        POINT_ALT,
        POINT_PSIVGX,
        POINT_THTVGX,
        POINT_DVBE,
        TIME,
    )
    store.define(Field("time", TIME, "real", "exec", "environment"))
    store.define(Field("grav", GRAV, "real", "out", "environment"))
    store.define(Field("tig", tig, "mat", "init/out", "newton"))
    store.define(Field("thtvgx", POINT_THTVGX, "real", "init/out", "newton"))
    store.define(Field("vbeg", vbeg, "vec", "state", "newton"))
    store.define(Field("sbii", sbii, "vec", "state", "newton"))
    store.set("mguidance", mguidance)
    store.set("wp_lonx", WP_LONX_46)
    store.set("wp_latx", WP_LATX_46)
    store.set("wp_alt", WP_ALT_46)
    store.set("point_gain", POINT_GAIN)
    if plant_control:
        _plant_control(
            store,
            phicx=phicx,
            ancomx=ancomx,
            alcomx=alcomx,
            anposlimx=anposlimx,
            anneglimx=anneglimx,
            allimx=allimx,
            philimx=PHILIMX_46,
        )
    return vehicle, guidance


def _ready_arc(
    mguidance=70,
    lonx=ARC_LONX,
    latx=ARC_LATX,
    alt=ARC_ALT,
    psivgx=ARC_PSIVGX,
    thtvgx=ARC_THTVGX,
    dvbe=ARC_DVBE,
    wp_lonx=WP_LONX_VEGAS,
    wp_latx=WP_LATX_VEGAS,
    wp_alt=WP_ALT_ARC,
    philimx=PHILIMX_ARC,
    alphax=ALPHAX_ARC,
    phimvx=PHIMVX_ARC,
    fspv=None,
    plant_control=True,
    plant_psivgx=False,
    anposlimx=2.0,
    anneglimx=-1.0,
    allimx=1.0,
    phicx=12.0,
    ancomx=1.5,
    alcomx=0.5,
):
    vehicle = _Vehicle()
    guidance = Hyper5Guidance()
    guidance.define(vehicle)
    store = vehicle.store
    if fspv is None:
        fspv = FSPV_ARC
    tig, sbii, vbeg = _round3_kinematics(
        lonx, latx, alt, psivgx, thtvgx, dvbe, TIME
    )
    store.define(Field("time", TIME, "real", "exec", "environment"))
    store.define(Field("FSPV", fspv, "vec", "out", "forces", ("plot",)))
    store.define(Field("grav", GRAV, "real", "out", "environment"))
    store.define(Field("tig", tig, "mat", "init/out", "newton"))
    store.define(Field("dvbe", dvbe, "real", "init/out", "newton"))
    store.define(Field("vbeg", vbeg, "vec", "state", "newton"))
    store.define(Field("sbii", sbii, "vec", "state", "newton"))
    if plant_psivgx:
        store.define(Field("psivgx", psivgx, "real", "init/out", "newton"))
    store.set("mguidance", mguidance)
    store.set("wp_lonx", wp_lonx)
    store.set("wp_latx", wp_latx)
    store.set("wp_alt", wp_alt)
    if plant_control:
        _plant_control(
            store,
            phicx=phicx,
            ancomx=ancomx,
            alcomx=alcomx,
            anposlimx=anposlimx,
            anneglimx=anneglimx,
            allimx=allimx,
            philimx=philimx,
            alphax=alphax,
            phimvx=phimvx,
        )
    return vehicle, guidance


def test_mguidance_66_writes_finite_commands():
    vehicle, guidance = _ready_pronav()
    guidance.execute(vehicle, _ctx())
    store = vehicle.store
    assert math.isfinite(store.get("alcomx"))
    assert math.isfinite(store.get("ancomx"))
    assert math.isfinite(store.get("phicx"))
    apnb = _expected_pronav(WOEB, UTBB, PRONAV_GAIN, CLOSING_SPEED, TBG, GRAV, BIAS)
    alcomx, ancomx = _clip_commands(
        apnb[1] / GRAV, -apnb[2] / GRAV, ALLIMX, ANPOSLIMX, ANNEGLIMX
    )
    assert alcomx != 0.0 or ancomx != 0.0
    assert _approx(store.get("alcomx"), alcomx)
    assert _approx(store.get("ancomx"), ancomx)
    assert store.get("phicx") == 12.0


def test_mguidance_0_no_raise_without_writing():
    vehicle = _Vehicle()
    guidance = Hyper5Guidance()
    guidance.define(vehicle)
    store = vehicle.store
    _plant_control(store)
    store.set("mguidance", 0)
    store.set("wp_sltrange", 1.0)
    store.set("wp_grdrange", 2.0)
    store.set("rad_min", 3.0)
    store.set("wp_flag", 5)
    guidance.execute(vehicle, _ctx())
    assert store.get("alcomx") == 0.5
    assert store.get("ancomx") == 1.5
    assert store.get("phicx") == 12.0
    assert store.get("wp_sltrange") == 1.0
    assert store.get("wp_grdrange") == 2.0
    assert store.get("rad_min") == 3.0
    assert store.get("wp_flag") == 5
    assert store.get("mguidance") == 0


@pytest.mark.parametrize("mguidance", [30, 33, 99, 3, 6, 40, 43, 60])
def test_unused_mguidance_raises_valueerror(mguidance):
    vehicle = _Vehicle()
    guidance = Hyper5Guidance()
    guidance.define(vehicle)
    vehicle.store.set("mguidance", mguidance)
    with pytest.raises(ValueError):
        guidance.execute(vehicle, _ctx())


def test_mguidance_44_writes_alcomx_ancomx_from_point():
    vehicle, guidance = _ready_point()
    apgv = guidance.guidance_point(vehicle)
    alcomx, ancomx = _clip_commands(
        apgv[1] / GRAV, -apgv[2] / GRAV, ALLIMX, ANPOSLIMX, ANNEGLIMX
    )
    vehicle, guidance = _ready_point()
    guidance.execute(vehicle, _ctx())
    store = vehicle.store
    assert math.isfinite(store.get("alcomx"))
    assert math.isfinite(store.get("ancomx"))
    assert _approx(store.get("alcomx"), alcomx)
    assert _approx(store.get("ancomx"), ancomx)
    assert store.get("phicx") == 12.0


def test_mguidance_70_writes_phicx_from_arc_and_zeros_load_commands():
    vehicle, guidance = _ready_arc()
    expected_phicx, _ = _expected_arc(
        WP_LONX_VEGAS,
        WP_LATX_VEGAS,
        WP_ALT_ARC,
        TIME,
        FSPV_ARC,
        GRAV,
        vehicle.store.get("tig"),
        ARC_DVBE,
        vehicle.store.get("vbeg"),
        vehicle.store.get("sbii"),
        ALPHAX_ARC,
        PHIMVX_ARC,
        PHILIMX_ARC,
    )
    guidance.execute(vehicle, _ctx())
    store = vehicle.store
    assert math.isfinite(store.get("phicx"))
    assert _approx(store.get("phicx"), expected_phicx)
    assert store.get("alcomx") == 0.0
    assert store.get("ancomx") == 0.0


def test_guidance_arc_demo_512_matches_cpp():
    vehicle, guidance = _ready_arc()
    store = vehicle.store
    expected_phicx, expected = _expected_arc(
        store.get("wp_lonx"),
        store.get("wp_latx"),
        store.get("wp_alt"),
        store.get("time"),
        store.get("FSPV"),
        store.get("grav"),
        store.get("tig"),
        store.get("dvbe"),
        store.get("vbeg"),
        store.get("sbii"),
        store.get("alphax"),
        store.get("phimvx"),
        store.get("philimx"),
    )
    got = guidance.guidance_arc(vehicle)
    assert math.isfinite(got)
    assert _approx(got, expected_phicx)
    np.testing.assert_allclose(store.get("SWBG"), expected["SWBG"], rtol=RTOL, atol=ATOL)
    assert _approx(store.get("wp_grdrange"), expected["wp_grdrange"])
    assert _approx(store.get("rad_min"), expected["rad_min"])
    assert store.get("wp_flag") == expected["wp_flag"]
    assert expected["wp_flag"] == 0


def test_guidance_arc_wp_flag_uses_0_2_rad_min_not_two():
    vehicle, guidance = _ready_arc(wp_lonx=ARC_LONX, wp_latx=ARC_LATX + 0.45)
    store = vehicle.store
    _, expected = _expected_arc(
        store.get("wp_lonx"),
        store.get("wp_latx"),
        store.get("wp_alt"),
        store.get("time"),
        store.get("FSPV"),
        store.get("grav"),
        store.get("tig"),
        store.get("dvbe"),
        store.get("vbeg"),
        store.get("sbii"),
        store.get("alphax"),
        store.get("phimvx"),
        store.get("philimx"),
    )
    assert expected["wp_grdrange"] >= 0.2 * expected["rad_min"]
    assert expected["wp_grdrange"] < 2 * expected["rad_min"]
    assert expected["wp_flag"] == 0
    guidance.guidance_arc(vehicle)
    assert store.get("wp_flag") == 0


def test_guidance_arc_wp_flag_plus_one_when_closing_inside_0_2_rad_min():
    vehicle, guidance = _ready_arc(wp_lonx=ARC_LONX + 0.05, wp_latx=ARC_LATX)
    store = vehicle.store
    _, expected = _expected_arc(
        store.get("wp_lonx"),
        store.get("wp_latx"),
        store.get("wp_alt"),
        store.get("time"),
        store.get("FSPV"),
        store.get("grav"),
        store.get("tig"),
        store.get("dvbe"),
        store.get("vbeg"),
        store.get("sbii"),
        store.get("alphax"),
        store.get("phimvx"),
        store.get("philimx"),
    )
    assert expected["wp_grdrange"] < 0.2 * expected["rad_min"]
    assert expected["wp_flag"] == 1
    guidance.guidance_arc(vehicle)
    assert store.get("wp_flag") == 1


def test_guidance_arc_min_turn_when_abs_psiwvx_ge_90():
    vehicle, guidance = _ready_arc(
        psivgx=0.0, wp_lonx=ARC_LONX, wp_latx=ARC_LATX - 2.0
    )
    store = vehicle.store
    expected_phicx, _ = _expected_arc(
        store.get("wp_lonx"),
        store.get("wp_latx"),
        store.get("wp_alt"),
        store.get("time"),
        store.get("FSPV"),
        store.get("grav"),
        store.get("tig"),
        store.get("dvbe"),
        store.get("vbeg"),
        store.get("sbii"),
        store.get("alphax"),
        store.get("phimvx"),
        store.get("philimx"),
    )
    got = guidance.guidance_arc(vehicle)
    assert _approx(got, expected_phicx)
    assert abs(expected_phicx) == pytest.approx(PHILIMX_ARC, rel=RTOL, abs=ATOL)


def test_guidance_arc_denom_zero_keeps_asin_argument_zero():
    vehicle, guidance = _ready_arc(
        wp_lonx=ARC_LONX, wp_latx=ARC_LATX, wp_alt=ARC_ALT
    )
    got = guidance.guidance_arc(vehicle)
    assert _approx(got, 0.0)


def test_guidance_arc_does_not_read_unused_psivgx():
    vehicle, guidance = _ready_arc(plant_psivgx=False)
    with pytest.raises(KeyError):
        vehicle.store.get("psivgx")
    phicx = guidance.guidance_arc(vehicle)
    assert math.isfinite(phicx)


def test_mguidance_66_clips_alcomx_and_ancomx():
    vehicle, guidance = _ready_pronav(allimx=0.01, anposlimx=0.02, anneglimx=-0.02)
    guidance.execute(vehicle, _ctx())
    store = vehicle.store
    apnb = _expected_pronav(WOEB, UTBB, PRONAV_GAIN, CLOSING_SPEED, TBG, GRAV, BIAS)
    alcomx, ancomx = _clip_commands(
        apnb[1] / GRAV, -apnb[2] / GRAV, 0.01, 0.02, -0.02
    )
    assert _approx(store.get("alcomx"), alcomx)
    assert _approx(store.get("ancomx"), ancomx)
    assert abs(store.get("alcomx")) <= 0.01 + ATOL
    assert store.get("ancomx") <= 0.02 + ATOL
    assert store.get("ancomx") >= -0.02 - ATOL
