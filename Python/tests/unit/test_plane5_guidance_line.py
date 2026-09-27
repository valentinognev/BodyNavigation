import math

import numpy as np
import pytest

from cadac.constants import RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr, polar_from_cart
from cadac.vehicles.flat3.falcon5.guidance import Plane5Guidance

# turning_to_IP waypoint #1 / origin heading 0; line ICs from later events
SWEL1 = 5000.0
SWEL2 = 2000.0
SWEL3 = 0.0
LINE_GAIN = 1.5
NL_GAIN_FACT = 0.4
DECREMENT = 800.0
PSIFLX = 180.0
THTFLX_30 = 0.0
THTFLX_33 = -30.0
PHILIMX = 70.0
GRAV = 9.81
THTVLX = 0.0
SBEL = np.array([0.0, 0.0, -3500.0])
VBEL = np.array([200.0, 0.0, 0.0])
ANPOSLIMX = 3.0
ANNEGLIMX = -1.0
ALLIMX = 1.0
PHICX = 0.0

LINE_FIELDS = {
    "line_gain": ("real", "data", (), 0.0),
    "nl_gain_fact": ("real", "data", (), 1.0),
    "decrement": ("real", "data", (), 0.0),
    "psiflx": ("real", "data", (), 0.0),
    "thtflx": ("real", "data", (), 0.0),
    "nl_gain": ("real", "dia", (), 0.0),
    "VBEF": ("vec", "dia", (), np.zeros(3)),
}


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


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


def _expected_line(
    swel,
    sbel,
    vbel,
    line_gain,
    nl_gain_fact,
    decrement,
    psiflx,
    thtflx,
    grav,
    thtvlx,
    philimx,
    write,
):
    tfl = mat2tr(psiflx * RAD, thtflx * RAD)
    swbl = swel - sbel
    polar = polar_from_cart(swbl)
    wp_sltrange = polar[0]
    psiol = polar[1]
    thtol = polar[2]
    tol = mat2tr(psiol, thtol)
    wp_grdrange = math.hypot(swbl[0], swbl[1])
    vbeo = tol @ vbel
    vbef = tfl @ vbel
    nl_gain = nl_gain_fact * (1 - math.exp(-wp_sltrange / decrement))
    algv1 = grav * math.sin(thtvlx * RAD)
    algv2 = line_gain * (-vbeo[1] + nl_gain * vbef[1])
    algv3 = line_gain * (-vbeo[2] + nl_gain * vbef[2]) - grav * math.cos(thtvlx * RAD)
    algv = np.array([algv1, algv2, algv3])
    dvbe = math.sqrt(float(vbel[0] ** 2 + vbel[1] ** 2 + vbel[2] ** 2))
    rad_min = dvbe * dvbe / (grav * math.tan(philimx * RAD))
    if wp_grdrange < 2 * rad_min:
        sh = np.array([swbl[0], swbl[1], 0.0])
        vh = np.array([vbel[0], vbel[1], 0.0])
        wp_flag = _cadac_sign(float(vh @ sh))
        if wp_flag == 1:
            write = 1
    else:
        wp_flag = 0
    return algv, {
        "wp_sltrange": wp_sltrange,
        "nl_gain": nl_gain,
        "VBEO": vbeo,
        "VBEF": vbef,
        "wp_grdrange": wp_grdrange,
        "SWBL": swbl,
        "rad_min": rad_min,
        "write": write,
        "wp_flag": wp_flag,
    }


def _ready(
    mguidance=30,
    swel1=SWEL1,
    swel2=SWEL2,
    swel3=SWEL3,
    line_gain=LINE_GAIN,
    nl_gain_fact=NL_GAIN_FACT,
    decrement=DECREMENT,
    psiflx=PSIFLX,
    thtflx=THTFLX_30,
    point_gain=1.0,
    sbel=SBEL,
    vbel=VBEL,
    grav=GRAV,
    thtvlx=THTVLX,
    philimx=PHILIMX,
    anposlimx=ANPOSLIMX,
    anneglimx=ANNEGLIMX,
    allimx=ALLIMX,
    phicx=PHICX,
    ancomx=0.0,
    alcomx=0.0,
    write=0,
    wp_flag=0,
):
    vehicle = _Vehicle()
    guidance = Plane5Guidance()
    guidance.define(vehicle)
    store = vehicle.store
    store.define(Field("SBEL", sbel, "vec", "state", "newton"))
    store.define(Field("VBEL", vbel, "vec", "state", "newton"))
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("thtvlx", thtvlx, "real", "out", "newton"))
    store.define(Field("philimx", philimx, "real", "data", "control"))
    store.define(Field("ancomx", ancomx, "real", "data", "control", ("plot",)))
    store.define(Field("alcomx", alcomx, "real", "data", "control", ("plot",)))
    store.define(Field("anposlimx", anposlimx, "real", "data", "control"))
    store.define(Field("anneglimx", anneglimx, "real", "data", "control"))
    store.define(Field("allimx", allimx, "real", "data", "control"))
    store.define(Field("phicx", phicx, "real", "data", "control", ("scrn", "plot")))
    store.set("mguidance", mguidance)
    store.set("swel1", swel1)
    store.set("swel2", swel2)
    store.set("swel3", swel3)
    store.set("point_gain", point_gain)
    store.set("line_gain", line_gain)
    store.set("nl_gain_fact", nl_gain_fact)
    store.set("decrement", decrement)
    store.set("psiflx", psiflx)
    store.set("thtflx", thtflx)
    store.set("write", write)
    store.set("wp_flag", wp_flag)
    return vehicle, guidance


def test_define_registers_line_guidance_fields():
    vehicle = _Vehicle()
    Plane5Guidance().define(vehicle)
    store = vehicle.store
    for name, (ftype, role, outputs, default) in LINE_FIELDS.items():
        assert name in store.names()
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "guidance"
        assert field.outputs == outputs
        if ftype == "vec":
            np.testing.assert_array_equal(store.get(name), default)
        else:
            assert store.get(name) == default


def test_mguidance_30_origin_heading_0_finite_alcomx_zero_ancomx():
    vehicle, guidance = _ready(mguidance=30)
    guidance.execute(vehicle, _ctx())
    store = vehicle.store
    assert math.isfinite(store.get("alcomx"))
    assert store.get("alcomx") != 0.0
    assert store.get("ancomx") == 0.0


def test_mguidance_33_origin_heading_0_finite_alcomx_and_ancomx():
    vehicle, guidance = _ready(mguidance=33, thtflx=THTFLX_33)
    guidance.execute(vehicle, _ctx())
    store = vehicle.store
    assert math.isfinite(store.get("alcomx"))
    assert math.isfinite(store.get("ancomx"))
    assert store.get("alcomx") != 0.0
    assert store.get("ancomx") != 0.0


def test_wp_flag_zero_outside_two_rad_min_at_origin_ne_waypoint():
    vehicle, guidance = _ready()
    _, expected = _expected_line(
        np.array([SWEL1, SWEL2, SWEL3]),
        SBEL,
        VBEL,
        LINE_GAIN,
        NL_GAIN_FACT,
        DECREMENT,
        PSIFLX,
        THTFLX_30,
        GRAV,
        THTVLX,
        PHILIMX,
        0,
    )
    assert expected["wp_grdrange"] >= 2 * expected["rad_min"]
    assert expected["wp_flag"] == 0

    guidance.execute(vehicle, _ctx())

    assert vehicle.store.get("wp_flag") == 0


def test_wp_flag_plus_one_when_closing_inside_two_rad_min():
    swel1, swel2, swel3 = 500.0, 200.0, 0.0
    vehicle, guidance = _ready(swel1=swel1, swel2=swel2, swel3=swel3)
    _, expected = _expected_line(
        np.array([swel1, swel2, swel3]),
        SBEL,
        VBEL,
        LINE_GAIN,
        NL_GAIN_FACT,
        DECREMENT,
        PSIFLX,
        THTFLX_30,
        GRAV,
        THTVLX,
        PHILIMX,
        0,
    )
    assert expected["wp_grdrange"] < 2 * expected["rad_min"]
    assert expected["wp_flag"] == 1

    guidance.execute(vehicle, _ctx())

    assert vehicle.store.get("wp_flag") == 1
    assert vehicle.store.get("write") == 1


def test_wp_flag_minus_one_when_fleeting_inside_two_rad_min():
    swel1, swel2, swel3 = -200.0, 50.0, 0.0
    vehicle, guidance = _ready(swel1=swel1, swel2=swel2, swel3=swel3)
    _, expected = _expected_line(
        np.array([swel1, swel2, swel3]),
        SBEL,
        VBEL,
        LINE_GAIN,
        NL_GAIN_FACT,
        DECREMENT,
        PSIFLX,
        THTFLX_30,
        GRAV,
        THTVLX,
        PHILIMX,
        0,
    )
    assert expected["wp_grdrange"] < 2 * expected["rad_min"]
    assert expected["wp_flag"] == -1

    guidance.execute(vehicle, _ctx())

    assert vehicle.store.get("wp_flag") == -1
    assert vehicle.store.get("write") == 0


def test_cadac_sign_zero_horizontal_dot_is_plus_one():
    swel1, swel2, swel3 = 0.0, 100.0, 0.0
    vehicle, guidance = _ready(swel1=swel1, swel2=swel2, swel3=swel3)
    _, expected = _expected_line(
        np.array([swel1, swel2, swel3]),
        SBEL,
        VBEL,
        LINE_GAIN,
        NL_GAIN_FACT,
        DECREMENT,
        PSIFLX,
        THTFLX_30,
        GRAV,
        THTVLX,
        PHILIMX,
        0,
    )
    assert expected["wp_grdrange"] < 2 * expected["rad_min"]
    assert expected["wp_flag"] == 1

    guidance.execute(vehicle, _ctx())

    assert vehicle.store.get("wp_flag") == 1


def test_mguidance_0_returns_without_writing():
    vehicle, guidance = _ready(mguidance=0, ancomx=1.5, alcomx=0.5, phicx=12.0)
    store = vehicle.store
    store.set("wp_sltrange", 1.0)
    store.set("wp_grdrange", 2.0)
    store.set("rad_min", 3.0)
    store.set("wp_flag", 5)
    store.set("write", 7)
    store.set("nl_gain", 0.25)
    store.set("VBEO", np.array([1.0, 2.0, 3.0]))
    store.set("VBEF", np.array([7.0, 8.0, 9.0]))
    store.set("SWBL", np.array([4.0, 5.0, 6.0]))

    guidance.execute(vehicle, _ctx())

    assert store.get("ancomx") == 1.5
    assert store.get("alcomx") == 0.5
    assert store.get("phicx") == 12.0
    assert store.get("wp_sltrange") == 1.0
    assert store.get("wp_grdrange") == 2.0
    assert store.get("rad_min") == 3.0
    assert store.get("wp_flag") == 5
    assert store.get("write") == 7
    assert store.get("nl_gain") == 0.25
    np.testing.assert_array_equal(store.get("VBEO"), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(store.get("VBEF"), np.array([7.0, 8.0, 9.0]))
    np.testing.assert_array_equal(store.get("SWBL"), np.array([4.0, 5.0, 6.0]))
    assert store.get("mguidance") == 0


@pytest.mark.parametrize("mguidance", [3, 43, 99])
def test_unknown_mguidance_raises_valueerror(mguidance):
    vehicle, guidance = _ready(mguidance=mguidance)
    with pytest.raises(ValueError):
        guidance.execute(vehicle, _ctx())


def test_guidance_line_one_step_matches_cpp_equations():
    vehicle, guidance = _ready()
    expected_algv, expected = _expected_line(
        np.array([SWEL1, SWEL2, SWEL3]),
        SBEL,
        VBEL,
        LINE_GAIN,
        NL_GAIN_FACT,
        DECREMENT,
        PSIFLX,
        THTFLX_30,
        GRAV,
        THTVLX,
        PHILIMX,
        0,
    )

    got = guidance.guidance_line(vehicle)

    store = vehicle.store
    np.testing.assert_allclose(got, expected_algv)
    assert store.get("wp_sltrange") == expected["wp_sltrange"]
    assert store.get("nl_gain") == expected["nl_gain"]
    np.testing.assert_allclose(store.get("VBEO"), expected["VBEO"])
    np.testing.assert_allclose(store.get("VBEF"), expected["VBEF"])
    assert store.get("wp_grdrange") == expected["wp_grdrange"]
    np.testing.assert_allclose(store.get("SWBL"), expected["SWBL"])
    assert store.get("rad_min") == expected["rad_min"]
    assert store.get("write") == expected["write"]
    assert store.get("wp_flag") == expected["wp_flag"]
    assert store.get("ancomx") == 0.0
    assert store.get("alcomx") == 0.0


def test_nl_gain_is_range_exponential_not_constant_factor():
    swel1, swel2, swel3 = 500.0, 0.0, -3500.0
    vehicle, guidance = _ready(swel1=swel1, swel2=swel2, swel3=swel3)
    swel = np.array([swel1, swel2, swel3])
    swbl = swel - SBEL
    wp_sltrange = polar_from_cart(swbl)[0]
    expected_nl = NL_GAIN_FACT * (1 - math.exp(-wp_sltrange / DECREMENT))
    assert abs(expected_nl - NL_GAIN_FACT) > 0.05

    guidance.guidance_line(vehicle)

    assert vehicle.store.get("nl_gain") == pytest.approx(expected_nl)
    assert vehicle.store.get("nl_gain") != pytest.approx(NL_GAIN_FACT)


def test_mguidance_30_writes_clipped_alcomx_zero_ancomx_unchanged_phicx():
    phicx = 12.0
    allimx = 0.01
    vehicle, guidance = _ready(mguidance=30, phicx=phicx, allimx=allimx, ancomx=1.5)
    algv, _ = _expected_line(
        np.array([SWEL1, SWEL2, SWEL3]),
        SBEL,
        VBEL,
        LINE_GAIN,
        NL_GAIN_FACT,
        DECREMENT,
        PSIFLX,
        THTFLX_30,
        GRAV,
        THTVLX,
        PHILIMX,
        0,
    )
    alcomx = algv[1] / GRAV
    if alcomx > allimx:
        alcomx = allimx
    if alcomx < -allimx:
        alcomx = -allimx
    ancomx = 0.0
    if ancomx > ANPOSLIMX:
        ancomx = ANPOSLIMX
    if ancomx < ANNEGLIMX:
        ancomx = ANNEGLIMX
    assert abs(algv[1] / GRAV) > allimx

    guidance.execute(vehicle, _ctx())

    store = vehicle.store
    assert store.get("alcomx") == alcomx
    assert store.get("ancomx") == ancomx
    assert store.get("phicx") == phicx


def test_mguidance_33_writes_clipped_alcomx_and_ancomx_from_algv():
    phicx = 12.0
    allimx = 0.01
    anposlimx = 0.01
    anneglimx = -0.01
    vehicle, guidance = _ready(
        mguidance=33,
        thtflx=THTFLX_33,
        phicx=phicx,
        allimx=allimx,
        anposlimx=anposlimx,
        anneglimx=anneglimx,
        ancomx=1.5,
    )
    algv, _ = _expected_line(
        np.array([SWEL1, SWEL2, SWEL3]),
        SBEL,
        VBEL,
        LINE_GAIN,
        NL_GAIN_FACT,
        DECREMENT,
        PSIFLX,
        THTFLX_33,
        GRAV,
        THTVLX,
        PHILIMX,
        0,
    )
    alcomx = algv[1] / GRAV
    if alcomx > allimx:
        alcomx = allimx
    if alcomx < -allimx:
        alcomx = -allimx
    ancomx = -algv[2] / GRAV
    if ancomx > anposlimx:
        ancomx = anposlimx
    if ancomx < anneglimx:
        ancomx = anneglimx
    assert abs(algv[1] / GRAV) > allimx
    assert abs(-algv[2] / GRAV) > anposlimx

    guidance.execute(vehicle, _ctx())

    store = vehicle.store
    assert store.get("alcomx") == alcomx
    assert store.get("ancomx") == ancomx
    assert store.get("phicx") == phicx


def test_mguidance_33_ancomx_uses_neg_algv3_over_grav():
    vehicle, guidance = _ready(
        mguidance=33,
        thtflx=THTFLX_33,
        allimx=100.0,
        anposlimx=100.0,
        anneglimx=-100.0,
    )
    algv, _ = _expected_line(
        np.array([SWEL1, SWEL2, SWEL3]),
        SBEL,
        VBEL,
        LINE_GAIN,
        NL_GAIN_FACT,
        DECREMENT,
        PSIFLX,
        THTFLX_33,
        GRAV,
        THTVLX,
        PHILIMX,
        0,
    )
    expected_ancomx = -algv[2] / GRAV
    expected_alcomx = algv[1] / GRAV
    assert expected_ancomx != 0.0
    assert expected_ancomx != algv[2] / GRAV

    guidance.execute(vehicle, _ctx())

    store = vehicle.store
    assert store.get("ancomx") == pytest.approx(expected_ancomx)
    assert store.get("alcomx") == pytest.approx(expected_alcomx)


def test_mguidance_40_still_point_guidance():
    vehicle, guidance = _ready(mguidance=40, ancomx=1.5)
    guidance.execute(vehicle, _ctx())
    store = vehicle.store
    assert math.isfinite(store.get("alcomx"))
    assert store.get("alcomx") != 0.0
    assert store.get("ancomx") == 0.0


def test_write_stays_set_when_fleeting_after_close():
    vehicle, guidance = _ready(swel1=-200.0, swel2=50.0, swel3=0.0, write=1)
    guidance.execute(vehicle, _ctx())
    assert vehicle.store.get("wp_flag") == -1
    assert vehicle.store.get("write") == 1
