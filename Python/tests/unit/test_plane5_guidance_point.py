import math

import numpy as np
import pytest

from cadac.constants import RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr, polar_from_cart
from cadac.vehicles.flat3.falcon5.guidance import Plane5Guidance

# turning_to_IP waypoint #1 / origin heading 0
SWEL1 = 5000.0
SWEL2 = 2000.0
SWEL3 = 0.0
POINT_GAIN = 1.0
PHILIMX = 70.0
GRAV = 9.81
THTVLX = 0.0
SBEL = np.array([0.0, 0.0, -3500.0])
VBEL = np.array([200.0, 0.0, 0.0])
ANPOSLIMX = 3.0
ANNEGLIMX = -1.0
ALLIMX = 1.0
PHICX = 0.0

POINT_FIELDS = {
    "mguidance": ("int", "data", ("scrn",)),
    "swel1": ("real", "data", ()),
    "swel2": ("real", "data", ()),
    "swel3": ("real", "data", ()),
    "point_gain": ("real", "data", ()),
    "wp_sltrange": ("real", "dia", ()),
    "VBEO": ("vec", "dia", ()),
    "wp_grdrange": ("real", "dia", ("scrn", "plot")),
    "SWBL": ("vec", "out", ()),
    "rad_min": ("real", "dia", ()),
    "write": ("int", "save", ("scrn", "plot")),
    "wp_flag": ("int", "dia", ("plot", "scrn")),
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


def _expected_point(swel, sbel, vbel, point_gain, grav, thtvlx, philimx, write):
    swbl = swel - sbel
    polar = polar_from_cart(swbl)
    wp_sltrange = polar[0]
    psiol = polar[1]
    thtol = polar[2]
    tol = mat2tr(psiol, thtol)
    wp_grdrange = math.hypot(swbl[0], swbl[1])
    vbeo = tol @ vbel
    apgv1 = grav * math.sin(thtvlx * RAD)
    apgv2 = point_gain * (-vbeo[1])
    apgv3 = point_gain * (-vbeo[2]) - grav * math.cos(thtvlx * RAD)
    apgv = np.array([apgv1, apgv2, apgv3])
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
    return apgv, {
        "wp_sltrange": wp_sltrange,
        "VBEO": vbeo,
        "wp_grdrange": wp_grdrange,
        "SWBL": swbl,
        "rad_min": rad_min,
        "write": write,
        "wp_flag": wp_flag,
    }


def _ready(
    mguidance=40,
    swel1=SWEL1,
    swel2=SWEL2,
    swel3=SWEL3,
    point_gain=POINT_GAIN,
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
    store.set("write", write)
    store.set("wp_flag", wp_flag)
    return vehicle, guidance


def test_name_is_guidance():
    assert Plane5Guidance().name == "guidance"


def test_define_registers_point_guidance_fields():
    vehicle = _Vehicle()
    Plane5Guidance().define(vehicle)
    store = vehicle.store
    for name, (ftype, role, outputs) in POINT_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "guidance"
        assert field.outputs == outputs
    assert store.get("mguidance") == 0
    assert type(store.get("mguidance")) is int
    assert store.get("swel1") == 0.0
    assert store.get("swel2") == 0.0
    assert store.get("swel3") == 0.0
    assert store.get("point_gain") == 0.0
    assert store.get("wp_sltrange") == 999999.0
    np.testing.assert_array_equal(store.get("VBEO"), np.zeros(3))
    assert store.get("wp_grdrange") == 999999.0
    np.testing.assert_array_equal(store.get("SWBL"), np.zeros(3))
    assert store.get("rad_min") == 0.0
    assert store.get("write") == 0
    assert type(store.get("write")) is int
    assert store.get("wp_flag") == 0
    assert type(store.get("wp_flag")) is int


def test_mguidance_40_origin_heading_0_swel_5000_2000_finite_commands():
    vehicle, guidance = _ready()
    guidance.execute(vehicle, _ctx())
    store = vehicle.store
    assert math.isfinite(store.get("alcomx"))
    assert math.isfinite(store.get("ancomx"))
    assert store.get("alcomx") != 0.0


def test_wp_flag_zero_outside_two_rad_min_at_origin_ne_waypoint():
    vehicle, guidance = _ready()
    apgv, expected = _expected_point(
        np.array([SWEL1, SWEL2, SWEL3]),
        SBEL,
        VBEL,
        POINT_GAIN,
        GRAV,
        THTVLX,
        PHILIMX,
        0,
    )
    assert expected["wp_grdrange"] >= 2 * expected["rad_min"]
    assert expected["wp_flag"] == 0

    guidance.execute(vehicle, _ctx())

    assert vehicle.store.get("wp_flag") == 0
    assert math.isfinite(apgv[1] / GRAV)


def test_wp_flag_plus_one_when_closing_inside_two_rad_min():
    swel1, swel2, swel3 = 500.0, 200.0, 0.0
    vehicle, guidance = _ready(swel1=swel1, swel2=swel2, swel3=swel3)
    _, expected = _expected_point(
        np.array([swel1, swel2, swel3]),
        SBEL,
        VBEL,
        POINT_GAIN,
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
    _, expected = _expected_point(
        np.array([swel1, swel2, swel3]),
        SBEL,
        VBEL,
        POINT_GAIN,
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
    _, expected = _expected_point(
        np.array([swel1, swel2, swel3]),
        SBEL,
        VBEL,
        POINT_GAIN,
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
    store.set("VBEO", np.array([1.0, 2.0, 3.0]))
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
    np.testing.assert_array_equal(store.get("VBEO"), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(store.get("SWBL"), np.array([4.0, 5.0, 6.0]))
    assert store.get("mguidance") == 0


@pytest.mark.parametrize("mguidance", [43, 99])
def test_unknown_mguidance_raises_valueerror(mguidance):
    vehicle, guidance = _ready(mguidance=mguidance)
    with pytest.raises(ValueError):
        guidance.execute(vehicle, _ctx())


def test_guidance_point_one_step_matches_cpp_equations():
    vehicle, guidance = _ready()
    expected_apgv, expected = _expected_point(
        np.array([SWEL1, SWEL2, SWEL3]),
        SBEL,
        VBEL,
        POINT_GAIN,
        GRAV,
        THTVLX,
        PHILIMX,
        0,
    )

    got = guidance.guidance_point(vehicle)

    store = vehicle.store
    np.testing.assert_allclose(got, expected_apgv)
    assert store.get("wp_sltrange") == expected["wp_sltrange"]
    np.testing.assert_allclose(store.get("VBEO"), expected["VBEO"])
    assert store.get("wp_grdrange") == expected["wp_grdrange"]
    np.testing.assert_allclose(store.get("SWBL"), expected["SWBL"])
    assert store.get("rad_min") == expected["rad_min"]
    assert store.get("write") == expected["write"]
    assert store.get("wp_flag") == expected["wp_flag"]
    assert store.get("ancomx") == 0.0
    assert store.get("alcomx") == 0.0


def test_mguidance_40_writes_clipped_alcomx_zero_ancomx_unchanged_phicx():
    phicx = 12.0
    allimx = 0.01
    vehicle, guidance = _ready(phicx=phicx, allimx=allimx, ancomx=1.5)
    apgv, _ = _expected_point(
        np.array([SWEL1, SWEL2, SWEL3]),
        SBEL,
        VBEL,
        POINT_GAIN,
        GRAV,
        THTVLX,
        PHILIMX,
        0,
    )
    alcomx = apgv[1] / GRAV
    if alcomx > allimx:
        alcomx = allimx
    if alcomx < -allimx:
        alcomx = -allimx
    ancomx = 0.0
    if ancomx > ANPOSLIMX:
        ancomx = ANPOSLIMX
    if ancomx < ANNEGLIMX:
        ancomx = ANNEGLIMX
    assert abs(apgv[1] / GRAV) > allimx

    guidance.execute(vehicle, _ctx())

    store = vehicle.store
    assert store.get("alcomx") == alcomx
    assert store.get("ancomx") == ancomx
    assert store.get("phicx") == phicx


def test_write_stays_set_when_fleeting_after_close():
    vehicle, guidance = _ready(swel1=-200.0, swel2=50.0, swel3=0.0, write=1)
    guidance.execute(vehicle, _ctx())
    assert vehicle.store.get("wp_flag") == -1
    assert vehicle.store.get("write") == 1
