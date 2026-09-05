import math

from cadac.constants import RAD
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.plane5.control import Plane5Control

# input_mcontrol_11 (not turning-to-IP)
GAIN_PSIVG = 12.0
GAIN_THTVG = 30.0
PSIVLCX = 10.0
THTVGCX = 5.0
ALPPOSLIMX = 15.0
ALPNEGLIMX = -10.0
MASS = 12701.0
AREA = 27.87
PDYNMC = 17000.0
CLA = 0.08
GRAV = 9.81
PHIMVX = 0.0
PSIVLX = 0.0
THTVL = 0.0
INT_STEP = 0.05

HEADING_FIELDS = {
    "psivlcx": ("real", "data", ("plot",)),
    "thtvgcx": ("real", "data", ("plot",)),
    "gain_thtvg": ("real", "data", ()),
    "gain_psivg": ("real", "data", ()),
    "avx": ("real", "diag", ("scrn", "plot")),
}


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _expected_heading(psivlcx, psivlx, gain_psivg):
    if abs(psivlcx) <= 135:
        psivgx_comp = psivlx
    else:
        if psivlx * psivlcx >= 0:
            psivgx_comp = psivlx
        else:
            if psivlx >= 0:
                sign_psivgx = 1
            else:
                sign_psivgx = -1
            psivgx_comp = 360 - psivlx * sign_psivgx
    return gain_psivg * (psivlcx - psivgx_comp)


def _expected_flightpath(
    thtvgcx,
    phimvx,
    thtvl,
    gain_thtvg,
    alpposlimx,
    alpneglimx,
    pdynmc,
    grav,
    mass,
    area,
    cla,
):
    avx = gain_thtvg * (thtvgcx * RAD - thtvl)
    anx = avx / math.cos(phimvx * RAD)
    alphax = (anx * mass * grav) / (pdynmc * area * cla)
    if alphax > alpposlimx:
        alphax = alpposlimx
    if alphax < alpneglimx:
        alphax = alpneglimx
    return alphax, anx, avx


def _ready(
    psivlcx=PSIVLCX,
    thtvgcx=THTVGCX,
    gain_psivg=GAIN_PSIVG,
    gain_thtvg=GAIN_THTVG,
    alpposlimx=ALPPOSLIMX,
    alpneglimx=ALPNEGLIMX,
    psivlx=PSIVLX,
    thtvl=THTVL,
    pdynmc=PDYNMC,
    grav=GRAV,
    mass=MASS,
    area=AREA,
    cla=CLA,
    phimvx=PHIMVX,
    alphax=5.0,
    anx=1.5,
):
    vehicle = _Vehicle()
    control = Plane5Control()
    control.define(vehicle)
    store = vehicle.store
    store.define(Field("psivlx", psivlx, "real", "init/diag", "newton", ("scrn", "plot")))
    store.define(Field("thtvl", thtvl, "real", "out", "newton"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("mass", mass, "real", "out", "propulsion"))
    store.define(Field("area", area, "real", "data", "aerodynamics"))
    store.define(Field("cla", cla, "real", "out", "aerodynamics"))
    store.set("psivlcx", psivlcx)
    store.set("thtvgcx", thtvgcx)
    store.set("gain_psivg", gain_psivg)
    store.set("gain_thtvg", gain_thtvg)
    store.set("alpposlimx", alpposlimx)
    store.set("alpneglimx", alpneglimx)
    store.set("phimvx", phimvx)
    store.set("alphax", alphax)
    store.set("anx", anx)
    return vehicle, control


def test_define_registers_heading_flightpath_fields():
    vehicle = _Vehicle()
    Plane5Control().define(vehicle)
    store = vehicle.store
    for name, (ftype, role, outputs) in HEADING_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "control"
        assert field.outputs == outputs
        assert store.get(name) == 0.0
    assert "alphax" in store.names()
    assert "anx" in store.names()
    assert "alpposlimx" in store.names()
    assert "alpneglimx" in store.names()


def test_control_heading_one_step_matches_cpp_equations():
    vehicle, control = _ready()
    expected = _expected_heading(PSIVLCX, PSIVLX, GAIN_PSIVG)

    got = control.control_heading(vehicle, PSIVLCX)

    store = vehicle.store
    assert got == expected
    assert expected == 120.0
    assert store.get("psivlcx") == PSIVLCX
    assert store.get("phimvx") == PHIMVX
    assert store.get("phicx") == 0.0


def test_control_heading_south_wrap_when_command_beyond_135_opposite_sign():
    psivlcx = 180.0
    psivlx = -10.0
    assert abs(psivlcx) > 135
    assert psivlx * psivlcx < 0
    vehicle, control = _ready(psivlcx=psivlcx, psivlx=psivlx)
    wrapped = _expected_heading(psivlcx, psivlx, GAIN_PSIVG)
    unwrapped = GAIN_PSIVG * (psivlcx - psivlx)

    got = control.control_heading(vehicle, psivlcx)

    assert got == wrapped
    assert wrapped != unwrapped
    assert wrapped == GAIN_PSIVG * (psivlcx - 350.0)


def test_control_heading_same_sign_beyond_135_does_not_wrap():
    psivlcx = 180.0
    psivlx = 170.0
    vehicle, control = _ready(psivlcx=psivlcx, psivlx=psivlx)
    expected = _expected_heading(psivlcx, psivlx, GAIN_PSIVG)
    wrapped = GAIN_PSIVG * (psivlcx - (360 - abs(psivlx)))

    got = control.control_heading(vehicle, psivlcx)

    assert got == expected
    assert expected == GAIN_PSIVG * (psivlcx - psivlx)
    assert expected != wrapped


def test_control_flightpath_one_step_matches_cpp_equations():
    thtvl = 2.0 * RAD
    vehicle, control = _ready(thtvl=thtvl)
    expected_alphax, expected_anx, expected_avx = _expected_flightpath(
        THTVGCX,
        PHIMVX,
        thtvl,
        GAIN_THTVG,
        ALPPOSLIMX,
        ALPNEGLIMX,
        PDYNMC,
        GRAV,
        MASS,
        AREA,
        CLA,
    )
    wrong_deg_thtvl, _, _ = _expected_flightpath(
        THTVGCX,
        PHIMVX,
        2.0,
        GAIN_THTVG,
        ALPPOSLIMX,
        ALPNEGLIMX,
        PDYNMC,
        GRAV,
        MASS,
        AREA,
        CLA,
    )

    got = control.control_flightpath(vehicle, THTVGCX, PHIMVX)

    store = vehicle.store
    assert got == expected_alphax
    assert expected_alphax != wrong_deg_thtvl
    assert store.get("anx") == expected_anx
    assert store.get("avx") == expected_avx
    assert store.get("thtvgcx") == THTVGCX
    assert store.get("alphax") == 5.0


def test_control_flightpath_clips_alphax_to_alpposlimx():
    thtvgcx = 20.0
    vehicle, control = _ready(thtvgcx=thtvgcx)
    clipped, anx, avx = _expected_flightpath(
        thtvgcx,
        PHIMVX,
        THTVL,
        GAIN_THTVG,
        ALPPOSLIMX,
        ALPNEGLIMX,
        PDYNMC,
        GRAV,
        MASS,
        AREA,
        CLA,
    )
    unlimited, _, _ = _expected_flightpath(
        thtvgcx,
        PHIMVX,
        THTVL,
        GAIN_THTVG,
        1e9,
        ALPNEGLIMX,
        PDYNMC,
        GRAV,
        MASS,
        AREA,
        CLA,
    )

    got = control.control_flightpath(vehicle, thtvgcx, PHIMVX)

    assert got == clipped
    assert clipped == ALPPOSLIMX
    assert unlimited > ALPPOSLIMX
    assert vehicle.store.get("anx") == anx
    assert vehicle.store.get("avx") == avx


def test_control_flightpath_banked_uses_rad_in_cos():
    phimvx = 30.0
    vehicle, control = _ready(phimvx=phimvx)
    expected_alphax, expected_anx, expected_avx = _expected_flightpath(
        THTVGCX,
        phimvx,
        THTVL,
        GAIN_THTVG,
        ALPPOSLIMX,
        ALPNEGLIMX,
        PDYNMC,
        GRAV,
        MASS,
        AREA,
        CLA,
    )
    without_rad, _, _ = _expected_flightpath(
        THTVGCX,
        phimvx / RAD,
        THTVL,
        GAIN_THTVG,
        ALPPOSLIMX,
        ALPNEGLIMX,
        PDYNMC,
        GRAV,
        MASS,
        AREA,
        CLA,
    )

    got = control.control_flightpath(vehicle, THTVGCX, phimvx)

    assert got == expected_alphax
    assert expected_alphax != without_rad
    assert vehicle.store.get("anx") == expected_anx
    assert vehicle.store.get("avx") == expected_avx


def test_control_flightpath_does_not_write_alphax():
    vehicle, control = _ready(alphax=5.0)
    alphax_before = vehicle.store.get("alphax")

    alphax = control.control_flightpath(vehicle, THTVGCX, PHIMVX)

    assert alphax != alphax_before
    assert vehicle.store.get("alphax") == alphax_before


def test_control_bank_does_not_write_heading_states():
    phicx = 30.0
    vehicle, control = _ready()
    store = vehicle.store
    store.set("phicx", phicx)
    store.set("philimx", 70.0)
    store.set("tphi", 1.0)
    alphax_before = store.get("alphax")
    anx_before = store.get("anx")
    avx_before = store.get("avx")
    phimvx_before = store.get("phimvx")

    phix = control.control_bank(vehicle, phicx, INT_STEP)

    assert phix == 0.75
    assert store.get("phix") == 0.75
    assert store.get("phimvx") == phimvx_before
    assert store.get("alphax") == alphax_before
    assert store.get("anx") == anx_before
    assert store.get("avx") == avx_before
