import math

import pytest

from cadac.constants import RAD
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.hyper5.control import Hyper5Control

# Demo 5.1 heading gain; Demo 4.7 Roadrunner plant / AoA limits
GAIN_PSIVG = 2.0
GAIN_THTVG = 30.0
ALPPOSLIMX = 6.0
ALPNEGLIMX = -4.0
MASS = 1352.0
AREA = 11.6986
PDYNMC = 72000.0
CLA = 0.08
GRAV = 9.81
PHIMVX = 0.0

RTOL = 1e-12
ATOL = 1e-14


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _expected_heading(psivgcx, psivgx, gain_psivg):
    if abs(psivgcx) <= 135:
        psivgx_comp = psivgx
    else:
        if psivgx * psivgcx >= 0:
            psivgx_comp = psivgx
        else:
            if psivgx >= 0:
                sign_psivgx = 1
            else:
                sign_psivgx = -1
            psivgx_comp = 360 - psivgx * sign_psivgx
    return gain_psivg * (psivgcx - psivgx_comp)


def _expected_flightpath(
    thtvgcx,
    phimvx,
    thtvg,
    gain_thtvg,
    alpposlimx,
    alpneglimx,
    pdynmc,
    grav,
    mass,
    area,
    cla,
):
    avx = gain_thtvg * (thtvgcx * RAD - thtvg)
    anx = avx / math.cos(phimvx * RAD)
    alphax = (anx * mass * grav) / (pdynmc * area * cla)
    if alphax > alpposlimx:
        alphax = alpposlimx
    if alphax < alpneglimx:
        alphax = alpneglimx
    return alphax, anx, avx


def _ready_heading(psivgcx, psivgx, gain_psivg=GAIN_PSIVG, phimvx=PHIMVX, phicx=0.0):
    vehicle = _Vehicle()
    control = Hyper5Control()
    control.define(vehicle)
    store = vehicle.store
    store.define(Field("psivgx", psivgx, "real", "init/out", "newton", ("scrn", "plot", "com")))
    store.set("psivgcx", psivgcx)
    store.set("gain_psivg", gain_psivg)
    store.set("phimvx", phimvx)
    store.set("phicx", phicx)
    return vehicle, control


def _ready_flightpath(
    thtvgcx,
    thtvg,
    phimvx=PHIMVX,
    gain_thtvg=GAIN_THTVG,
    alpposlimx=ALPPOSLIMX,
    alpneglimx=ALPNEGLIMX,
    pdynmc=PDYNMC,
    grav=GRAV,
    mass=MASS,
    area=AREA,
    cla=CLA,
    alphax=5.0,
):
    vehicle = _Vehicle()
    control = Hyper5Control()
    control.define(vehicle)
    store = vehicle.store
    store.define(Field("thtvg", thtvg, "real", "out", "newton"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("mass", mass, "real", "out", "propulsion"))
    store.define(Field("area", area, "real", "data", "aerodynamics"))
    store.define(Field("cla", cla, "real", "out", "aerodynamics"))
    store.set("thtvgcx", thtvgcx)
    store.set("gain_thtvg", gain_thtvg)
    store.set("alpposlimx", alpposlimx)
    store.set("alpneglimx", alpneglimx)
    store.set("phimvx", phimvx)
    store.set("alphax", alphax)
    return vehicle, control


def test_control_heading_south_wrap_when_command_beyond_135_opposite_sign():
    psivgcx = 180.0
    psivgx = -10.0
    assert abs(psivgcx) > 135
    assert psivgx * psivgcx < 0
    vehicle, control = _ready_heading(psivgcx=psivgcx, psivgx=psivgx, phimvx=12.0, phicx=7.0)
    wrapped = _expected_heading(psivgcx, psivgx, GAIN_PSIVG)
    unwrapped = GAIN_PSIVG * (psivgcx - psivgx)

    got = control.control_heading(vehicle, psivgcx)

    store = vehicle.store
    assert _approx(got, wrapped)
    assert wrapped != pytest.approx(unwrapped, rel=RTOL, abs=ATOL)
    assert _approx(wrapped, -340.0)
    assert _approx(wrapped, GAIN_PSIVG * (psivgcx - 350.0))
    assert store.get("phimvx") == 12.0
    assert store.get("phicx") == 7.0
    assert store.get("psivgcx") == psivgcx
    assert "psivlx" not in store.names()


def test_control_flightpath_clips_alphax_to_alpposlimx():
    thtvgcx = 70.0
    thtvg = 2.0 * RAD
    vehicle, control = _ready_flightpath(thtvgcx=thtvgcx, thtvg=thtvg)
    clipped, anx, avx = _expected_flightpath(
        thtvgcx,
        PHIMVX,
        thtvg,
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
        thtvg,
        GAIN_THTVG,
        1e9,
        ALPNEGLIMX,
        PDYNMC,
        GRAV,
        MASS,
        AREA,
        CLA,
    )
    wrong_deg_thtvg, _, _ = _expected_flightpath(
        thtvgcx,
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

    got = control.control_flightpath(vehicle, thtvgcx, PHIMVX)

    store = vehicle.store
    assert _approx(got, clipped)
    assert clipped == ALPPOSLIMX
    assert unlimited > ALPPOSLIMX
    assert clipped != pytest.approx(wrong_deg_thtvg, rel=RTOL, abs=ATOL)
    assert _approx(store.get("anx"), anx)
    assert _approx(store.get("avx"), avx)
    assert store.get("alphax") == 5.0
    assert store.get("thtvgcx") == thtvgcx
    assert "thtvl" not in store.names()
