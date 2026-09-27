from math import cos

import numpy as np
import pytest

from cadac.constants import RAD
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round3.cruise5.control import Cruise5Control

RTOL = 1e-12
ATOL = 1e-14
GH = 0.3
GV = 1.0
ALTDLIM = 50.0
ALTCOM = 7000.0
GRAV = 9.81
ANPOS = 3.0
ANNEG = -1.0
VBEG = np.array([200.0, 0.0, 0.0])
PHIMVX = 0.0


def _expected_alt(altcom, phimvx, alt, vbeg, gh, gv, altdlim, grav, anpos, anneg):
    ealt = gh * (altcom - alt)
    if ealt > altdlim:
        ealt = altdlim
    if ealt < -altdlim:
        ealt = -altdlim
    altd = -vbeg[2]
    ancomx = (gv * (ealt - altd) / grav + 1) * (1 / cos(phimvx * RAD))
    if ancomx > anpos:
        ancomx = anpos
    if ancomx < anneg:
        ancomx = anneg
    return ancomx, altd


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ready(alt=ALTCOM):
    vehicle = type("V", (), {"store": StateStore()})()
    control = Cruise5Control()
    control.define(vehicle)
    store = vehicle.store
    store.define(Field("alt", alt, "real", "init/out", "newton", ("scrn", "plot", "com")))
    store.define(Field("grav", GRAV, "real", "out", "environment"))
    store.define(Field("vbeg", VBEG, "vec", "state", "newton", ("scrn", "plot", "com")))
    store.set("gh", GH)
    store.set("gv", GV)
    store.set("altdlim", ALTDLIM)
    store.set("anposlimx", ANPOS)
    store.set("anneglimx", ANNEG)
    store.set("alt", alt)
    store.set("grav", GRAV)
    store.set("vbeg", VBEG)
    return vehicle, control


@pytest.mark.parametrize("alt", [7000.0, 6900.0, 6000.0])
def test_control_altitude_matches_cpp_replica(alt):
    vehicle, control = _ready(alt=alt)
    expected_ancomx, expected_altd = _expected_alt(
        ALTCOM, PHIMVX, alt, VBEG, GH, GV, ALTDLIM, GRAV, ANPOS, ANNEG,
    )

    got = control.control_altitude(vehicle, 7000.0, 0.0)

    assert _approx(got, expected_ancomx)
    assert _approx(vehicle.store.get("altd"), expected_altd)


def test_control_altitude_does_not_write_ancomx():
    vehicle, control = _ready(alt=7000.0)
    ancomx_before = vehicle.store.get("ancomx")

    ancomx = control.control_altitude(vehicle, 7000.0, 0.0)

    assert ancomx != ancomx_before
    assert vehicle.store.get("ancomx") == ancomx_before
