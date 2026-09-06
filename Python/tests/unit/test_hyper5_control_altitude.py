import math

import numpy as np
import pytest

from cadac.constants import RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.hyper5.control import Hyper5Control

# Demo 5.1 fly-out
GH = 0.2
GV = 0.3
ALTDLIM = 50.0
ALTCOM = 24000.0
ANPOSLIMX = 2.0
ANNEGLIMX = -1.0
ALT = 23900.0
GRAV = 9.81
VBEG = np.array([1475.0, 0.0, 0.0])
PHIMVX = 0.0
INT_STEP = 0.05
ANCOMX = 1.5

RTOL = 1e-12
ATOL = 1e-14

ALTITUDE_FIELDS = {
    "altdlim": ("real", "data", ()),
    "gh": ("real", "data", ()),
    "gv": ("real", "data", ()),
    "altd": ("real", "diag", ("plot",)),
    "altcom": ("real", "data", ("plot",)),
}

LATER_FIELDS = (
    "mcontrol",
    "alcomx",
    "TBV",
    "TBG",
    "gain_psivg",
    "gain_thtvg",
    "psivgcx",
    "thtvgcx",
    "allimx",
    "gcp",
    "alx",
    "avx",
)

PLANT_FIELDS = ("alt", "grav", "VBEG", "VBEL")


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _expected_altitude(
    altcom,
    phimvx,
    alt,
    grav,
    vbeg,
    gh,
    gv,
    altdlim,
    anposlimx,
    anneglimx,
):
    ealt = gh * (altcom - alt)
    if ealt > altdlim:
        ealt = altdlim
    if ealt < -altdlim:
        ealt = -altdlim
    altd = -vbeg[2]
    ancomx = (gv * (ealt - altd) / grav + 1) * (1 / math.cos(phimvx * RAD))
    if ancomx > anposlimx:
        ancomx = anposlimx
    if ancomx < anneglimx:
        ancomx = anneglimx
    return ancomx, altd


def _ctx(int_step=INT_STEP):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _ready(
    alt=ALT,
    grav=GRAV,
    vbeg=VBEG,
    altcom=ALTCOM,
    gh=GH,
    gv=GV,
    altdlim=ALTDLIM,
    anposlimx=ANPOSLIMX,
    anneglimx=ANNEGLIMX,
    phimvx=PHIMVX,
    ancomx=ANCOMX,
):
    vehicle = _Vehicle()
    control = Hyper5Control()
    control.define(vehicle)
    store = vehicle.store
    store.define(Field("alt", alt, "real", "out", "newton", ("scrn", "plot")))
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("VBEG", vbeg, "vec", "state", "newton"))
    store.set("altcom", altcom)
    store.set("gh", gh)
    store.set("gv", gv)
    store.set("altdlim", altdlim)
    store.set("anposlimx", anposlimx)
    store.set("anneglimx", anneglimx)
    store.set("ancomx", ancomx)
    store.set("phimvx", phimvx)
    return vehicle, control


def test_define_registers_altitude_fields():
    vehicle = _Vehicle()
    Hyper5Control().define(vehicle)
    store = vehicle.store
    for name, (ftype, role, outputs) in ALTITUDE_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "control"
        assert field.outputs == outputs
        assert store.get(name) == 0.0
    for name in LATER_FIELDS:
        assert name not in store.names()
    for name in PLANT_FIELDS:
        assert name not in store.names()


def test_execute_is_pass():
    vehicle, control = _ready()
    store = vehicle.store
    names = list(ALTITUDE_FIELDS) + ["ancomx", "phimvx", "anposlimx", "anneglimx"]
    before = {name: store.get(name) for name in names}
    before["alt"] = store.get("alt")
    before["grav"] = store.get("grav")
    before["VBEG"] = store.get("VBEG").copy()

    control.execute(vehicle, _ctx())

    for name, value in before.items():
        if name == "VBEG":
            np.testing.assert_array_equal(store.get(name), value)
        else:
            assert store.get(name) == value


def test_control_altitude_one_step_matches_cpp_equations():
    vehicle, control = _ready()
    expected_ancomx, expected_altd = _expected_altitude(
        ALTCOM,
        PHIMVX,
        ALT,
        GRAV,
        VBEG,
        GH,
        GV,
        ALTDLIM,
        ANPOSLIMX,
        ANNEGLIMX,
    )

    got = control.control_altitude(vehicle, ALTCOM, PHIMVX)

    store = vehicle.store
    assert _approx(got, expected_ancomx)
    assert _approx(store.get("altd"), expected_altd)
    assert expected_altd == 0.0
    assert _approx(expected_ancomx, 1 + 6.0 / GRAV)
    assert store.get("altcom") == ALTCOM
    assert store.get("ancomx") == ANCOMX
    assert "VBEL" not in store.names()


def test_altitude_rate_limiter_clips_when_error_exceeds_altdlim():
    alt = 24300.0
    vbeg = np.array([1475.0, 0.0, 40.0])
    assert abs(GH * (ALTCOM - alt)) > ALTDLIM
    vehicle, control = _ready(alt=alt, vbeg=vbeg)
    limited, altd = _expected_altitude(
        ALTCOM,
        PHIMVX,
        alt,
        GRAV,
        vbeg,
        GH,
        GV,
        ALTDLIM,
        ANPOSLIMX,
        ANNEGLIMX,
    )
    unlimited, _ = _expected_altitude(
        ALTCOM,
        PHIMVX,
        alt,
        GRAV,
        vbeg,
        GH,
        GV,
        1e9,
        ANPOSLIMX,
        ANNEGLIMX,
    )

    got = control.control_altitude(vehicle, ALTCOM, PHIMVX)

    assert _approx(got, limited)
    assert limited != pytest.approx(unlimited, rel=RTOL, abs=ATOL)
    assert _approx(vehicle.store.get("altd"), altd)
    assert altd == -40.0


def test_banked_one_step_uses_rad_in_cos():
    phimvx = 30.0
    vehicle, control = _ready(phimvx=phimvx)
    expected_ancomx, expected_altd = _expected_altitude(
        ALTCOM,
        phimvx,
        ALT,
        GRAV,
        VBEG,
        GH,
        GV,
        ALTDLIM,
        ANPOSLIMX,
        ANNEGLIMX,
    )
    without_rad, _ = _expected_altitude(
        ALTCOM,
        phimvx / RAD,
        ALT,
        GRAV,
        VBEG,
        GH,
        GV,
        ALTDLIM,
        ANPOSLIMX,
        ANNEGLIMX,
    )

    got = control.control_altitude(vehicle, ALTCOM, phimvx)

    assert _approx(got, expected_ancomx)
    assert expected_ancomx != pytest.approx(without_rad, rel=RTOL, abs=ATOL)
    assert _approx(vehicle.store.get("altd"), expected_altd)


def test_control_altitude_does_not_write_ancomx():
    vehicle, control = _ready(ancomx=ANCOMX)
    ancomx_before = vehicle.store.get("ancomx")

    ancomx = control.control_altitude(vehicle, ALTCOM, PHIMVX)

    assert ancomx != pytest.approx(ancomx_before, rel=RTOL, abs=ATOL)
    assert vehicle.store.get("ancomx") == ancomx_before


def test_ancomx_clipped_to_anposlimx():
    vbeg = np.array([1475.0, 0.0, 200.0])
    vehicle, control = _ready(vbeg=vbeg)
    expected, altd = _expected_altitude(
        ALTCOM,
        PHIMVX,
        ALT,
        GRAV,
        vbeg,
        GH,
        GV,
        ALTDLIM,
        ANPOSLIMX,
        ANNEGLIMX,
    )
    unlimited, _ = _expected_altitude(
        ALTCOM,
        PHIMVX,
        ALT,
        GRAV,
        vbeg,
        GH,
        GV,
        ALTDLIM,
        1e9,
        ANNEGLIMX,
    )

    got = control.control_altitude(vehicle, ALTCOM, PHIMVX)

    assert _approx(got, expected)
    assert expected == ANPOSLIMX
    assert unlimited > ANPOSLIMX
    assert _approx(vehicle.store.get("altd"), altd)
    assert vehicle.store.get("ancomx") == ANCOMX


def test_control_bank_does_not_write_altitude_states():
    phicx = 30.0
    vehicle, control = _ready()
    store = vehicle.store
    store.set("phicx", phicx)
    store.set("philimx", 70.0)
    store.set("tphi", 1.0)
    ancomx_before = store.get("ancomx")
    altd_before = store.get("altd")
    phimvx_before = store.get("phimvx")

    phix = control.control_bank(vehicle, phicx, INT_STEP)

    assert _approx(phix, 0.75)
    assert _approx(store.get("phix"), 0.75)
    assert store.get("phimvx") == phimvx_before
    assert store.get("ancomx") == ancomx_before
    assert store.get("altd") == altd_before
    assert store.get("altd") == 0.0
    assert store.get("altcom") == ALTCOM
