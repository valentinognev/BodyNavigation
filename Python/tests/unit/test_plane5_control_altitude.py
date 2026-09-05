import math

import numpy as np

from cadac.constants import RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.plane5.control import Plane5Control

# turning_to_IP
GH = 0.3
GV = 1.0
ALTDLIM = 50.0
ALTCOM = 3000.0
ANPOSLIMX = 3.0
ANNEGLIMX = -1.0
ALT = 3500.0
GRAV = 9.81
VBEL = np.array([200.0, 0.0, 0.0])
PHIMVX = 0.0
INT_STEP = 0.05

ALTITUDE_FIELDS = {
    "altdlim": ("real", "data", ()),
    "gh": ("real", "data", ()),
    "gv": ("real", "data", ()),
    "altd": ("real", "diag", ("plot",)),
    "altcom": ("real", "data", ("plot",)),
}

NOT_YET = (
    "mcontrol",
    "TBV",
    "alcomx",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(int_step=INT_STEP):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _expected_altitude(
    altcom,
    phimvx,
    alt,
    grav,
    vbel,
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
    altd = -vbel[2]
    ancomx = (gv * (ealt - altd) / grav + 1) * (1 / math.cos(phimvx * RAD))
    if ancomx > anposlimx:
        ancomx = anposlimx
    if ancomx < anneglimx:
        ancomx = anneglimx
    return ancomx, altd


def _ready(
    alt=ALT,
    grav=GRAV,
    vbel=VBEL,
    altcom=ALTCOM,
    gh=GH,
    gv=GV,
    altdlim=ALTDLIM,
    anposlimx=ANPOSLIMX,
    anneglimx=ANNEGLIMX,
    phimvx=PHIMVX,
    ancomx=1.5,
):
    vehicle = _Vehicle()
    control = Plane5Control()
    control.define(vehicle)
    store = vehicle.store
    store.define(Field("alt", alt, "real", "diag", "newton", ("scrn", "plot")))
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("VBEL", vbel, "vec", "state", "newton"))
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
    Plane5Control().define(vehicle)
    store = vehicle.store
    for name, (ftype, role, outputs) in ALTITUDE_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "control"
        assert field.outputs == outputs
        assert store.get(name) == 0.0
    for name in NOT_YET:
        assert name not in store.names()


def test_control_altitude_one_step_matches_cpp_equations():
    vehicle, control = _ready()
    expected_ancomx, expected_altd = _expected_altitude(
        ALTCOM,
        PHIMVX,
        ALT,
        GRAV,
        VBEL,
        GH,
        GV,
        ALTDLIM,
        ANPOSLIMX,
        ANNEGLIMX,
    )

    got = control.control_altitude(vehicle, ALTCOM, PHIMVX)

    store = vehicle.store
    assert got == expected_ancomx
    assert store.get("altd") == expected_altd
    assert expected_altd == 0.0
    assert store.get("altcom") == ALTCOM
    assert store.get("ancomx") == 1.5


def test_altitude_rate_limiter_clips_when_error_exceeds_altdlim():
    vbel = np.array([200.0, 0.0, 40.0])
    assert abs(GH * (ALTCOM - ALT)) > ALTDLIM
    vehicle, control = _ready(vbel=vbel)
    limited, altd = _expected_altitude(
        ALTCOM,
        PHIMVX,
        ALT,
        GRAV,
        vbel,
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
        vbel,
        GH,
        GV,
        1e9,
        ANPOSLIMX,
        ANNEGLIMX,
    )

    got = control.control_altitude(vehicle, ALTCOM, PHIMVX)

    assert got == limited
    assert limited != unlimited
    assert vehicle.store.get("altd") == altd
    assert altd == -40.0


def test_banked_one_step_uses_rad_in_cos():
    phimvx = 30.0
    alt = 3040.0
    vehicle, control = _ready(alt=alt, phimvx=phimvx)
    expected_ancomx, expected_altd = _expected_altitude(
        ALTCOM,
        phimvx,
        alt,
        GRAV,
        VBEL,
        GH,
        GV,
        ALTDLIM,
        ANPOSLIMX,
        ANNEGLIMX,
    )
    without_rad, _ = _expected_altitude(
        ALTCOM,
        phimvx / RAD,
        alt,
        GRAV,
        VBEL,
        GH,
        GV,
        ALTDLIM,
        ANPOSLIMX,
        ANNEGLIMX,
    )

    got = control.control_altitude(vehicle, ALTCOM, phimvx)

    assert got == expected_ancomx
    assert expected_ancomx != without_rad
    assert vehicle.store.get("altd") == expected_altd


def test_control_altitude_does_not_write_ancomx():
    vehicle, control = _ready(ancomx=1.5)
    ancomx_before = vehicle.store.get("ancomx")

    ancomx = control.control_altitude(vehicle, ALTCOM, PHIMVX)

    assert ancomx != ancomx_before
    assert vehicle.store.get("ancomx") == ancomx_before


def test_execute_still_bank_wrap_only():
    phicx = 30.0
    vehicle, control = _ready()
    store = vehicle.store
    store.set("phicx", phicx)
    store.set("philimx", 70.0)
    store.set("tphi", 1.0)
    ancomx_before = store.get("ancomx")
    altd_before = store.get("altd")

    control.execute(vehicle, _ctx())

    assert store.get("phimvx") == 0.75
    assert store.get("phix") == 0.75
    assert store.get("ancomx") == ancomx_before
    assert store.get("altd") == altd_before
    assert store.get("altd") == 0.0
