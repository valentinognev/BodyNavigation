import math

import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import cadtbv
from cadac.vehicles.plane5.control import Plane5Control

# turning_to_IP
ALCOMX = 0.5
GCP = 2.0
ALLIMX = 1.0
PHILIMX = 70.0
TPHI = 1.0
ALTCOM = 3000.0
ALTDLIM = 50.0
GH = 0.3
GV = 1.0
ANPOSLIMX = 3.0
ANNEGLIMX = -1.0
GACP = 10.0
TA = 0.8
ALPPOSLIMX = 15.0
ALPNEGLIMX = -10.0
MASS = 12701.0
DVBE = 200.0
AREA = 27.87
PDYNMC = 17000.0
THRUST = 20000.0
CLA = 0.08
GRAV = 9.81
FSPV = np.array([2.0, 1.0, -12.0])
ALT = 3500.0
VBEL = np.array([200.0, 0.0, 0.0])
INT_STEP = 0.05
ANCOMX_44 = 1.5

LATERAL_FIELDS = {
    "mcontrol": ("int", "data", ("scrn",)),
    "TBV": ("mat", "out", ()),
    "alcomx": ("real", "data", ("plot",)),
    "allimx": ("real", "data", ()),
    "gcp": ("real", "data", ()),
    "alx": ("real", "diag", ("plot",)),
}


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


def _expected_lateral(alcomx, allimx, gcp, phimvx, alphax, fspv, grav):
    alpha = alphax * RAD
    phimv = phimvx * RAD
    tbv = cadtbv(phimv, alpha)
    fspb = tbv @ fspv
    anx = -fspb[2] / grav
    if alcomx > allimx:
        alcomx = allimx
    if alcomx < -allimx:
        alcomx = -allimx
    sign = 1 if anx >= 0 else -1
    phic = gcp * sign / (math.fabs(anx) + 0.001) * alcomx
    phicx = phic * DEG
    alx = fspv[1] / grav
    return phicx, alx


def _ready(
    mcontrol=46,
    alcomx=ALCOMX,
    gcp=GCP,
    allimx=ALLIMX,
    philimx=PHILIMX,
    tphi=TPHI,
    altcom=ALTCOM,
    altdlim=ALTDLIM,
    gh=GH,
    gv=GV,
    anposlimx=ANPOSLIMX,
    anneglimx=ANNEGLIMX,
    gacp=GACP,
    ta=TA,
    alpposlimx=ALPPOSLIMX,
    alpneglimx=ALPNEGLIMX,
    fspv=FSPV,
    grav=GRAV,
    mass=MASS,
    dvbe=DVBE,
    pdynmc=PDYNMC,
    thrust=THRUST,
    area=AREA,
    cla=CLA,
    alt=ALT,
    vbel=VBEL,
    alphax=0.0,
    phimvx=0.0,
    phicx=0.0,
    ancomx=0.0,
):
    vehicle = _Vehicle()
    control = Plane5Control()
    control.define(vehicle)
    store = vehicle.store
    store.define(Field("FSPV", fspv, "vec", "out", "forces", ("plot",)))
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("mass", mass, "real", "out", "propulsion"))
    store.define(Field("dvbe", dvbe, "real", "out", "newton"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("thrust", thrust, "real", "out", "propulsion"))
    store.define(Field("area", area, "real", "data", "aerodynamics"))
    store.define(Field("cla", cla, "real", "out", "aerodynamics"))
    store.define(Field("alt", alt, "real", "diag", "newton", ("scrn", "plot")))
    store.define(Field("VBEL", vbel, "vec", "state", "newton"))
    store.set("mcontrol", mcontrol)
    store.set("alcomx", alcomx)
    store.set("gcp", gcp)
    store.set("allimx", allimx)
    store.set("philimx", philimx)
    store.set("tphi", tphi)
    store.set("altcom", altcom)
    store.set("altdlim", altdlim)
    store.set("gh", gh)
    store.set("gv", gv)
    store.set("anposlimx", anposlimx)
    store.set("anneglimx", anneglimx)
    store.set("gacp", gacp)
    store.set("ta", ta)
    store.set("alpposlimx", alpposlimx)
    store.set("alpneglimx", alpneglimx)
    store.set("alphax", alphax)
    store.set("phimvx", phimvx)
    store.set("phicx", phicx)
    store.set("ancomx", ancomx)
    return vehicle, control


def test_define_registers_lateral_mcontrol_fields():
    vehicle = _Vehicle()
    Plane5Control().define(vehicle)
    store = vehicle.store
    for name, (ftype, role, outputs) in LATERAL_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "control"
        assert field.outputs == outputs
    assert store.get("mcontrol") == 0
    assert type(store.get("mcontrol")) is int
    np.testing.assert_array_equal(store.get("TBV"), np.zeros((3, 3)))
    assert store.get("alcomx") == 0.0
    assert store.get("allimx") == 0.0
    assert store.get("gcp") == 0.0
    assert store.get("alx") == 0.0


def test_mcontrol_46_turning_to_ip_produces_finite_ancomx_and_phimvx():
    vehicle, control = _ready(mcontrol=46)
    control.execute(vehicle, _ctx())
    store = vehicle.store
    ancomx = store.get("ancomx")
    phimvx = store.get("phimvx")
    assert math.isfinite(ancomx)
    assert math.isfinite(phimvx)
    assert ancomx != 0.0
    assert phimvx != 0.0


@pytest.mark.parametrize("mcontrol", [0, 3, 6, 16, 99])
def test_unknown_mcontrol_raises_valueerror(mcontrol):
    vehicle, control = _ready(mcontrol=mcontrol)
    with pytest.raises(ValueError):
        control.execute(vehicle, _ctx())


def test_control_lateral_one_step_matches_cpp_equations():
    vehicle, control = _ready()
    expected_phicx, expected_alx = _expected_lateral(
        ALCOMX, ALLIMX, GCP, 0.0, 0.0, FSPV, GRAV
    )

    got = control.control_lateral(vehicle, ALCOMX)

    store = vehicle.store
    assert got == expected_phicx
    assert store.get("alx") == expected_alx
    assert store.get("alcomx") == ALCOMX
    assert store.get("phicx") == 0.0
    np.testing.assert_array_equal(store.get("TBV"), np.zeros((3, 3)))


def test_control_lateral_clips_alcomx_to_allimx():
    alcomx = 2.0
    vehicle, control = _ready(alcomx=alcomx)
    clipped, _ = _expected_lateral(alcomx, ALLIMX, GCP, 0.0, 0.0, FSPV, GRAV)
    unlimited, _ = _expected_lateral(alcomx, alcomx, GCP, 0.0, 0.0, FSPV, GRAV)

    got = control.control_lateral(vehicle, alcomx)

    assert got == clipped
    assert clipped != unlimited
    assert vehicle.store.get("alcomx") == alcomx


def test_control_lateral_negative_anx_flips_sign():
    fspv = np.array([2.0, 1.0, 12.0])
    vehicle, control = _ready(fspv=fspv)
    expected_phicx, expected_alx = _expected_lateral(
        ALCOMX, ALLIMX, GCP, 0.0, 0.0, fspv, GRAV
    )
    positive, _ = _expected_lateral(ALCOMX, ALLIMX, GCP, 0.0, 0.0, FSPV, GRAV)

    got = control.control_lateral(vehicle, ALCOMX)

    assert got == expected_phicx
    assert expected_phicx == -positive
    assert vehicle.store.get("alx") == expected_alx


def test_control_lateral_uses_point_zero_zero_one_in_denominator():
    fspv = np.array([0.0, 1.0, 0.0])
    vehicle, control = _ready(fspv=fspv)
    expected_phicx = GCP / 0.001 * ALCOMX * DEG

    got = control.control_lateral(vehicle, ALCOMX)

    assert got == expected_phicx
    assert math.isfinite(got)


def test_control_lateral_does_not_write_phicx():
    vehicle, control = _ready()
    phicx_before = vehicle.store.get("phicx")

    phicx = control.control_lateral(vehicle, ALCOMX)

    assert phicx != phicx_before
    assert vehicle.store.get("phicx") == phicx_before


def test_mcontrol_46_writes_chain_outputs_and_tbv():
    vehicle, control = _ready(mcontrol=46)
    store = vehicle.store
    phicx_lat, _ = _expected_lateral(ALCOMX, ALLIMX, GCP, 0.0, 0.0, FSPV, GRAV)
    phixd_new = (phicx_lat - 0.0) / TPHI
    phimvx_exp = integrate(phixd_new, 0.0, 0.0, INT_STEP)

    control.execute(vehicle, _ctx())

    assert store.get("phicx") == phicx_lat
    assert store.get("phimvx") == phimvx_exp
    assert math.isfinite(store.get("ancomx"))
    assert store.get("ancomx") != 0.0
    np.testing.assert_allclose(
        store.get("TBV"),
        cadtbv(store.get("phimvx") * RAD, store.get("alphax") * RAD),
    )
    assert store.get("alphax") != 0.0
    assert store.get("alx") == FSPV[1] / GRAV


def test_mcontrol_44_uses_store_ancomx_without_altitude():
    vehicle, control = _ready(mcontrol=44, ancomx=ANCOMX_44)
    store = vehicle.store
    altd_before = store.get("altd")

    control.execute(vehicle, _ctx())

    assert store.get("ancomx") == ANCOMX_44
    assert store.get("altd") == altd_before
    assert math.isfinite(store.get("phimvx"))
    assert store.get("phimvx") != 0.0
    np.testing.assert_allclose(
        store.get("TBV"),
        cadtbv(store.get("phimvx") * RAD, store.get("alphax") * RAD),
    )
    assert store.get("alphax") != 0.0
