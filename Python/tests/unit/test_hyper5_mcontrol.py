import math

import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import cadtbv
from cadac.vehicles.round3.hyper5.control import Hyper5Control

# Demo 4.7
ALCOMX = 0.5
GCP = 2.0
ALLIMX = 1.0
PHILIMX = 70.0
TPHI = 1.0
ANPOSLIMX = 2.0
ANNEGLIMX = -2.0
GACP = 10.0
TA = 0.8
ALPPOSLIMX = 6.0
ALPNEGLIMX = -4.0
MASS = 1352.0
DVBE = 254.0
AREA = 11.6986
PDYNMC = 72000.0
THRUST = 0.0
CLA = 0.08
GRAV = 9.81
FSPV = np.array([2.0, 1.0, -12.0])
ALPHAX = -1.5
PHIMVX = 0.0
INT_STEP = 0.05
TGV = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
)

RTOL = 1e-12
ATOL = 1e-14

DISPATCH_FIELDS = {
    "mcontrol": ("int", "data", ("scrn",)),
    "TBV": ("mat", "out", ()),
    "TBG": ("mat", "out", ()),
    "alcomx": ("real", "data", ("scrn", "plot")),
    "allimx": ("real", "data", ()),
    "gcp": ("real", "data", ()),
    "alx": ("real", "diag", ("plot",)),
    "alphacx": ("real", "data", ()),
    "phimvcx": ("real", "data", ()),
}

class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ctx(int_step=INT_STEP):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _expected_lateral(alcomx, allimx, phimvx, alphax, fspv, grav):
    tbv = cadtbv(phimvx * RAD, alphax * RAD)
    fspb = tbv @ fspv
    anx = -fspb[2] / grav
    if alcomx > allimx:
        alcomx = allimx
    if alcomx < -allimx:
        alcomx = -allimx
    phic = math.atan2(alcomx, anx)
    phicx = phic * DEG
    alx = fspv[1] / grav
    return phicx, alx


def _plane5_gcp_phicx(alcomx, allimx, gcp, phimvx, alphax, fspv, grav):
    tbv = cadtbv(phimvx * RAD, alphax * RAD)
    fspb = tbv @ fspv
    anx = -fspb[2] / grav
    if alcomx > allimx:
        alcomx = allimx
    if alcomx < -allimx:
        alcomx = -allimx
    sign = 1 if anx >= 0 else -1
    phic = gcp * sign / (math.fabs(anx) + 0.001) * alcomx
    return phic * DEG


def _expected_load(
    ancomx,
    int_step,
    phimvx,
    alphax,
    anposlimx,
    anneglimx,
    gacp,
    ta,
    alpposlimx,
    alpneglimx,
    fspv,
    grav,
    mass,
    dvbe,
    pdynmc,
    thrust,
    area,
    cla,
    xi,
    xid,
    alp,
    alpd,
):
    tbv = cadtbv(phimvx * RAD, alphax * RAD)
    fspb = tbv @ fspv
    if ancomx > anposlimx:
        ancomx = anposlimx
    if ancomx < anneglimx:
        ancomx = anneglimx
    anx = -fspb[2] / grav
    eanx = ancomx - anx
    tip = dvbe * mass / (pdynmc * area * cla / RAD + thrust)
    gr = 0.0
    if ta > 0:
        gr = gacp * tip / dvbe
        gi = gr / ta
        xid_new = gi * eanx
        xi = integrate(xid_new, xid, xi, int_step)
        xid = xid_new
    else:
        xi = 0.0
    qq = gr * eanx + xi
    alpd_new = qq - alp / tip
    alp = integrate(alpd_new, alpd, alp, int_step)
    alpd = alpd_new
    alpx = alp * DEG
    if alpx > alpposlimx:
        alpx = alpposlimx
    if alpx < alpneglimx:
        alpx = alpneglimx
    return alpx


def _ready(
    mcontrol=44,
    alcomx=ALCOMX,
    gcp=GCP,
    allimx=ALLIMX,
    philimx=PHILIMX,
    tphi=TPHI,
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
    alphax=ALPHAX,
    phimvx=PHIMVX,
    phicx=0.0,
    ancomx=0.0,
    alphacx=0.0,
    tgv=TGV,
):
    vehicle = _Vehicle()
    control = Hyper5Control()
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
    store.define(Field("tgv", tgv, "mat", "init", "newton"))
    store.set("mcontrol", mcontrol)
    store.set("alcomx", alcomx)
    store.set("gcp", gcp)
    store.set("allimx", allimx)
    store.set("philimx", philimx)
    store.set("tphi", tphi)
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
    store.set("alphacx", alphacx)
    return vehicle, control


def test_define_registers_dispatcher_and_lateral_fields():
    vehicle = _Vehicle()
    Hyper5Control().define(vehicle)
    store = vehicle.store
    for name, (ftype, role, outputs) in DISPATCH_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "control"
        assert field.outputs == outputs
    assert store.get("mcontrol") == 0
    assert type(store.get("mcontrol")) is int
    np.testing.assert_array_equal(store.get("TBV"), np.zeros((3, 3)))
    np.testing.assert_array_equal(store.get("TBG"), np.zeros((3, 3)))
    assert store.get("alcomx") == 0.0
    assert store.get("allimx") == 0.0
    assert store.get("gcp") == 0.0
    assert store.get("alx") == 0.0
    assert store.get("alphacx") == 0.0
    assert store.get("phimvcx") == 0.0


def test_define_skips_existing_tbv_and_tbg():
    vehicle = _Vehicle()
    store = vehicle.store
    tbv = ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0))
    tbg = ((4.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 6.0))
    store.define(Field("TBV", tbv, "mat", "diag", "newton"))
    store.define(Field("TBG", tbg, "mat", "diag", "newton"))
    Hyper5Control().define(vehicle)
    np.testing.assert_array_equal(store.get("TBV"), np.diag([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(store.get("TBG"), np.diag([4.0, 5.0, 6.0]))
    assert store.field("TBV").module == "newton"
    assert store.field("TBG").module == "newton"


def test_mcontrol_44_demo_47_alcomx_produces_finite_phimvx_and_alphax():
    vehicle, control = _ready(mcontrol=44, alcomx=ALCOMX)
    control.execute(vehicle, _ctx())
    store = vehicle.store
    phimvx = store.get("phimvx")
    alphax = store.get("alphax")
    assert math.isfinite(phimvx)
    assert math.isfinite(alphax)
    assert phimvx != 0.0
    assert alphax != 0.0


def test_mcontrol_minus_one_raises():
    vehicle, control = _ready(mcontrol=-1)
    with pytest.raises(ValueError):
        control.execute(vehicle, _ctx())


def test_mcontrol_0_zeros_phimvx_and_alphax():
    vehicle, control = _ready(mcontrol=0, phimvx=12.0, alphax=-1.5)
    store = vehicle.store
    phicx_before = store.get("phicx")
    ancomx_before = store.get("ancomx")
    control.execute(vehicle, _ctx())
    assert store.get("phimvx") == 0.0
    assert store.get("alphax") == 0.0
    assert store.get("phicx") == phicx_before
    assert store.get("ancomx") == ancomx_before
    np.testing.assert_allclose(
        store.get("TBV"),
        cadtbv(0.0, 0.0),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        store.get("TBG"),
        cadtbv(0.0, 0.0) @ TGV.T,
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("mcontrol", [1, 10, 11, 46, 99])
def test_unknown_mcontrol_raises_valueerror(mcontrol):
    vehicle, control = _ready(mcontrol=mcontrol)
    with pytest.raises(ValueError):
        control.execute(vehicle, _ctx())


def test_control_lateral_one_step_matches_cpp_atan2():
    vehicle, control = _ready()
    expected_phicx, expected_alx = _expected_lateral(
        ALCOMX, ALLIMX, PHIMVX, ALPHAX, FSPV, GRAV
    )
    plane5 = _plane5_gcp_phicx(
        ALCOMX, ALLIMX, GCP, PHIMVX, ALPHAX, FSPV, GRAV
    )

    got = control.control_lateral(vehicle, ALCOMX)

    store = vehicle.store
    assert _approx(got, expected_phicx)
    assert _approx(store.get("alx"), expected_alx)
    assert got != pytest.approx(plane5, rel=RTOL, abs=ATOL)
    assert store.get("alcomx") == ALCOMX
    assert store.get("phicx") == 0.0
    np.testing.assert_array_equal(store.get("TBV"), np.zeros((3, 3)))


def test_control_lateral_ignores_gcp():
    vehicle, control = _ready(gcp=99.0)
    expected_phicx, _ = _expected_lateral(
        ALCOMX, ALLIMX, PHIMVX, ALPHAX, FSPV, GRAV
    )
    got = control.control_lateral(vehicle, ALCOMX)
    assert _approx(got, expected_phicx)


def test_control_lateral_clips_alcomx_to_allimx():
    alcomx = 2.0
    vehicle, control = _ready(alcomx=alcomx)
    clipped, _ = _expected_lateral(alcomx, ALLIMX, PHIMVX, ALPHAX, FSPV, GRAV)
    unlimited, _ = _expected_lateral(alcomx, alcomx, PHIMVX, ALPHAX, FSPV, GRAV)

    got = control.control_lateral(vehicle, alcomx)

    assert _approx(got, clipped)
    assert clipped != pytest.approx(unlimited, rel=RTOL, abs=ATOL)
    assert vehicle.store.get("alcomx") == alcomx


def test_control_lateral_does_not_write_phicx_or_tbv():
    vehicle, control = _ready()
    store = vehicle.store
    phicx_before = store.get("phicx")
    tbv_before = store.get("TBV").copy()

    phicx = control.control_lateral(vehicle, ALCOMX)

    assert phicx != phicx_before
    assert store.get("phicx") == phicx_before
    np.testing.assert_array_equal(store.get("TBV"), tbv_before)


def test_mcontrol_44_writes_chain_tbv_and_tbg():
    vehicle, control = _ready(mcontrol=44, alcomx=ALCOMX)
    store = vehicle.store
    phicx_lat, expected_alx = _expected_lateral(
        ALCOMX, ALLIMX, PHIMVX, ALPHAX, FSPV, GRAV
    )
    phicx_cmd = phicx_lat
    if phicx_cmd > PHILIMX:
        phicx_cmd = PHILIMX
    if phicx_cmd < -PHILIMX:
        phicx_cmd = -PHILIMX
    phixd_new = (phicx_cmd - 0.0) / TPHI
    phimvx_exp = integrate(phixd_new, 0.0, 0.0, INT_STEP)
    alphax_exp = _expected_load(
        0.0,
        INT_STEP,
        PHIMVX,
        ALPHAX,
        ANPOSLIMX,
        ANNEGLIMX,
        GACP,
        TA,
        ALPPOSLIMX,
        ALPNEGLIMX,
        FSPV,
        GRAV,
        MASS,
        DVBE,
        PDYNMC,
        THRUST,
        AREA,
        CLA,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    tbv_exp = cadtbv(phimvx_exp * RAD, alphax_exp * RAD)
    tbg_exp = tbv_exp @ TGV.T

    control.execute(vehicle, _ctx())

    assert _approx(store.get("phicx"), phicx_lat)
    assert _approx(store.get("phimvx"), phimvx_exp)
    assert _approx(store.get("alphax"), alphax_exp)
    assert store.get("alx") == expected_alx
    np.testing.assert_allclose(store.get("TBV"), tbv_exp, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("TBG"), tbg_exp, rtol=RTOL, atol=ATOL)


def test_mcontrol_3_is_int_three_bank_and_alphacx():
    alphacx = 2.0
    phicx = 30.0
    vehicle, control = _ready(mcontrol=3, alphacx=alphacx, phicx=phicx)
    store = vehicle.store
    phixd_new = (phicx - 0.0) / TPHI
    phimvx_exp = integrate(phixd_new, 0.0, 0.0, INT_STEP)
    tbv_exp = cadtbv(phimvx_exp * RAD, alphacx * RAD)

    control.execute(vehicle, _ctx())

    assert store.get("alphax") == alphacx
    assert _approx(store.get("phimvx"), phimvx_exp)
    np.testing.assert_allclose(store.get("TBV"), tbv_exp, rtol=RTOL, atol=ATOL)


def test_mcontrol_4_writes_local_zero_phimvx():
    vehicle, control = _ready(mcontrol=4, phimvx=12.0, ancomx=1.5)
    control.execute(vehicle, _ctx())
    assert vehicle.store.get("phimvx") == 0.0
    assert math.isfinite(vehicle.store.get("alphax"))
    assert vehicle.store.get("alphax") != 0.0
