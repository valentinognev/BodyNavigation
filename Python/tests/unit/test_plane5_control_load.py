import numpy as np

from cadac.constants import DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import cadtbv
from cadac.vehicles.plane5.control import Plane5Control

# turning_to_IP
GACP = 10.0
TA = 0.8
ANPOSLIMX = 3.0
ANNEGLIMX = -1.0
ALPPOSLIMX = 15.0
ALPNEGLIMX = -10.0
INT_STEP = 0.05
MASS = 12701.0
DVBE = 200.0
AREA = 27.87
PDYNMC = 17000.0
THRUST = 20000.0
CLA = 0.08
GRAV = 9.81
FSPV = np.array([2.0, 1.0, -12.0])
PHIMVX = 20.0
ALPHAX = 5.0
ANCOMX = 1.5

LOAD_FIELDS = {
    "anx": ("real", "diag", ("scrn", "plot")),
    "alphax": ("real", "out", ("scrn", "plot")),
    "anposlimx": ("real", "data", ()),
    "anneglimx": ("real", "data", ()),
    "gacp": ("real", "data", ()),
    "ta": ("real", "data", ()),
    "xi": ("real", "state", ()),
    "xid": ("real", "state", ()),
    "qq": ("real", "diag", ("plot",)),
    "tip": ("real", "diag", ("plot",)),
    "alp": ("real", "state", ("plot",)),
    "alpd": ("real", "state", ()),
    "alpposlimx": ("real", "data", ()),
    "alpneglimx": ("real", "data", ()),
    "ancomx": ("real", "data", ("plot",)),
}

NOT_YET = (
    "mcontrol",
    "TBV",
    "alcomx",
    "altcom",
    "psivlcx",
    "thtvgcx",
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


def _expected_cadtbv(phi, alpha):
    amat = np.zeros((3, 3))
    salpha = np.sin(alpha)
    calpha = np.cos(alpha)
    sphi = np.sin(phi)
    cphi = np.cos(phi)
    amat[0, 0] = calpha
    amat[0, 1] = sphi * salpha
    amat[0, 2] = -cphi * salpha
    amat[1, 1] = cphi
    amat[1, 2] = sphi
    amat[2, 0] = salpha
    amat[2, 1] = -sphi * calpha
    amat[2, 2] = cphi * calpha
    return amat


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
    alpha = alphax * RAD
    phimv = phimvx * RAD
    tbv = cadtbv(phimv, alpha)
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
    return alpx, xi, xid, alp, alpd, anx, qq, tip


def _ready(
    ancomx=ANCOMX,
    phimvx=PHIMVX,
    alphax=ALPHAX,
    gacp=GACP,
    ta=TA,
    anposlimx=ANPOSLIMX,
    anneglimx=ANNEGLIMX,
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
    xi=0.0,
    xid=0.0,
    alp=0.0,
    alpd=0.0,
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
    store.set("ancomx", ancomx)
    store.set("phimvx", phimvx)
    store.set("alphax", alphax)
    store.set("gacp", gacp)
    store.set("ta", ta)
    store.set("anposlimx", anposlimx)
    store.set("anneglimx", anneglimx)
    store.set("alpposlimx", alpposlimx)
    store.set("alpneglimx", alpneglimx)
    store.set("xi", xi)
    store.set("xid", xid)
    store.set("alp", alp)
    store.set("alpd", alpd)
    return vehicle, control


def test_cadtbv_zeros_then_assign_loc_leaves_row1_col0_zero():
    phi = 0.3
    alpha = 0.2
    amat = cadtbv(phi, alpha)
    expected = _expected_cadtbv(phi, alpha)
    np.testing.assert_array_equal(amat, expected)
    assert amat[1, 0] == 0.0
    assert expected[1, 0] == 0.0


def test_define_registers_load_factor_fields():
    vehicle = _Vehicle()
    Plane5Control().define(vehicle)
    store = vehicle.store
    for name, (ftype, role, outputs) in LOAD_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "control"
        assert field.outputs == outputs
        assert store.get(name) == 0.0
    for name in NOT_YET:
        assert name not in store.names()


def test_control_load_one_step_matches_cpp_equations():
    vehicle, control = _ready()
    expected = _expected_load(
        ANCOMX,
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
    alpx, xi, xid, alp, alpd, anx, qq, tip = expected

    got = control.control_load(vehicle, ANCOMX, INT_STEP)

    store = vehicle.store
    assert got == alpx
    assert store.get("xi") == xi
    assert store.get("xid") == xid
    assert store.get("alp") == alp
    assert store.get("alpd") == alpd
    assert store.get("anx") == anx
    assert store.get("qq") == qq
    assert store.get("tip") == tip
    assert store.get("alphax") == ALPHAX
    assert store.get("ancomx") == ANCOMX


def test_ta_not_positive_zeros_integral_and_leaves_gr_zero():
    alp0 = 0.1
    vehicle, control = _ready(ta=0.0, alp=alp0)
    expected = _expected_load(
        ANCOMX,
        INT_STEP,
        PHIMVX,
        ALPHAX,
        ANPOSLIMX,
        ANNEGLIMX,
        GACP,
        0.0,
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
        alp0,
        0.0,
    )
    alpx, xi, xid, alp, alpd, anx, qq, tip = expected

    got = control.control_load(vehicle, ANCOMX, INT_STEP)

    store = vehicle.store
    assert got == alpx
    assert xi == 0.0
    assert xid == 0.0
    assert qq == 0.0
    assert store.get("xi") == 0.0
    assert store.get("xid") == 0.0
    assert store.get("qq") == 0.0
    assert store.get("qq") == qq
    assert store.get("alp") == alp
    assert store.get("alpd") == alpd
    assert store.get("tip") == tip
    assert store.get("anx") == anx
    assert store.get("alphax") == ALPHAX


def test_ancomx_clipped_to_anposlimx():
    ancomx = 5.0
    vehicle, control = _ready(ancomx=ancomx)
    expected = _expected_load(
        ancomx,
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
    unlimited = _expected_load(
        ancomx,
        INT_STEP,
        PHIMVX,
        ALPHAX,
        ancomx,
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
    alpx, xi, xid, alp, alpd, anx, qq, tip = expected

    got = control.control_load(vehicle, ancomx, INT_STEP)

    store = vehicle.store
    assert got == alpx
    assert expected != unlimited
    assert store.get("xi") == xi
    assert store.get("xid") == xid
    assert store.get("alp") == alp
    assert store.get("alpd") == alpd
    assert store.get("anx") == anx
    assert store.get("qq") == qq
    assert store.get("tip") == tip
    assert store.get("ancomx") == ancomx


def test_alpx_clipped_to_alpposlimx_without_clipping_alp_state():
    alp0 = 0.3
    vehicle, control = _ready(alp=alp0)
    expected = _expected_load(
        ANCOMX,
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
        alp0,
        0.0,
    )
    alpx, xi, xid, alp, alpd, anx, qq, tip = expected

    got = control.control_load(vehicle, ANCOMX, INT_STEP)

    store = vehicle.store
    assert got == alpx
    assert got == ALPPOSLIMX
    assert alp * DEG > ALPPOSLIMX
    assert store.get("alp") == alp
    assert store.get("alp") != alpx
    assert store.get("alpd") == alpd
    assert store.get("xi") == xi
    assert store.get("xid") == xid
    assert store.get("anx") == anx
    assert store.get("qq") == qq
    assert store.get("tip") == tip
    assert store.get("alphax") == ALPHAX


def test_control_load_does_not_write_alphax():
    vehicle, control = _ready()
    alphax_before = vehicle.store.get("alphax")

    alpx = control.control_load(vehicle, ANCOMX, INT_STEP)

    assert alpx != alphax_before
    assert vehicle.store.get("alphax") == alphax_before


def test_execute_still_bank_wrap_only():
    phicx = 30.0
    vehicle, control = _ready()
    store = vehicle.store
    store.set("phicx", phicx)
    store.set("philimx", 70.0)
    store.set("tphi", 1.0)
    alphax_before = store.get("alphax")
    ancomx_before = store.get("ancomx")

    control.execute(vehicle, _ctx())

    assert store.get("phimvx") == 0.75
    assert store.get("phix") == 0.75
    assert store.get("alphax") == alphax_before
    assert store.get("ancomx") == ancomx_before
    assert store.get("alp") == 0.0
    assert store.get("xi") == 0.0
