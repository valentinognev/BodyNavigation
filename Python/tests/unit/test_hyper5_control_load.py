import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import cadtbv
from cadac.vehicles.round3.hyper5.control import Hyper5Control

# Demo 4.7
GACP = 10.0
TA = 0.8
ANPOSLIMX = 2.0
ANNEGLIMX = -2.0
ALPPOSLIMX = 6.0
ALPNEGLIMX = -4.0
INT_STEP = 0.05
MASS = 1352.0
DVBE = 254.0
AREA = 11.6986
PHIMVX = 0.0
ALPHAX = -1.5

PDYNMC = 72000.0
THRUST = 0.0
CLA = 0.08
GRAV = 9.81
FSPV = np.array([2.0, 1.0, -12.0])
ANCOMX = 1.5

RTOL = 1e-12
ATOL = 1e-14

LOAD_FIELDS = {
    "anposlimx": ("real", "data", ()),
    "anneglimx": ("real", "data", ()),
    "gacp": ("real", "data", ()),
    "ta": ("real", "data", ()),
    "alphax": ("real", "out", ("scrn", "plot")),
    "alpposlimx": ("real", "data", ()),
    "alpneglimx": ("real", "data", ()),
    "xi": ("real", "state", ()),
    "xid": ("real", "state", ()),
    "alp": ("real", "state", ()),
    "alpd": ("real", "state", ()),
    "anx": ("real", "diag", ("scrn", "plot")),
    "qq": ("real", "diag", ("plot",)),
    "tip": ("real", "diag", ("plot",)),
    "ancomx": ("real", "data", ("scrn", "plot")),
}


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


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


def test_define_registers_load_factor_fields():
    vehicle = _Vehicle()
    Hyper5Control().define(vehicle)
    store = vehicle.store
    for name, (ftype, role, outputs) in LOAD_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "control"
        assert field.outputs == outputs
        assert store.get(name) == 0.0


def test_define_skips_existing_fields():
    vehicle = _Vehicle()
    store = vehicle.store
    store.define(Field("phimvx", 12.0, "real", "out", "newton", ("scrn", "plot")))
    Hyper5Control().define(vehicle)
    assert store.get("phimvx") == 12.0
    assert store.field("phimvx").module == "newton"


def test_execute_dispatches_mcontrol_4():
    vehicle, control = _ready()
    store = vehicle.store
    store.define(
        Field(
            "tgv",
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            "mat",
            "init",
            "newton",
        )
    )
    store.set("mcontrol", 4)
    alphax_before = store.get("alphax")
    ancomx_before = store.get("ancomx")
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
    alpx = expected[0]

    control.execute(vehicle, _ctx())

    assert store.get("mcontrol") == 4
    assert store.get("phimvx") == 0.0
    assert store.get("ancomx") == ancomx_before
    assert _approx(store.get("alphax"), alpx)
    assert store.get("alphax") != alphax_before
    assert store.get("alphax") != 0.0


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
    assert _approx(got, alpx)
    assert _approx(store.get("xi"), xi)
    assert _approx(store.get("xid"), xid)
    assert _approx(store.get("alp"), alp)
    assert _approx(store.get("alpd"), alpd)
    assert _approx(store.get("anx"), anx)
    assert _approx(store.get("qq"), qq)
    assert _approx(store.get("tip"), tip)
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
    assert _approx(got, alpx)
    assert xi == 0.0
    assert xid == 0.0
    assert qq == 0.0
    assert store.get("xi") == 0.0
    assert store.get("xid") == 0.0
    assert store.get("qq") == 0.0
    assert _approx(store.get("alp"), alp)
    assert _approx(store.get("alpd"), alpd)
    assert _approx(store.get("tip"), tip)
    assert _approx(store.get("anx"), anx)
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
    assert _approx(got, alpx)
    assert expected != unlimited
    assert _approx(store.get("xi"), xi)
    assert _approx(store.get("xid"), xid)
    assert _approx(store.get("alp"), alp)
    assert _approx(store.get("alpd"), alpd)
    assert _approx(store.get("anx"), anx)
    assert _approx(store.get("qq"), qq)
    assert _approx(store.get("tip"), tip)
    assert store.get("ancomx") == ancomx


def test_alpx_clipped_to_alpposlimx_without_clipping_alp_state():
    alp0 = 0.2
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
    assert _approx(got, alpx)
    assert got == ALPPOSLIMX
    assert alp * DEG > ALPPOSLIMX
    assert _approx(store.get("alp"), alp)
    assert store.get("alp") != alpx
    assert _approx(store.get("alpd"), alpd)
    assert _approx(store.get("xi"), xi)
    assert _approx(store.get("xid"), xid)
    assert _approx(store.get("anx"), anx)
    assert _approx(store.get("qq"), qq)
    assert _approx(store.get("tip"), tip)
    assert store.get("alphax") == ALPHAX


def test_control_load_does_not_write_alphax():
    vehicle, control = _ready()
    alphax_before = vehicle.store.get("alphax")

    alpx = control.control_load(vehicle, ANCOMX, INT_STEP)

    assert alpx != alphax_before
    assert vehicle.store.get("alphax") == alphax_before


def test_control_bank_does_not_write_load_states():
    phicx = 30.0
    vehicle, control = _ready()
    store = vehicle.store
    store.set("phicx", phicx)
    store.set("philimx", 70.0)
    store.set("tphi", 1.0)
    alphax_before = store.get("alphax")
    ancomx_before = store.get("ancomx")
    phimvx_before = store.get("phimvx")

    phix = control.control_bank(vehicle, phicx, INT_STEP)

    assert _approx(phix, 0.75)
    assert _approx(store.get("phix"), 0.75)
    assert store.get("phimvx") == phimvx_before
    assert store.get("alphax") == alphax_before
    assert store.get("ancomx") == ancomx_before
    assert store.get("alp") == 0.0
    assert store.get("xi") == 0.0
