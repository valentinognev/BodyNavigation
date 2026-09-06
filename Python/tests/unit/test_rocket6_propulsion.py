import numpy as np
import pytest

from cadac.constants import AGRAV
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import StateStore
from cadac.vehicles.rocket6.propulsion import Rocket6Propulsion

RTOL = 1e-12
ATOL = 1e-14

# insertion stage-1 (input.asc)
SPI = 279.2
FUEL_FLOW_RATE = 514.1
MPROP = 3
DT = 0.001
VMASS0 = 48984.0
FMASS0 = 31175.0
XCG_0 = 10.53
XCG_1 = 6.76
MOI_ROLL_0 = 21.94e3
MOI_ROLL_1 = 6.95e3
MOI_TRANS_0 = 671.62e3
MOI_TRANS_1 = 158.83e3

DEFINED = (
    "mprop",
    "acowl",
    "vmass",
    "vmass0",
    "xcg",
    "IBBB",
    "fmass0",
    "fmasse",
    "fmassd",
    "spi",
    "thrust",
    "fmassr",
    "xcg_0",
    "xcg_1",
    "fuel_flow_rate",
    "vmass0_st",
    "fmass0_st",
    "moi_roll_0",
    "moi_roll_1",
    "moi_trans_0",
    "moi_trans_1",
    "mfreeze_prop",
    "thrustf",
    "vmassf",
    "IBBBF",
)

ROLES = {
    "mprop": "data",
    "acowl": "data",
    "vmass": "out",
    "vmass0": "data",
    "xcg": "out",
    "IBBB": "out",
    "fmass0": "data",
    "fmasse": "state",
    "fmassd": "state",
    "spi": "data",
    "thrust": "out",
    "fmassr": "save",
    "xcg_0": "data",
    "xcg_1": "data",
    "fuel_flow_rate": "data",
    "vmass0_st": "data",
    "fmass0_st": "data",
    "moi_roll_0": "data",
    "moi_roll_1": "data",
    "moi_trans_0": "data",
    "moi_trans_1": "data",
    "mfreeze_prop": "save",
    "thrustf": "save",
    "vmassf": "save",
    "IBBBF": "save",
}

OUTPUTS = {
    "mprop": (),
    "acowl": (),
    "vmass": ("scrn", "plot"),
    "vmass0": (),
    "xcg": ("plot",),
    "IBBB": (),
    "fmass0": (),
    "fmasse": ("scrn", "plot"),
    "fmassd": (),
    "spi": (),
    "thrust": ("scrn", "plot"),
    "fmassr": ("scrn", "plot"),
    "xcg_0": (),
    "xcg_1": (),
    "fuel_flow_rate": (),
    "vmass0_st": (),
    "fmass0_st": (),
    "moi_roll_0": (),
    "moi_roll_1": (),
    "moi_trans_0": (),
    "moi_trans_1": (),
    "mfreeze_prop": (),
    "thrustf": (),
    "vmassf": (),
    "IBBBF": (),
}

INT_FIELDS = ("mprop", "mfreeze_prop")
MAT_FIELDS = ("IBBB", "IBBBF")
NOT_DEFINED = ("press", "time", "rho", "pdynmc", "vmach", "mfreeze")


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(dt=DT):
    return SimContext(
        sim_time=0.0,
        int_step=dt,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ready(*, mprop=MPROP, **kw):
    vehicle = _Vehicle()
    prop = Rocket6Propulsion()
    prop.define(vehicle)
    prop.initialize(vehicle, _ctx())
    store = vehicle.store
    data = {
        "mprop": mprop,
        "spi": SPI,
        "fuel_flow_rate": FUEL_FLOW_RATE,
        "vmass0": VMASS0,
        "fmass0": FMASS0,
        "xcg_0": XCG_0,
        "xcg_1": XCG_1,
        "moi_roll_0": MOI_ROLL_0,
        "moi_roll_1": MOI_ROLL_1,
        "moi_trans_0": MOI_TRANS_0,
        "moi_trans_1": MOI_TRANS_1,
    }
    data.update(kw)
    for name, value in data.items():
        store.set(name, value)
    return vehicle, prop


def _expected(store, dt):
    mprop = store.get("mprop")
    vmass0 = store.get("vmass0")
    fmass0 = store.get("fmass0")
    spi = store.get("spi")
    xcg_0 = store.get("xcg_0")
    xcg_1 = store.get("xcg_1")
    fuel_flow_rate = store.get("fuel_flow_rate")
    moi_roll_0 = store.get("moi_roll_0")
    moi_roll_1 = store.get("moi_roll_1")
    moi_trans_0 = store.get("moi_trans_0")
    moi_trans_1 = store.get("moi_trans_1")
    vmass = store.get("vmass")
    xcg = store.get("xcg")
    ibbb = np.array(store.get("IBBB"), dtype=float)
    fmassr = store.get("fmassr")
    fmasse = store.get("fmasse")
    fmassd = store.get("fmassd")
    thrust = 0.0
    if mprop == 0:
        fmassd = 0.0
        thrust = 0.0
        fmasse = 0.0
        fmassr = 0.0
    if mprop > 0:
        if mprop == 3 or mprop == 4:
            thrust = spi * fuel_flow_rate * AGRAV
            ibbb0 = np.zeros((3, 3))
            ibbb0[0, 0] = moi_roll_0
            ibbb0[1, 1] = moi_trans_0
            ibbb0[2, 2] = moi_trans_0
            ibbb1 = np.zeros((3, 3))
            ibbb1[0, 0] = moi_roll_1
            ibbb1[1, 1] = moi_trans_1
            ibbb1[2, 2] = moi_trans_1
        if spi != 0:
            fmassd_next = thrust / (spi * AGRAV)
            fmasse = integrate(fmassd_next, fmassd, fmasse, dt)
            fmassd = fmassd_next
        vmass = vmass0 - fmasse
        fmassr = fmass0 - fmasse
        mass_ratio = fmasse / fmass0
        ibbb = ibbb0 + (ibbb1 - ibbb0) * mass_ratio
        xcg = xcg_0 + (xcg_1 - xcg_0) * mass_ratio
        if fmassr <= 0:
            mprop = 0
            thrust = 0.0
    return {
        "mprop": mprop,
        "thrust": thrust,
        "fmasse": fmasse,
        "fmassd": fmassd,
        "fmassr": fmassr,
        "vmass": vmass,
        "xcg": xcg,
        "IBBB": ibbb,
    }


def test_name_is_propulsion():
    assert Rocket6Propulsion().name == "propulsion"


def test_constructor_has_no_deck():
    prop = Rocket6Propulsion()
    assert not hasattr(prop, "deck")


def test_define_registers_cpp_fields_not_externals():
    vehicle = _Vehicle()
    Rocket6Propulsion().define(vehicle)
    store = vehicle.store
    for name in DEFINED:
        assert name in store.names(), name
        field = store.field(name)
        assert field.module == "propulsion"
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        if name in INT_FIELDS:
            assert field.type == "int"
            assert store.get(name) == 0
        elif name in MAT_FIELDS:
            assert field.type == "mat"
            assert np.array_equal(store.get(name), np.zeros((3, 3)))
        else:
            assert field.type == "real"
            assert store.get(name) == 0.0
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_initialize_is_noop():
    vehicle = _Vehicle()
    prop = Rocket6Propulsion()
    prop.define(vehicle)
    store = vehicle.store
    store.set("vmass0", VMASS0)
    store.set("fmass0", FMASS0)
    prop.initialize(vehicle, _ctx())
    assert store.get("vmass") == 0.0
    assert store.get("fmasse") == 0.0
    assert store.get("thrust") == 0.0
    assert store.get("mprop") == 0
    assert store.get("vmass0") == VMASS0
    assert store.get("fmass0") == FMASS0


def test_stage1_mprop3_thrust_is_spi_times_flow_times_agrav():
    # Break: ramjet formula, missing AGRAV, or mprop=3 treated as off.
    vehicle, prop = _ready(mprop=3, spi=SPI, fuel_flow_rate=FUEL_FLOW_RATE)
    store = vehicle.store
    prop.execute(vehicle, _ctx(DT))
    want = SPI * FUEL_FLOW_RATE * AGRAV
    assert store.get("thrust") == want
    assert _approx(store.get("thrust"), want)


def test_mprop4_same_analytic_thrust_as_mprop3():
    vehicle, prop = _ready(mprop=4, spi=SPI, fuel_flow_rate=FUEL_FLOW_RATE)
    store = vehicle.store
    prop.execute(vehicle, _ctx(DT))
    want = SPI * FUEL_FLOW_RATE * AGRAV
    assert store.get("thrust") == want


def test_mprop0_after_burn_zeros_thrust_and_fuel_state():
    # BECO sets mprop=0 only; propulsion must clear leftover fmasse on that step.
    vehicle, prop = _ready(mprop=3)
    store = vehicle.store
    prop.execute(vehicle, _ctx(DT))
    leftover = store.get("fmasse")
    assert leftover > 0.0
    store.set("mprop", 0)
    prop.execute(vehicle, _ctx(DT))
    assert store.get("thrust") == 0.0
    assert store.get("fmasse") == 0.0
    assert store.get("fmassr") == 0.0
    assert store.get("fmassd") == 0.0


def test_mprop2_raises():
    vehicle, prop = _ready(mprop=2)
    with pytest.raises(ValueError):
        prop.execute(vehicle, _ctx())


def test_other_mprop_raises():
    for mprop in (1, 5, -1):
        vehicle, prop = _ready(mprop=mprop)
        with pytest.raises(ValueError):
            prop.execute(vehicle, _ctx())


def test_fuel_exhaustion_sets_mprop_zero():
    vehicle, prop = _ready(mprop=3, fmass0=1.0)
    store = vehicle.store
    saw_empty = False
    for _ in range(20):
        prop.execute(vehicle, _ctx(DT))
        if store.get("fmassr") <= 0:
            saw_empty = True
            assert store.get("mprop") == 0
            assert store.get("thrust") == 0.0
            break
    assert saw_empty
    assert store.get("mprop") == 0


def test_execute_matches_cadac_formulas():
    vehicle, prop = _ready(mprop=3)
    store = vehicle.store
    want = _expected(store, DT)
    prop.execute(vehicle, _ctx(DT))
    assert store.get("mprop") == want["mprop"]
    for name in ("thrust", "fmasse", "fmassd", "fmassr", "vmass", "xcg"):
        assert np.isfinite(store.get(name)), name
        assert _approx(store.get(name), want[name]), name
    assert np.allclose(store.get("IBBB"), want["IBBB"], rtol=RTOL, atol=ATOL)


def test_linear_ibbb_and_xcg_vs_fuel_expended():
    vehicle, prop = _ready(mprop=3)
    store = vehicle.store
    prop.execute(vehicle, _ctx(DT))
    fmasse = store.get("fmasse")
    mass_ratio = fmasse / FMASS0
    want_xcg = XCG_0 + (XCG_1 - XCG_0) * mass_ratio
    want_roll = MOI_ROLL_0 + (MOI_ROLL_1 - MOI_ROLL_0) * mass_ratio
    want_trans = MOI_TRANS_0 + (MOI_TRANS_1 - MOI_TRANS_0) * mass_ratio
    ibbb = store.get("IBBB")
    assert _approx(store.get("xcg"), want_xcg)
    assert _approx(ibbb[0, 0], want_roll)
    assert _approx(ibbb[1, 1], want_trans)
    assert _approx(ibbb[2, 2], want_trans)
    assert ibbb[0, 1] == 0.0
    assert ibbb[0, 2] == 0.0
    assert ibbb[1, 0] == 0.0
    assert ibbb[1, 2] == 0.0
    assert ibbb[2, 0] == 0.0
    assert ibbb[2, 1] == 0.0
    assert _approx(store.get("vmass"), VMASS0 - fmasse)
    assert _approx(store.get("fmassr"), FMASS0 - fmasse)
