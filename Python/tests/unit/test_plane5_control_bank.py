from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import StateStore
from cadac.vehicles.plane5.control import Plane5Control

# turning_to_IP
PHILIMX = 70.0
TPHI = 1.0
INT_STEP = 0.05

BANK_FIELDS = {
    "phimvx": ("real", "out", ("scrn", "plot")),
    "phicx": ("real", "data", ("scrn", "plot")),
    "phix": ("real", "state", ("plot",)),
    "phixd": ("real", "state", ()),
    "philimx": ("real", "data", ()),
    "tphi": ("real", "data", ()),
}

NOT_YET = (
    "mcontrol",
    "TBV",
    "alcomx",
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


def _expected_bank(phicx, phix, phixd, philimx, tphi, int_step):
    if phicx > philimx:
        phicx = philimx
    if phicx < -philimx:
        phicx = -philimx
    phixd_new = (phicx - phix) / tphi
    phix = integrate(phixd_new, phixd, phix, int_step)
    return phix, phixd_new


def _ready(phicx, philimx=PHILIMX, tphi=TPHI):
    vehicle = _Vehicle()
    control = Plane5Control()
    control.define(vehicle)
    store = vehicle.store
    store.set("phicx", phicx)
    store.set("philimx", philimx)
    store.set("tphi", tphi)
    return vehicle, control


def test_name_is_control():
    assert Plane5Control().name == "control"


def test_define_registers_bank_fields():
    vehicle = _Vehicle()
    Plane5Control().define(vehicle)
    store = vehicle.store
    for name, (ftype, role, outputs) in BANK_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "control"
        assert field.outputs == outputs
        assert store.get(name) == 0.0
    for name in NOT_YET:
        assert name not in store.names()


def test_execute_steps_command_to_phimvx():
    phicx = 30.0
    vehicle, control = _ready(phicx)
    expected, expected_d = _expected_bank(phicx, 0.0, 0.0, PHILIMX, TPHI, INT_STEP)

    control.execute(vehicle, _ctx())

    store = vehicle.store
    assert store.get("phimvx") == expected
    assert store.get("phix") == expected
    assert store.get("phixd") == expected_d
    assert store.get("phicx") == phicx
    assert expected == 0.75


def test_execute_second_step_uses_stored_slope():
    phicx = 30.0
    vehicle, control = _ready(phicx)
    ctx = _ctx()
    control.execute(vehicle, ctx)
    store = vehicle.store
    expected, expected_d = _expected_bank(
        phicx,
        store.get("phix"),
        store.get("phixd"),
        PHILIMX,
        TPHI,
        INT_STEP,
    )

    control.execute(vehicle, ctx)

    assert store.get("phimvx") == expected
    assert store.get("phix") == expected
    assert store.get("phixd") == expected_d


def test_limiter_clips_command_above_philimx():
    phicx = 90.0
    vehicle, control = _ready(phicx)
    expected, expected_d = _expected_bank(phicx, 0.0, 0.0, PHILIMX, TPHI, INT_STEP)
    unlimited, _ = _expected_bank(phicx, 0.0, 0.0, phicx, TPHI, INT_STEP)

    control.execute(vehicle, _ctx())

    store = vehicle.store
    assert store.get("phimvx") == expected
    assert store.get("phix") == expected
    assert store.get("phixd") == expected_d
    assert store.get("phicx") == phicx
    assert expected != unlimited
    assert expected == 1.75


def test_limiter_clips_command_below_neg_philimx():
    phicx = -90.0
    vehicle, control = _ready(phicx)
    expected, expected_d = _expected_bank(phicx, 0.0, 0.0, PHILIMX, TPHI, INT_STEP)

    control.execute(vehicle, _ctx())

    store = vehicle.store
    assert store.get("phimvx") == expected
    assert store.get("phix") == expected
    assert store.get("phixd") == expected_d
    assert store.get("phicx") == phicx
    assert expected == -1.75


def test_control_bank_returns_phix_without_writing_phimvx():
    phicx = 10.0
    vehicle, control = _ready(phicx)
    expected, expected_d = _expected_bank(phicx, 0.0, 0.0, PHILIMX, TPHI, INT_STEP)

    phix = control.control_bank(vehicle, phicx, INT_STEP)

    store = vehicle.store
    assert phix == expected
    assert store.get("phix") == expected
    assert store.get("phixd") == expected_d
    assert store.get("phimvx") == 0.0
