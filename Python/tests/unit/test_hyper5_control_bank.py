import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.hyper5.control import Hyper5Control

# Demo 4.7
PHILIMX = 70.0
TPHI = 1.0
INT_STEP = 0.05

RTOL = 1e-12
ATOL = 1e-14

BANK_FIELDS = {
    "phimvx": ("real", "out", ("scrn", "plot")),
    "phicx": ("real", "data", ("scrn", "plot")),
    "phix": ("real", "state", ("plot",)),
    "phixd": ("real", "state", ()),
    "philimx": ("real", "data", ()),
    "tphi": ("real", "data", ()),
}


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _expected_bank(phicx, phix, phixd, philimx, tphi, int_step):
    if phicx > philimx:
        phicx = philimx
    if phicx < -philimx:
        phicx = -philimx
    phixd_new = (phicx - phix) / tphi
    phix = phix + (phixd_new + phixd) * int_step / 2
    return phix, phixd_new


def _ctx(int_step=INT_STEP):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _ready(phicx, philimx=PHILIMX, tphi=TPHI):
    vehicle = _Vehicle()
    control = Hyper5Control()
    control.define(vehicle)
    store = vehicle.store
    store.set("phicx", phicx)
    store.set("philimx", philimx)
    store.set("tphi", tphi)
    return vehicle, control


def test_name_is_control():
    assert Hyper5Control().name == "control"


def test_define_registers_bank_fields():
    vehicle = _Vehicle()
    Hyper5Control().define(vehicle)
    store = vehicle.store
    for name, (ftype, role, outputs) in BANK_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "control"
        assert field.outputs == outputs
        assert store.get(name) == 0.0


def test_execute_dispatches_mcontrol_0():
    vehicle, control = _ready(30.0)
    store = vehicle.store
    store.define(
        Field(
            "TGV",
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            "mat",
            "init",
            "newton",
        )
    )
    store.set("phimvx", 12.0)
    store.set("alphax", -1.5)

    control.execute(vehicle, _ctx())

    assert store.get("mcontrol") == 0
    assert store.get("phimvx") == 0.0
    assert store.get("alphax") == 0.0
    assert store.get("phicx") == 30.0
    assert store.get("phix") == 0.0


def test_control_bank_one_step_lag():
    phicx = 30.0
    vehicle, control = _ready(phicx)
    expected, expected_d = _expected_bank(phicx, 0.0, 0.0, PHILIMX, TPHI, INT_STEP)

    phix = control.control_bank(vehicle, phicx, INT_STEP)

    store = vehicle.store
    assert _approx(phix, expected)
    assert _approx(store.get("phix"), expected)
    assert _approx(store.get("phixd"), expected_d)
    assert store.get("phicx") == phicx
    assert _approx(expected, 0.75)
    assert store.get("phimvx") == 0.0


def test_control_bank_second_step_uses_stored_slope():
    phicx = 30.0
    vehicle, control = _ready(phicx)
    control.control_bank(vehicle, phicx, INT_STEP)
    store = vehicle.store
    expected, expected_d = _expected_bank(
        phicx,
        store.get("phix"),
        store.get("phixd"),
        PHILIMX,
        TPHI,
        INT_STEP,
    )

    phix = control.control_bank(vehicle, phicx, INT_STEP)

    assert _approx(phix, expected)
    assert _approx(store.get("phix"), expected)
    assert _approx(store.get("phixd"), expected_d)
    assert store.get("phimvx") == 0.0
    assert _approx(expected, 2.23125)


def test_limiter_clips_command_above_philimx():
    phicx = 90.0
    vehicle, control = _ready(phicx)
    expected, expected_d = _expected_bank(phicx, 0.0, 0.0, PHILIMX, TPHI, INT_STEP)
    unlimited, _ = _expected_bank(phicx, 0.0, 0.0, phicx, TPHI, INT_STEP)

    phix = control.control_bank(vehicle, phicx, INT_STEP)

    store = vehicle.store
    assert _approx(phix, expected)
    assert _approx(store.get("phix"), expected)
    assert _approx(store.get("phixd"), expected_d)
    assert store.get("phicx") == phicx
    assert expected != pytest.approx(unlimited, rel=RTOL, abs=ATOL)
    assert _approx(expected, 1.75)
    assert store.get("phimvx") == 0.0


def test_limiter_clips_command_below_neg_philimx():
    phicx = -90.0
    vehicle, control = _ready(phicx)
    expected, expected_d = _expected_bank(phicx, 0.0, 0.0, PHILIMX, TPHI, INT_STEP)

    phix = control.control_bank(vehicle, phicx, INT_STEP)

    store = vehicle.store
    assert _approx(phix, expected)
    assert _approx(store.get("phix"), expected)
    assert _approx(store.get("phixd"), expected_d)
    assert store.get("phicx") == phicx
    assert _approx(expected, -1.75)
    assert store.get("phimvx") == 0.0


def test_control_bank_does_not_write_phimvx_or_phicx():
    phicx = 10.0
    vehicle, control = _ready(phicx)
    store = vehicle.store
    store.set("phimvx", 12.0)
    expected, expected_d = _expected_bank(phicx, 0.0, 0.0, PHILIMX, TPHI, INT_STEP)

    phix = control.control_bank(vehicle, phicx, INT_STEP)

    assert _approx(phix, expected)
    assert _approx(store.get("phix"), expected)
    assert _approx(store.get("phixd"), expected_d)
    assert store.get("phimvx") == 12.0
    assert store.get("phicx") == phicx
