import pytest
from cadac.kernel.state import StateStore
from cadac.vehicles.round3.cruise5.control import Cruise5Control

PHILIMX = 70.0
TPHI = 0.5
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


def _expected_bank(phicx, phix, phixd, philimx, tphi, int_step):
    if phicx > philimx:
        phicx = philimx
    if phicx < -philimx:
        phicx = -philimx
    phixd_new = (phicx - phix) / tphi
    phix = phix + (phixd_new + phixd) * int_step / 2
    return phix, phixd_new


def _ready(phicx=30.0):
    vehicle = type("V", (), {"store": StateStore()})()
    control = Cruise5Control()
    control.define(vehicle)
    store = vehicle.store
    store.set("philimx", PHILIMX)
    store.set("tphi", TPHI)
    store.set("phicx", phicx)
    return vehicle, control


def test_name_is_control():
    assert Cruise5Control().name == "control"


def test_define_registers_bank_fields():
    vehicle = type("V", (), {"store": StateStore()})()
    Cruise5Control().define(vehicle)
    store = vehicle.store
    for name, (ftype, role, outputs) in BANK_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "control"
        assert field.outputs == outputs
        assert store.get(name) == 0.0


def test_control_bank_one_step_input1_tphi():
    vehicle = type("V", (), {"store": StateStore()})()
    control = Cruise5Control()
    control.define(vehicle)
    store = vehicle.store
    store.set("philimx", PHILIMX)
    store.set("tphi", TPHI)
    expected, expected_d = _expected_bank(30.0, 0.0, 0.0, PHILIMX, TPHI, INT_STEP)
    phix = control.control_bank(vehicle, 30.0, INT_STEP)
    assert phix == pytest.approx(expected, rel=RTOL, abs=ATOL)
    assert expected == pytest.approx(1.5, rel=RTOL, abs=ATOL)
    assert store.get("phixd") == pytest.approx(expected_d, rel=RTOL, abs=ATOL)
    assert store.get("phimvx") == 0.0


def test_control_bank_second_step_uses_stored_slope():
    vehicle, control = _ready(30.0)
    control.control_bank(vehicle, 30.0, INT_STEP)
    store = vehicle.store
    expected, expected_d = _expected_bank(
        30.0,
        store.get("phix"),
        store.get("phixd"),
        PHILIMX,
        TPHI,
        INT_STEP,
    )
    phix = control.control_bank(vehicle, 30.0, INT_STEP)
    assert phix == pytest.approx(expected, rel=RTOL, abs=ATOL)
    assert store.get("phix") == pytest.approx(expected, rel=RTOL, abs=ATOL)
    assert store.get("phixd") == pytest.approx(expected_d, rel=RTOL, abs=ATOL)
    assert store.get("phimvx") == 0.0
    assert expected == pytest.approx(4.425, rel=RTOL, abs=ATOL)


def test_control_bank_clips_phicx_90_to_philimx():
    vehicle, control = _ready(90.0)
    expected, expected_d = _expected_bank(90.0, 0.0, 0.0, PHILIMX, TPHI, INT_STEP)
    unlimited, _ = _expected_bank(90.0, 0.0, 0.0, 90.0, TPHI, INT_STEP)
    phix = control.control_bank(vehicle, 90.0, INT_STEP)
    store = vehicle.store
    assert phix == pytest.approx(expected, rel=RTOL, abs=ATOL)
    assert store.get("phix") == pytest.approx(expected, rel=RTOL, abs=ATOL)
    assert store.get("phixd") == pytest.approx(expected_d, rel=RTOL, abs=ATOL)
    assert expected != pytest.approx(unlimited, rel=RTOL, abs=ATOL)
    assert expected == pytest.approx(3.5, rel=RTOL, abs=ATOL)
    assert store.get("phimvx") == 0.0


def test_control_bank_clips_phicx_neg90_to_neg_philimx():
    vehicle, control = _ready(-90.0)
    expected, expected_d = _expected_bank(-90.0, 0.0, 0.0, PHILIMX, TPHI, INT_STEP)
    phix = control.control_bank(vehicle, -90.0, INT_STEP)
    store = vehicle.store
    assert phix == pytest.approx(expected, rel=RTOL, abs=ATOL)
    assert store.get("phix") == pytest.approx(expected, rel=RTOL, abs=ATOL)
    assert store.get("phixd") == pytest.approx(expected_d, rel=RTOL, abs=ATOL)
    assert expected == pytest.approx(-3.5, rel=RTOL, abs=ATOL)
    assert store.get("phimvx") == 0.0
