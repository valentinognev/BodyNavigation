import numpy as np
import pytest
from cadac.kernel.executive import SimContext
from cadac.kernel.state import StateStore
from cadac.vehicles.cruise5.seeker import Cruise5Seeker

SEEKER_FIELDS = {
    "mseeker": ("int", "data/save", ("scrn",), 0),
    "acq_range": ("real", "data", (), 0.0),
    "range_go": ("real", "out", ("plot", "scrn"), 0.0),
    "STBG": ("vec", "out", ("plot",), None),
    "WOEB": ("vec", "out", (), None),
    "closing_speed": ("real", "out", (), 0.0),
    "time_go": ("real", "out", ("plot", "scrn"), 0.0),
    "psisbx": ("real", "out", ("plot", "scrn"), 0.0),
    "thtsbx": ("real", "out", ("plot", "scrn"), 0.0),
    "targ_com_slot": ("int", "save", (), 0),
    "UTBB": ("vec", "out", (), None),
    "acquisition": ("int", "init/save", ("scrn",), 0),
}


def test_mseeker_0_no_write():
    vehicle = type("V", (), {"store": StateStore()})()
    seeker = Cruise5Seeker()
    seeker.define(vehicle)
    store = vehicle.store
    store.set("mseeker", 0)
    store.set("range_go", 7.0)
    seeker.execute(vehicle, SimContext(0.0, 0.05, 0.0, 0.0, [], 0))
    assert store.get("range_go") == 7.0
    assert store.get("mseeker") == 0

def test_mseeker_1_raises():
    vehicle = type("V", (), {"store": StateStore()})()
    seeker = Cruise5Seeker()
    seeker.define(vehicle)
    vehicle.store.set("mseeker", 1)
    with pytest.raises(ValueError, match="mseeker"):
        seeker.execute(vehicle, SimContext(0.0, 0.05, 0.0, 0.0, [], 0))


def test_name_is_seeker():
    assert Cruise5Seeker().name == "seeker"


def test_define_registers_def_seeker_fields():
    vehicle = type("V", (), {"store": StateStore()})()
    Cruise5Seeker().define(vehicle)
    store = vehicle.store
    zeros = np.zeros(3)
    for name, (ftype, role, outputs, default) in SEEKER_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "seeker"
        assert field.outputs == outputs
        if ftype == "vec":
            np.testing.assert_array_equal(store.get(name), zeros)
        else:
            assert store.get(name) == default
    assert type(store.get("mseeker")) is int
    assert type(store.get("targ_com_slot")) is int
    assert type(store.get("acquisition")) is int


@pytest.mark.parametrize("mseeker", (2, 3, 99, -1))
def test_nonzero_mseeker_raises(mseeker):
    vehicle = type("V", (), {"store": StateStore()})()
    seeker = Cruise5Seeker()
    seeker.define(vehicle)
    vehicle.store.set("mseeker", mseeker)
    with pytest.raises(ValueError, match="mseeker"):
        seeker.execute(vehicle, SimContext(0.0, 0.05, 0.0, 0.0, [], 0))
