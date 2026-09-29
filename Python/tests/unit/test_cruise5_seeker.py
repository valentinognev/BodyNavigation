import numpy as np
import pytest

from cadac.constants import DEG, RAD, REARTH
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.earth import cadine, cadtei, cadtge
from cadac.math.frames import mat2tr
from cadac.vehicles.round3.cruise5.seeker import Cruise5Seeker

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

LONX = 14.7
LATX = 35.4
ALT = 7000.0
PSIVGX = 90.0
THTVGX = 0.0
DVBE = 200.0
TIME = 0.0
TGT_ALT = 0.0
TGT_PSIVGX = 0.0
TGT_THTVGX = 0.0
TGT_DVBE = 0.0


def _round3_state(lonx, latx, alt, psivgx, thtvgx, dvbe, time):
    tei = cadtei(time)
    tge = cadtge(lonx * RAD, latx * RAD)
    tig = tei.T @ tge.T
    sbii = cadine(lonx * RAD, latx * RAD, alt, time)
    vbeg = mat2tr(psivgx * RAD, thtvgx * RAD).T @ np.array([dvbe, 0.0, 0.0])
    return tig, sbii, vbeg


def _cruise5_with_target(range_m=500.0):
    dlat = (range_m / REARTH) * DEG
    tgt_latx = LATX + dlat
    tig, sbii, vbeg = _round3_state(LONX, LATX, ALT, PSIVGX, THTVGX, DVBE, TIME)
    _, tgt_sbii, tgt_vbeg = _round3_state(
        LONX, tgt_latx, TGT_ALT, TGT_PSIVGX, TGT_THTVGX, TGT_DVBE, TIME
    )
    vehicle = type("V", (), {"store": StateStore()})()
    Cruise5Seeker().define(vehicle)
    store = vehicle.store
    store.define(Field("tig", tig, "mat", "init/out", "newton"))
    store.define(Field("vbeg", vbeg, "vec", "state", "newton"))
    store.define(Field("sbii", sbii, "vec", "state", "newton"))
    store.define(Field("TBG", np.eye(3), "mat", "out", "control"))
    store.define(Field("lonx", LONX, "real", "init/diag", "newton"))
    store.define(Field("latx", LATX, "real", "init/diag", "newton"))
    combus = [
        Packet(
            name="t1",
            type="TARGET3",
            status=1,
            vars={
                "lonx": LONX,
                "latx": tgt_latx,
                "alt": TGT_ALT,
                "psivgx": TGT_PSIVGX,
                "thtvgx": TGT_THTVGX,
                "dvbe": TGT_DVBE,
                "vbeg": tgt_vbeg,
                "sbii": tgt_sbii,
            },
        ),
    ]
    return vehicle, combus


def _exec(veh, module_name, combus=None):
    assert module_name == "seeker"
    Cruise5Seeker().execute(
        veh,
        SimContext(0.0, 0.05, 0.0, 0.0, combus if combus is not None else [], 0),
    )


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


@pytest.mark.parametrize("mseeker", (99, -1))
def test_nonzero_mseeker_raises(mseeker):
    """Unknown modes still raise; 2/4 are scene-matching (Task 73, needs nmap)."""
    vehicle = type("V", (), {"store": StateStore()})()
    seeker = Cruise5Seeker()
    seeker.define(vehicle)
    vehicle.store.set("mseeker", mseeker)
    vehicle.store.set("nmap", 0)
    with pytest.raises(ValueError, match="mseeker"):
        seeker.execute(vehicle, SimContext(0.0, 0.05, 0.0, 0.0, [], 0))


def test_cruise5_mseeker_1_acquires_within_acq_range():
    veh, combus = _cruise5_with_target(range_m=500.0)
    veh.store.set("mseeker", 1)
    veh.store.set("acq_range", 1000.0)
    _exec(veh, "seeker", combus=combus)
    assert veh.store.get("mseeker") == 3
    assert veh.store.get("acquisition") == 1


def test_cruise5_mseeker_3_writes_woeb_and_closing_speed():
    veh, combus = _cruise5_with_target(range_m=500.0)
    veh.store.set("mseeker", 3)
    veh.store.set("acquisition", 1)
    _exec(veh, "seeker", combus=combus)
    assert "WOEB" in veh.store
    assert "closing_speed" in veh.store
