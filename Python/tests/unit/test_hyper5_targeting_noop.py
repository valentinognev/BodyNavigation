from types import SimpleNamespace

import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round3.hyper5.targeting import Hyper5Targeting

TARGETING_FIELDS = {
    "mtargeting": ("int", "data", 0, ("scrn", "plot")),
    "del_radius": ("real", "data", 0.0, ()),
    "clost_tgt_slot": ("int", "out", 0, ()),
    "tgtng_sat_slot": ("int", "out", 0, ()),
}

NOT_DEFINED = (
    "wp_lonx",
    "wp_latx",
    "wp_alt",
    "mguidance",
    "SWBG",
    "wp_flag",
    "mseeker",
    "time",
    "lonx",
    "latx",
    "alt",
    "sbii",
    "SBII",
)

GUIDANCE_OWNED = ("wp_lonx", "wp_latx", "wp_alt")


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.05,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _ready(mtargeting=0):
    vehicle = SimpleNamespace(store=StateStore())
    targeting = Hyper5Targeting()
    targeting.define(vehicle)
    vehicle.store.set("mtargeting", mtargeting)
    targeting.initialize(vehicle, _ctx())
    return vehicle, targeting


def test_name_is_targeting():
    assert Hyper5Targeting().name == "targeting"


def test_define_registers_cpp_def_targeting_fields():
    vehicle = SimpleNamespace(store=StateStore())
    Hyper5Targeting().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(TARGETING_FIELDS)
    for name, (ftype, role, default, outputs) in TARGETING_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "targeting"
        assert field.outputs == outputs
        assert store.get(name) == default
        if ftype == "int":
            assert type(store.get(name)) is int


def test_define_does_not_register_guidance_or_plant():
    vehicle = SimpleNamespace(store=StateStore())
    Hyper5Targeting().define(vehicle)
    store = vehicle.store
    for name in NOT_DEFINED:
        with pytest.raises(KeyError):
            store.get(name)


def test_initialize_is_pass():
    vehicle, _targeting = _ready()
    store = vehicle.store
    assert store.get("mtargeting") == 0
    assert store.get("del_radius") == 0.0
    assert store.get("clost_tgt_slot") == 0
    assert store.get("tgtng_sat_slot") == 0


def test_terminate_exists_and_is_pass():
    vehicle, targeting = _ready()
    store = vehicle.store
    store.set("mtargeting", 0)
    store.set("del_radius", 12.0)
    store.set("clost_tgt_slot", 3)
    store.set("tgtng_sat_slot", 4)
    targeting.terminate(vehicle, _ctx())
    assert store.get("mtargeting") == 0
    assert store.get("del_radius") == 12.0
    assert store.get("clost_tgt_slot") == 3
    assert store.get("tgtng_sat_slot") == 4


def test_execute_mtargeting_zero_does_not_raise():
    vehicle, targeting = _ready(mtargeting=0)
    targeting.execute(vehicle, _ctx())


def test_execute_mtargeting_zero_does_not_write():
    vehicle, targeting = _ready(mtargeting=0)
    store = vehicle.store
    for name in GUIDANCE_OWNED:
        store.define(Field(name, 7.0, "real", "data", "guidance"))
    store.set("del_radius", 11.0)
    store.set("clost_tgt_slot", 8)
    store.set("tgtng_sat_slot", 9)
    targeting.execute(vehicle, _ctx())
    assert store.get("mtargeting") == 0
    assert store.get("del_radius") == 11.0
    assert store.get("clost_tgt_slot") == 8
    assert store.get("tgtng_sat_slot") == 9
    for name in GUIDANCE_OWNED:
        assert store.get(name) == 7.0


def test_execute_mtargeting_zero_does_not_require_combus_or_waypoints():
    vehicle, targeting = _ready(mtargeting=0)
    store = vehicle.store
    for name in NOT_DEFINED:
        assert name not in store.names()
    targeting.execute(vehicle, _ctx())
    for name in NOT_DEFINED:
        assert name not in store.names()


@pytest.mark.parametrize("mtargeting", (2, 99, -1))
def test_execute_mtargeting_nonzero_raises(mtargeting):
    vehicle, targeting = _ready(mtargeting=mtargeting)
    store = vehicle.store
    for name in GUIDANCE_OWNED:
        store.define(Field(name, 7.0, "real", "data", "guidance"))
    store.set("clost_tgt_slot", 8)
    store.set("tgtng_sat_slot", 9)
    with pytest.raises(ValueError, match="unknown mtargeting"):
        targeting.execute(vehicle, _ctx())
    for name in GUIDANCE_OWNED:
        assert store.get(name) == 7.0
    assert store.get("clost_tgt_slot") == 8
    assert store.get("tgtng_sat_slot") == 9
    assert store.get("mtargeting") == mtargeting
