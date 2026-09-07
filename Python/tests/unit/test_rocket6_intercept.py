from unittest.mock import patch

from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.rocket6.intercept import Rocket6Intercept

SCRN_PLOT = ("scrn", "plot")

DEFINED = {
    "write": ("int", "init", 1, ()),
    "modes": ("int", "diag", 0, SCRN_PLOT),
}

# newton / guidance / control names intercept reads but does not define
NOT_DEFINED = (
    "alt",
    "mguide",
    "maut",
    "mprop",
    "time",
    "dvbe",
    "psivdx",
    "thtvdx",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()
        self.health = 1


def _ctx(combus=None, vehicle_slot=0):
    return SimContext(
        sim_time=0.0,
        int_step=0.001,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=vehicle_slot,
    )


def _plant(store, *, alt=100.0, mguide=0, maut=0, mprop=0):
    store.define(Field("alt", alt, "real", "out", "newton"))
    store.define(Field("mguide", mguide, "int", "data", "guidance"))
    store.define(Field("maut", maut, "int", "data", "control"))
    store.define(Field("mprop", mprop, "int", "data", "propulsion"))


def _ready(*, alt=100.0, write=1, mguide=0, maut=0, mprop=0, combus="default"):
    vehicle = _Vehicle()
    intercept = Rocket6Intercept()
    intercept.define(vehicle)
    store = vehicle.store
    store.set("write", write)
    _plant(store, alt=alt, mguide=mguide, maut=maut, mprop=mprop)
    if combus == "default":
        combus = [Packet(name="SLV", type="HYPER6", status=1, vars={})]
    return vehicle, intercept, _ctx(combus=combus)


def test_name_is_intercept():
    assert Rocket6Intercept().name == "intercept"


def test_define_registers_write_and_modes():
    vehicle = _Vehicle()
    Rocket6Intercept().define(vehicle)
    store = vehicle.store
    assert store.names() == list(DEFINED)
    for name, (ftype, role, default, outputs) in DEFINED.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "intercept"
        assert field.outputs == outputs
        assert store.get(name) == default
        assert type(store.get(name)) is int


def test_define_does_not_register_newton_guidance_control_names():
    vehicle = _Vehicle()
    Rocket6Intercept().define(vehicle)
    for name in NOT_DEFINED:
        assert name not in vehicle.store.names()


def test_initialize_is_pass():
    vehicle = _Vehicle()
    intercept = Rocket6Intercept()
    intercept.define(vehicle)
    intercept.initialize(vehicle, _ctx())
    assert vehicle.store.get("write") == 1
    assert vehicle.store.get("modes") == 0
    assert vehicle.health == 1


def test_terminate_is_pass():
    vehicle = _Vehicle()
    intercept = Rocket6Intercept()
    intercept.define(vehicle)
    intercept.terminate(vehicle, _ctx())
    assert vehicle.store.get("write") == 1
    assert vehicle.health == 1


def test_alt_100_health_stays_1():
    vehicle, intercept, ctx = _ready(alt=100.0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert vehicle.store.get("write") == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1


def test_alt_minus_1_health_0():
    vehicle, intercept, ctx = _ready(alt=-1.0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert vehicle.store.get("write") == 0
    assert type(vehicle.store.get("write")) is int
    assert ctx.combus[ctx.vehicle_slot].status == 0


def test_alt_0_sets_health_and_write_0():
    vehicle, intercept, ctx = _ready(alt=0.0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert vehicle.store.get("write") == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0


def test_write_0_does_not_kill():
    vehicle, intercept, ctx = _ready(alt=-1.0, write=0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert vehicle.store.get("write") == 0
    assert ctx.combus[ctx.vehicle_slot].status == 1


def test_combus_none_sets_health_without_raising():
    vehicle, intercept, ctx = _ready(alt=-1.0, combus=None)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert vehicle.store.get("write") == 0


def test_modes_diagnostic():
    # mguide=5, maut=53 → mauty=5, mautp=3, mprop=4 → 5*1000+5*100+3*10+4
    vehicle, intercept, ctx = _ready(alt=100.0, mguide=5, maut=53, mprop=4)
    intercept.execute(vehicle, ctx)
    assert vehicle.store.get("modes") == 5534
    assert type(vehicle.store.get("modes")) is int
    assert vehicle.health == 1


def test_modes_zero_flags():
    vehicle, intercept, ctx = _ready(alt=100.0, mguide=0, maut=0, mprop=0)
    intercept.execute(vehicle, ctx)
    assert vehicle.store.get("modes") == 0


def test_never_calls_sys_exit():
    def boom(*args, **kwargs):
        raise AssertionError("sys.exit called")

    with patch("sys.exit", side_effect=boom):
        vehicle, intercept, ctx = _ready(alt=100.0)
        intercept.execute(vehicle, ctx)
        assert vehicle.health == 1
        vehicle, intercept, ctx = _ready(alt=-1.0)
        intercept.execute(vehicle, ctx)
        assert vehicle.health == 0
