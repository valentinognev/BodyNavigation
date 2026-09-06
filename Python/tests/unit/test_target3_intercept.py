from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import StateStore
from cadac.vehicles.hyper5.target import Target3, Target3Intercept

RTOL = 1e-12
ATOL = 1e-14

INTERCEPT_FIELDS = {
    "targ_health": ("int", "diag", 0, ()),
}

NOT_DEFINED = (
    "write",
    "halt",
    "miss",
    "hit_time",
    "MISS_G",
    "time_m",
    "SBTGM",
    "STMEG",
    "SBMEG",
    "mseeker",
    "range_go",
    "STBG",
    "stop_run",
    "time",
    "alt",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()
        self.health = 1


def _ctx(status=1, vehicle_slot=0, extra=None):
    combus = [
        Packet(name="Truck_t1", type="TARGET3", status=status, vars={}),
    ]
    if extra:
        combus.extend(extra)
    return SimContext(
        sim_time=0.0,
        int_step=0.05,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=vehicle_slot,
    )


def _ready(status=1):
    vehicle = _Vehicle()
    intercept = Target3Intercept()
    intercept.define(vehicle)
    intercept.initialize(vehicle, _ctx(status=status))
    return vehicle, intercept


def test_name_is_intercept():
    assert Target3Intercept().name == "intercept"


def test_define_registers_cpp_def_intercept_fields():
    vehicle = _Vehicle()
    Target3Intercept().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(INTERCEPT_FIELDS)
    for name, (ftype, role, default, outputs) in INTERCEPT_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "intercept"
        assert field.outputs == outputs
        assert store.get(name) == default
        assert type(store.get(name)) is int


def test_define_does_not_register_hyper_or_plant():
    vehicle = _Vehicle()
    Target3Intercept().define(vehicle)
    for name in NOT_DEFINED:
        assert name not in vehicle.store.names()


def test_initialize_is_pass():
    vehicle, _intercept = _ready()
    assert vehicle.store.get("targ_health") == 0
    assert vehicle.health == 1


def test_terminate_exists_and_is_pass():
    vehicle, intercept = _ready()
    vehicle.store.set("targ_health", 1)
    intercept.terminate(vehicle, _ctx())
    assert vehicle.store.get("targ_health") == 1
    assert vehicle.health == 1


def test_combus_status_0_sets_targ_health_0():
    vehicle, intercept = _ready()
    ctx = _ctx(status=0)
    intercept.execute(vehicle, ctx)
    assert vehicle.store.get("targ_health") == 0
    assert type(vehicle.store.get("targ_health")) is int
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 0


def test_combus_status_1_sets_targ_health_1():
    vehicle, intercept = _ready()
    ctx = _ctx(status=1)
    intercept.execute(vehicle, ctx)
    assert vehicle.store.get("targ_health") == 1


def test_combus_status_hit_sets_targ_health_minus_1():
    vehicle, intercept = _ready()
    ctx = _ctx(status=-1)
    intercept.execute(vehicle, ctx)
    assert vehicle.store.get("targ_health") == -1


def test_reads_combus_at_vehicle_slot():
    vehicle, intercept = _ready()
    extra = [Packet(name="other", type="TARGET3", status=0, vars={})]
    ctx = _ctx(status=1, vehicle_slot=1, extra=extra)
    intercept.execute(vehicle, ctx)
    assert vehicle.store.get("targ_health") == 0


def test_skip_cout(capsys):
    vehicle, intercept = _ready()
    intercept.execute(vehicle, _ctx(status=0))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_target3_intercept_combus_status_0():
    vehicle = Target3("Truck_t1")
    vehicle.define()
    ctx = _ctx(status=0)
    named = {m.name: m for m in vehicle.modules}
    named["intercept"].execute(vehicle, ctx)
    assert vehicle.store.get("targ_health") == 0
