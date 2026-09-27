import numpy as np

from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat3.falcon5.intercept import Plane5Intercept

SWBL = np.array([100.0, 200.0, 50.0])
TIME = 12.0
PSIVLX = 45.0
THTVLX = -5.0


class _Vehicle:
    def __init__(self):
        self.store = StateStore()
        self.health = 1


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.05,
        event_time=0.0,
        out_fact=0.0,
        combus=[Packet(name="p1", type="PLANE", status=1, vars={})],
        vehicle_slot=0,
    )


def _register_inputs(
    store,
    write=1,
    mguidance=30,
    wp_flag=-1,
    swbl=SWBL,
    time=TIME,
    psivlx=PSIVLX,
    thtvlx=THTVLX,
):
    store.define(Field("write", write, "int", "save", "guidance", ("scrn", "plot")))
    store.define(Field("mguidance", mguidance, "int", "data", "guidance", ("scrn",)))
    store.define(Field("wp_flag", wp_flag, "int", "dia", "guidance", ("plot", "scrn")))
    store.define(Field("SWBL", swbl, "vec", "out", "guidance"))
    store.define(Field("time", time, "real", "exec", "kinematics", ("scrn", "plot")))
    store.define(Field("psivlx", psivlx, "real", "out", "newton", ("scrn", "plot")))
    store.define(Field("thtvlx", thtvlx, "real", "out", "newton", ("scrn", "plot")))


def _ready(
    stop_run=1,
    write=1,
    mguidance=30,
    wp_flag=-1,
    swbl=SWBL,
    time=TIME,
    psivlx=PSIVLX,
    thtvlx=THTVLX,
):
    vehicle = _Vehicle()
    intercept = Plane5Intercept()
    intercept.define(vehicle)
    vehicle.store.set("stop_run", stop_run)
    _register_inputs(
        vehicle.store,
        write=write,
        mguidance=mguidance,
        wp_flag=wp_flag,
        swbl=swbl,
        time=time,
        psivlx=psivlx,
        thtvlx=thtvlx,
    )
    return vehicle, intercept, _ctx()


def test_name_is_intercept():
    assert Plane5Intercept().name == "intercept"


def test_define_registers_stop_run_only():
    vehicle = _Vehicle()
    Plane5Intercept().define(vehicle)
    store = vehicle.store
    field = store.field("stop_run")
    assert field.type == "int"
    assert field.role == "data"
    assert field.module == "intercept"
    assert field.outputs == ()
    assert field.value == 0
    assert type(field.value) is int
    assert store.names() == ["stop_run"]


def test_stop_run_1_mguidance_30_sets_packet_status_0():
    vehicle, intercept, ctx = _ready(stop_run=1, mguidance=30, write=1, wp_flag=-1)
    intercept.execute(vehicle, ctx)
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert vehicle.health == 0
    assert vehicle.store.get("write") == 0


def test_stop_run_1_mguidance_40_sets_packet_status_0():
    vehicle, intercept, ctx = _ready(stop_run=1, mguidance=40, write=1, wp_flag=-1)
    intercept.execute(vehicle, ctx)
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert vehicle.health == 0
    assert vehicle.store.get("write") == 0


def test_stop_run_1_mguidance_33_sets_packet_status_0():
    vehicle, intercept, ctx = _ready(stop_run=1, mguidance=33, write=1, wp_flag=-1)
    intercept.execute(vehicle, ctx)
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert vehicle.health == 0
    assert vehicle.store.get("write") == 0


def test_stop_run_0_mguidance_30_clears_write_but_does_not_kill():
    vehicle, intercept, ctx = _ready(stop_run=0, mguidance=30, write=1, wp_flag=-1)
    intercept.execute(vehicle, ctx)
    assert vehicle.store.get("write") == 0
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1


def test_stop_run_0_mguidance_40_clears_write_but_does_not_kill():
    vehicle, intercept, ctx = _ready(stop_run=0, mguidance=40, write=1, wp_flag=-1)
    intercept.execute(vehicle, ctx)
    assert vehicle.store.get("write") == 0
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1


def test_stop_run_0_mguidance_33_clears_write_but_does_not_kill():
    vehicle, intercept, ctx = _ready(stop_run=0, mguidance=33, write=1, wp_flag=-1)
    intercept.execute(vehicle, ctx)
    assert vehicle.store.get("write") == 0
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1


def test_write_zero_does_not_stop_even_if_stop_run_1():
    vehicle, intercept, ctx = _ready(stop_run=1, mguidance=30, write=0, wp_flag=-1)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 0


def test_wp_flag_plus_one_does_not_clear_write_or_stop():
    vehicle, intercept, ctx = _ready(stop_run=1, mguidance=30, write=1, wp_flag=1)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 1


def test_wp_flag_zero_does_not_clear_write_or_stop():
    vehicle, intercept, ctx = _ready(stop_run=1, mguidance=33, write=1, wp_flag=0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 1


def test_other_mguidance_does_not_stop_or_clear_write():
    vehicle, intercept, ctx = _ready(stop_run=1, mguidance=0, write=1, wp_flag=-1)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert vehicle.store.get("write") == 1


def test_stop_run_not_one_does_not_kill_on_waypoint_cross():
    vehicle, intercept, ctx = _ready(stop_run=2, mguidance=30, write=1, wp_flag=-1)
    intercept.execute(vehicle, ctx)
    assert vehicle.store.get("write") == 0
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1


def test_always_writes_write_back_when_unchanged():
    vehicle, intercept, ctx = _ready(stop_run=0, mguidance=30, write=1, wp_flag=1)
    intercept.execute(vehicle, ctx)
    assert vehicle.store.get("write") == 1
    assert type(vehicle.store.get("write")) is int
