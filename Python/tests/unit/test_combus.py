from cadac.kernel.combus import Packet, packet_from_store
from cadac.kernel.events import EventEngine
from cadac.kernel.executive import run_loop
from cadac.kernel.module import DummyModule
from cadac.kernel.state import Field, StateStore


class _Vehicle:
    def __init__(self):
        self.store = StateStore()
        self.store.define(Field("time", 0.0, "real", "exec", "environment"))
        self.events = EventEngine([])
        self.event_time = 0.0


class _Capture:
    name = "watch"

    def __init__(self):
        self.combus = None
        self.vars_during_execute = []

    def define(self, vehicle):
        pass

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        self.combus = ctx.combus
        packet = ctx.combus[ctx.vehicle_slot]
        if packet is None:
            self.vars_during_execute.append(None)
        else:
            self.vars_during_execute.append(dict(packet.vars))

    def terminate(self, vehicle, ctx):
        pass


def test_packet_two_named_vars():
    store = StateStore()
    store.define(Field("lonx", -80.55, "real", "state", "newton"))
    store.define(Field("latx", 28.45, "real", "state", "newton"))
    store.define(Field("alt", 3000.0, "real", "state", "newton"))
    packet = packet_from_store(store, ["lonx", "latx"])
    assert isinstance(packet, Packet)
    assert packet.vars["lonx"] == -80.55
    assert packet.vars["latx"] == 28.45
    assert "alt" not in packet.vars
    assert packet.status == 1
    assert type(packet.status) is int


def test_run_loop_publishes_packet_after_modules():
    vehicle = _Vehicle()
    vehicle.name = "c1"
    vehicle.type = "CRUISE3"
    vehicle.com_names = ["time"]
    watch = _Capture()
    run_loop(
        vehicles=[vehicle],
        modules_by_vehicle={vehicle: [DummyModule(), watch]},
        module_order=["dummy", "watch"],
        end_time=0.0,
        int_step=0.1,
    )
    assert type(watch.combus) is list
    packet = watch.combus[0]
    assert isinstance(packet, Packet)
    assert packet.name == "c1"
    assert packet.type == "CRUISE3"
    assert packet.status == 1
    assert packet.vars["time"] == 0.1
    assert watch.vars_during_execute[0] == {}
    assert watch.vars_during_execute[1]["time"] == 0.0


class _Kill:
    name = "kill"

    def define(self, vehicle):
        pass

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        ctx.combus[ctx.vehicle_slot].status = 0
        vehicle.store.set("time", 99.0)

    def terminate(self, vehicle, ctx):
        pass


def test_publish_restores_health_status():
    vehicle = _Vehicle()
    vehicle.name = "c1"
    vehicle.type = "CRUISE3"
    vehicle.com_names = ["time"]
    watch = _Capture()
    run_loop(
        vehicles=[vehicle],
        modules_by_vehicle={vehicle: [_Kill(), watch]},
        module_order=["kill", "watch"],
        end_time=0.0,
        int_step=0.1,
    )
    assert watch.combus[0].status == 0
    assert watch.combus[0].vars["time"] == 99.0
    assert len(watch.vars_during_execute) == 1


def test_packets_indexed_by_vehicle_slot():
    first = _Vehicle()
    first.name = "c1"
    first.type = "CRUISE3"
    first.store.define(Field("mark", 1, "int", "data", "test"))
    first.com_names = ["mark"]
    second = _Vehicle()
    second.name = "c2"
    second.type = "CRUISE3"
    second.store.define(Field("mark", 2, "int", "data", "test"))
    second.com_names = ["mark"]
    watch = _Capture()
    run_loop(
        vehicles=[first, second],
        modules_by_vehicle={first: [watch], second: [watch]},
        module_order=["watch"],
        end_time=0.0,
        int_step=0.1,
    )
    assert watch.combus[0].name == "c1"
    assert watch.combus[0].vars["mark"] == 1
    assert watch.combus[1].name == "c2"
    assert watch.combus[1].vars["mark"] == 2
