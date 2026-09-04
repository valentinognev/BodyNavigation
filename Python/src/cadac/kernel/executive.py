from dataclasses import dataclass

from cadac.kernel.combus import Packet, packet_from_store


@dataclass
class SimContext:
    sim_time: float
    int_step: float
    event_time: float
    out_fact: float
    combus: object
    vehicle_slot: int


def run_loop(
    vehicles, modules_by_vehicle, module_order, end_time, int_step, on_step=None
):
    sim_time = 0.0
    times = []
    combus = [
        Packet(
            name=getattr(vehicle, "name", ""),
            type=getattr(vehicle, "type", ""),
            status=_health(vehicle),
            vars={},
        )
        for vehicle in vehicles
    ]
    while sim_time <= (end_time + int_step):
        times.append(sim_time)
        for slot, vehicle in enumerate(vehicles):
            event_time = getattr(vehicle, "event_time", 0.0)
            engine = getattr(vehicle, "events", None)
            store = getattr(vehicle, "store", None)
            if engine is not None and store is not None:
                if engine.evaluate(store):
                    event_time = 0.0
            vehicle.event_time = event_time
            ctx = SimContext(
                sim_time=sim_time,
                int_step=int_step,
                event_time=event_time,
                out_fact=0.0,
                combus=combus,
                vehicle_slot=slot,
            )
            if combus[slot].status == 1:
                named = {
                    module.name: module for module in modules_by_vehicle[vehicle]
                }
                for name in module_order:
                    named[name].execute(vehicle, ctx)
                int_step = ctx.int_step
                com_names = getattr(vehicle, "com_names", None)
                if com_names:
                    saved = combus[slot].status
                    packet = packet_from_store(store, com_names)
                    packet.name = getattr(vehicle, "name", "")
                    packet.type = getattr(vehicle, "type", "")
                    packet.status = saved
                    combus[slot] = packet
            if on_step is not None:
                on_step(vehicle, ctx)
            vehicle.event_time = event_time + int_step
        sim_time += int_step
    return times


def _health(vehicle):
    if hasattr(vehicle, "health"):
        return vehicle.health
    if hasattr(vehicle, "status"):
        return vehicle.status
    return 1
