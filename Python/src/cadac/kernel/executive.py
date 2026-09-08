from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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
    vehicles: Sequence[Any],
    modules_by_vehicle: Mapping[Any, Sequence[Any]],
    module_order: Sequence[str],
    end_time: float,
    int_step: float,
    on_step: Callable[..., Any] | None = None,
) -> list[float]:
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
    for slot, vehicle in enumerate(vehicles):
        _publish(combus, slot, vehicle)
    chains = []
    for vehicle in vehicles:
        named = {module.name: module for module in modules_by_vehicle[vehicle]}
        chains.append(tuple(named[name] for name in module_order if name in named))
    contexts = [
        SimContext(
            sim_time=0.0,
            int_step=int_step,
            event_time=0.0,
            out_fact=0.0,
            combus=combus,
            vehicle_slot=slot,
        )
        for slot, vehicle in enumerate(vehicles)
    ]
    while sim_time <= (end_time + int_step):
        times.append(sim_time)
        for slot, vehicle in enumerate(vehicles):
            event_time = getattr(vehicle, "event_time", 0.0)
            engine = getattr(vehicle, "events", None)
            store = getattr(vehicle, "store", None)
            if engine is not None and store is not None and engine.evaluate(store):
                event_time = 0.0
            vehicle.event_time = event_time
            ctx = contexts[slot]
            ctx.sim_time = sim_time
            ctx.int_step = int_step
            ctx.event_time = event_time
            if combus[slot].status == 1:
                for module in chains[slot]:
                    module.execute(vehicle, ctx)
                int_step = ctx.int_step
                _publish(combus, slot, vehicle)
            if on_step is not None:
                on_step(vehicle, ctx)
            vehicle.event_time = event_time + int_step
        sim_time += int_step
    return times


def _publish(combus: list[Packet], slot: int, vehicle: Any) -> None:
    com_names = getattr(vehicle, "com_names", None)
    if not com_names:
        return
    store = getattr(vehicle, "store", None)
    if store is None:
        return
    saved = combus[slot].status
    packet = packet_from_store(store, com_names)
    packet.name = getattr(vehicle, "name", "")
    packet.type = getattr(vehicle, "type", "")
    packet.status = saved
    combus[slot] = packet


def _health(vehicle: Any) -> int:
    if hasattr(vehicle, "health"):
        return vehicle.health
    if hasattr(vehicle, "status"):
        return vehicle.status
    return 1
