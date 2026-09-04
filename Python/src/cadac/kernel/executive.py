from dataclasses import dataclass


@dataclass
class SimContext:
    sim_time: float
    int_step: float
    event_time: float
    out_fact: float
    combus: object
    vehicle_slot: int


def run_loop(vehicles, modules_by_vehicle, module_order, end_time, int_step):
    sim_time = 0.0
    times = []
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
            if _health(vehicle) == 1:
                ctx = SimContext(
                    sim_time=sim_time,
                    int_step=int_step,
                    event_time=event_time,
                    out_fact=0.0,
                    combus=None,
                    vehicle_slot=slot,
                )
                named = {
                    module.name: module for module in modules_by_vehicle[vehicle]
                }
                for name in module_order:
                    named[name].execute(vehicle, ctx)
            vehicle.event_time = event_time + int_step
        sim_time += int_step
    return times


def _health(vehicle):
    if hasattr(vehicle, "health"):
        return vehicle.health
    if hasattr(vehicle, "status"):
        return vehicle.status
    return 1
