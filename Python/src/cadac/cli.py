import argparse
from dataclasses import dataclass
from pathlib import Path

from cadac.constants import EPS
from cadac.io.deck import load_deck
from cadac.io.scenario import load_scenario
from cadac.kernel.executive import SimContext, run_loop
from cadac.tables.lookup import Datadeck
from cadac.vehicles.cruise3.vehicle import Cruise3


@dataclass
class RunResult:
    plot_rows: list[dict]


def _deck(path):
    return Datadeck.from_tables(load_deck(path))


def _plot_row(store):
    row = {}
    for name in store.names():
        field = store.field(name)
        if "plot" not in field.outputs:
            continue
        if field.type == "vec":
            for i, component in enumerate(field.value, start=1):
                row[f"{name}{i}"] = float(component)
        else:
            row[name] = (
                int(field.value) if field.type == "int" else float(field.value)
            )
    return row


def run_scenario(path):
    path = Path(path)
    cfg = load_scenario(path)
    try:
        int_step = float(cfg.timing["int_step"])
    except KeyError as exc:
        raise ValueError(f"{path}: int_step missing") from exc
    if int_step <= 0:
        raise ValueError(f"{path}: int_step {int_step}")
    plot_step = float(cfg.timing.get("plot_step", int_step))
    phases = {module.name: module.phases for module in cfg.modules}
    module_order = [module.name for module in cfg.modules if "exec" in module.phases]

    vehicles = []
    modules_by_vehicle = {}
    for spec in cfg.vehicles:
        if spec.type != "CRUISE3":
            raise ValueError(f"{path}: unknown vehicle type {spec.type!r}")
        if spec.aero_deck is None or spec.prop_deck is None:
            raise ValueError(f"{path}: CRUISE3 requires aero_deck and prop_deck")
        vehicle = Cruise3(
            spec.name,
            _deck(spec.aero_deck),
            _deck(spec.prop_deck),
            spec.events,
        )
        vehicle.define()
        for name, value in spec.params.items():
            try:
                vehicle.store.set(name, value)
            except KeyError as exc:
                raise ValueError(f"{path}: unknown param {name!r}") from exc
        ctx = SimContext(
            sim_time=0.0,
            int_step=int_step,
            event_time=0.0,
            out_fact=0.0,
            combus=None,
            vehicle_slot=len(vehicles),
        )
        for module in vehicle.modules:
            if "init" in phases.get(module.name, ()):
                module.initialize(vehicle, ctx)
        vehicles.append(vehicle)
        modules_by_vehicle[vehicle] = vehicle.modules

    plot_rows = []
    plot_time = 0.0
    nveh = len(vehicles)

    def on_step(vehicle, ctx):
        nonlocal plot_time
        if abs(plot_time - ctx.sim_time) < (ctx.int_step / 2 + EPS):
            plot_rows.append(_plot_row(vehicle.store))
            if ctx.vehicle_slot == nveh - 1:
                plot_time += plot_step * (1.0 + ctx.out_fact)

    run_loop(
        vehicles,
        modules_by_vehicle,
        module_order,
        cfg.end_time,
        int_step,
        on_step=on_step,
    )
    return RunResult(plot_rows=plot_rows)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cadac")
    sub = parser.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("scenario")
    args = parser.parse_args(argv)
    if args.command == "run":
        run_scenario(args.scenario)
