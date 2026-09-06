import argparse
from dataclasses import dataclass
from pathlib import Path

from cadac.constants import EPS
from cadac.io.deck import load_deck
from cadac.io.plot import PLOT_COLUMNS, flagged_plot_columns, plot_row, write_plot_csv
from cadac.io.scenario import load_scenario
from cadac.kernel.executive import SimContext, run_loop
from cadac.tables.lookup import Datadeck
from cadac.vehicles.cruise3.vehicle import Cruise3
from cadac.vehicles.hyper5.satellite import Satellite3
from cadac.vehicles.hyper5.target import Target3
from cadac.vehicles.hyper5.vehicle import Hyper5
from cadac.vehicles.hyper6.vehicle import Hyper6
from cadac.vehicles.plane5.vehicle import Plane5
from cadac.vehicles.plane6.vehicle import Plane6

_VEHICLE_TYPES = {
    "CRUISE3": Cruise3,
    "PLANE": Plane5,
    "PLANE6": Plane6,
    "HYPER5": Hyper5,
    "HYPER6": Hyper6,
    "TARGET3": Target3,
    "SATELLITE3": Satellite3,
}
if "_VEHICLE_FAMILIES" not in globals():
    _VEHICLE_FAMILIES = {}
_NO_DECK_TYPES = frozenset({"TARGET3", "SATELLITE3"})


@dataclass
class RunResult:
    plot_rows: list[dict]


def _deck(path):
    return Datadeck.from_tables(load_deck(path))


def make_plot_on_step(plot_rows, plot_step, nveh, columns_fn=None):
    if columns_fn is None:
        columns_fn = _plot_columns
    plot_time = 0.0

    def on_step(vehicle, ctx):
        nonlocal plot_time
        if abs(plot_time - ctx.sim_time) < (ctx.int_step / 2 + EPS):
            if ctx.vehicle_slot == 0:
                plot_rows.append(plot_row(vehicle.store, columns=columns_fn(vehicle)))
            if ctx.vehicle_slot == nveh - 1:
                plot_time += plot_step * (1.0 + ctx.out_fact)

    return on_step


def _build_vehicle(path, spec):
    if spec.family is not None:
        cls = _VEHICLE_FAMILIES.get((spec.family, spec.type))
    else:
        cls = _VEHICLE_TYPES.get(spec.type)
    if cls is None:
        if spec.family is not None:
            raise ValueError(
                f"{path}: unknown vehicle family {spec.family!r} type {spec.type!r}"
            )
        raise ValueError(f"{path}: unknown vehicle type {spec.type!r}")
    if spec.type in _NO_DECK_TYPES:
        return cls(spec.name, spec.events)
    if spec.type == "HYPER5":
        if spec.aero_deck is None:
            raise ValueError(f"{path}: {spec.type} requires aero_deck")
        mprop = spec.params.get("mprop", 0)
        if mprop != 0 and spec.prop_deck is None:
            raise ValueError(f"{path}: {spec.type} requires prop_deck")
        prop = _deck(spec.prop_deck) if spec.prop_deck is not None else None
        return cls(spec.name, _deck(spec.aero_deck), prop, spec.events)
    if spec.aero_deck is None or spec.prop_deck is None:
        raise ValueError(f"{path}: {spec.type} requires aero_deck and prop_deck")
    return cls(
        spec.name,
        _deck(spec.aero_deck),
        _deck(spec.prop_deck),
        spec.events,
    )


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
        vehicle = _build_vehicle(path, spec)
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
    nveh = len(vehicles)
    csv_columns = _plot_columns(vehicles[0]) if vehicles else list(PLOT_COLUMNS)
    on_step = make_plot_on_step(plot_rows, plot_step, nveh)

    run_loop(
        vehicles,
        modules_by_vehicle,
        module_order,
        cfg.end_time,
        int_step,
        on_step=on_step,
    )
    if cfg.options["plot"] and cfg.options["csv"]:
        write_plot_csv(
            path.parent / "plot.csv",
            cfg.title,
            csv_columns,
            [[row[column] for column in csv_columns] for row in plot_rows],
        )
    return RunResult(plot_rows=plot_rows)


def _plot_columns(vehicle):
    if vehicle.type == "CRUISE3":
        return list(PLOT_COLUMNS)
    return flagged_plot_columns(vehicle.store)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cadac")
    sub = parser.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("scenario")
    args = parser.parse_args(argv)
    if args.command == "run":
        run_scenario(args.scenario)
