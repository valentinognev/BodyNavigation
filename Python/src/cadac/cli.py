import argparse
from dataclasses import dataclass, field
from pathlib import Path

from cadac.constants import EPS
from cadac.io.deck import load_deck
from cadac.stoch import seed
from cadac.io.plot import PLOT_COLUMNS, column_modules, flagged_plot_columns, plot_row, write_plot_csv
from cadac.io.scenario import load_scenario
from cadac.kernel.executive import SimContext, run_loop
from cadac.tables.lookup import Datadeck
from cadac.vehicles.flat6.agm6.aircraft import Agm6Aircraft
from cadac.vehicles.flat6.agm6.target import Agm6Target
from cadac.vehicles.flat6.agm6.vehicle import Agm6Missile
from cadac.vehicles.flat3.aim5.aircraft import Aim5Aircraft
from cadac.vehicles.flat3.aim5.vehicle import Aim5
from cadac.vehicles.round3.hyper3.vehicle import Cruise3
from cadac.vehicles.round3.cruise5.satellite import Cruise5Satellite
from cadac.vehicles.round3.cruise5.target import Cruise5Target
from cadac.vehicles.round3.cruise5.vehicle import Cruise5
from cadac.vehicles.round3.hyper5.satellite import Satellite3
from cadac.vehicles.round3.hyper5.target import Target3
from cadac.vehicles.round3.hyper5.vehicle import Hyper5
from cadac.vehicles.round6.hyper6.radar import Hyper6Radar
from cadac.vehicles.round6.hyper6.satellite import Hyper6Satellite
from cadac.vehicles.round6.hyper6.vehicle import Hyper6
from cadac.vehicles.flat3.falcon5.vehicle import Plane5
from cadac.vehicles.flat6.falcon6.vehicle import Plane6
from cadac.vehicles.round6.rocket6.vehicle import Rocket6
from cadac.vehicles.planar.magsix.vehicle import Rotor
from cadac.vehicles.flat6.sam6.aircraft import Sam6Aircraft
from cadac.vehicles.flat6.sam6.radar import Sam6Radar
from cadac.vehicles.flat6.sam6.rocket import Sam6Rocket
from cadac.vehicles.flat6.sam6.vehicle import Sam6Missile
from cadac.vehicles.flat6.sraam6.target import Sraam6Target
from cadac.vehicles.flat6.sraam6.vehicle import Sraam6Missile

_VEHICLE_TYPES = {
    "CRUISE3": Cruise3,
    "PLANE": Plane5,
    "PLANE6": Plane6,
    "HYPER5": Hyper5,
    "HYPER6": Hyper6,
    "TARGET3": Target3,
    "SATELLITE3": Satellite3,
    "AIM5": Aim5,
    "ROTOR": Rotor,
    "SAT3": Hyper6Satellite,
    "RADAR0": Hyper6Radar,
}
# SAM6 RADAR0 stays a family key and does not collide with this global.
_VEHICLE_FAMILIES: dict[tuple[str, str], type] = {
    ("aim5", "AIM5"): Aim5,
    ("aim5", "AIRCRAFT3"): Aim5Aircraft,
    ("cruise5", "CRUISE3"): Cruise5,
    ("cruise5", "TARGET3"): Cruise5Target,
    ("cruise5", "SATELLITE3"): Cruise5Satellite,
    ("magsix", "ROTOR"): Rotor,
    ("rocket6", "HYPER6"): Rocket6,
}
_NO_DECK_TYPES = frozenset({"TARGET3", "SATELLITE3", "ROTOR", "SAT3", "RADAR0"})
_NO_DECK_FAMILY_TYPES = {
    ("aim5", "AIRCRAFT3"),
    ("sraam6", "TARGET3"),
    ("agm6", "TARGET3"),
    ("agm6", "AIRCRAFT3"),
}


def register_family_type(family, type_name, cls) -> None:
    key = (family, type_name)
    existing = _VEHICLE_FAMILIES.get(key)
    if existing is cls:
        return
    if existing is not None:
        raise ValueError(
            f"family type {type_name!r} for family {family!r} already registered"
        )
    _VEHICLE_FAMILIES[key] = cls


register_family_type("sam6", "MISSILE6", Sam6Missile)
register_family_type("sam6", "AIRCRAFT3", Sam6Aircraft)
register_family_type("sam6", "ROCKET5", Sam6Rocket)
register_family_type("sam6", "RADAR0", Sam6Radar)
register_family_type("sraam6", "MISSILE6", Sraam6Missile)
register_family_type("sraam6", "TARGET3", Sraam6Target)
register_family_type("agm6", "MISSILE6", Agm6Missile)
register_family_type("agm6", "TARGET3", Agm6Target)
register_family_type("agm6", "AIRCRAFT3", Agm6Aircraft)


def _resolve_vehicle(family, vtype):
    if vtype == "GROUND0":
        raise ValueError("Ground0 is not a vehicle; its tracks are owned by RADAR0")
    if family:
        cls = _VEHICLE_FAMILIES.get((family, vtype))
        if cls is None:
            raise ValueError(f"unknown vehicle type {vtype!r} for family {family!r}")
        return cls
    cls = _VEHICLE_TYPES.get(vtype)
    if cls is None:
        raise ValueError(f"unknown vehicle type {vtype!r}")
    return cls


@dataclass
class RunResult:
    plot_rows: list[dict]
    tracks: list[dict] = field(default_factory=list)
    column_modules: dict = field(default_factory=dict)


def _deck(path):
    return Datadeck.from_tables(load_deck(path))


def _build_sam6_vehicle(path, spec, cls):
    if spec.type == "MISSILE6":
        if spec.aero_deck is None or spec.prop_deck is None:
            raise ValueError(f"{path}: {spec.type} requires aero_deck and prop_deck")
        return cls(
            spec.name,
            _deck(spec.aero_deck),
            _deck(spec.prop_deck),
            spec.events,
        )
    if spec.type == "AIRCRAFT3":
        return cls(spec.name, spec.events)
    if spec.type == "ROCKET5":
        if spec.aero_deck is None:
            raise ValueError(f"{path}: {spec.type} requires aero_deck")
        if spec.prop_deck is not None:
            raise ValueError(f"{path}: {spec.type} must not have prop_deck")
        return cls(spec.name, _deck(spec.aero_deck), spec.events)
    if spec.type == "RADAR0":
        mtrack = spec.params.get("mtrack", 0)
        if mtrack == 1 and (spec.sam_deck is None or spec.srmb_deck is None):
            raise ValueError(f"{path}: {spec.type} requires sam_deck and srmb_deck")
        sam = _deck(spec.sam_deck) if spec.sam_deck is not None else None
        srmb = _deck(spec.srmb_deck) if spec.srmb_deck is not None else None
        return cls(spec.name, spec.events, sam_deck=sam, srmb_deck=srmb)
    raise ValueError(
        f"{path}: unknown vehicle type {spec.type!r} for family {spec.family!r}"
    )


def track_names(names):
    counts = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    seen = {}
    labels = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        if counts[name] == 1:
            labels.append(name)
        else:
            labels.append(f"{name} {seen[name]}")
    return labels


def track_columns(store):
    if store is None:
        return None
    if all(name in store for name in ("latx", "lonx", "alt")):
        return ["latx", "lonx", "alt"]
    for stem in ("SBEL", "SAEL"):
        field = store.field(stem) if stem in store else None
        if field is not None and field.type == "vec":
            return [f"{stem}1", f"{stem}2", f"{stem}3"]
    return None


def series_columns(vehicle, columns_fn):
    store = getattr(vehicle, "store", None)
    if store is None:
        return None
    columns = []
    seen = set()

    def add(name):
        if name not in seen:
            seen.add(name)
            columns.append(name)

    for name in columns_fn(vehicle):
        add(name)
    for name in track_columns(store) or []:
        add(name)
    if not columns:
        return None
    return columns


def make_plot_on_step(plot_rows, plot_step, nveh, columns_fn=None, tracks=None):
    if columns_fn is None:
        columns_fn = _plot_columns
    plot_time = 0.0

    def on_step(vehicle, ctx):
        nonlocal plot_time
        if abs(plot_time - ctx.sim_time) < (ctx.int_step / 2 + EPS):
            if ctx.vehicle_slot == 0:
                plot_rows.append(plot_row(vehicle.store, columns=columns_fn(vehicle)))
            if tracks is not None:
                columns = series_columns(vehicle, columns_fn)
                if columns is not None:
                    row = plot_row(vehicle.store, columns=columns)
                    if "time" not in row:
                        row = {"time": ctx.sim_time, **row}
                    tracks[ctx.vehicle_slot].append(row)
            if ctx.vehicle_slot == nveh - 1:
                plot_time += plot_step * (1.0 + ctx.out_fact)

    return on_step


def _build_vehicle(path, spec):
    try:
        cls = _resolve_vehicle(spec.family, spec.type)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if spec.family == "sam6":
        return _build_sam6_vehicle(path, spec, cls)
    if spec.family == "agm6" and spec.type == "MISSILE6":
        if spec.aero_deck is None:
            raise ValueError(f"{path}: {spec.type} requires aero_deck")
        weather = _deck(spec.weather_deck) if spec.weather_deck is not None else None
        return cls(spec.name, _deck(spec.aero_deck), spec.events, weather)
    if (spec.family, spec.type) in _NO_DECK_FAMILY_TYPES or spec.type in _NO_DECK_TYPES:
        return cls(spec.name, spec.events)
    if cls is Rocket6:
        if spec.aero_deck is None:
            raise ValueError(f"{path}: {spec.type} requires aero_deck")
        prop = _deck(spec.prop_deck) if spec.prop_deck is not None else None
        weather = _deck(spec.weather_deck) if spec.weather_deck is not None else None
        return cls(
            spec.name,
            _deck(spec.aero_deck),
            spec.events,
            weather,
            prop,
        )
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
    seed(cfg.iseed)
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
    track_rows = [[] for _ in vehicles]
    on_step = make_plot_on_step(plot_rows, plot_step, nveh, tracks=track_rows)

    run_loop(
        vehicles,
        modules_by_vehicle,
        module_order,
        cfg.end_time,
        int_step,
        on_step=on_step,
    )
    if vehicles and type(vehicles[0]) is Rotor:
        # MAGSIX Rotor::plot_data ignores merge, so the C++ post-loop dump is a real row.
        plot_rows.append(plot_row(vehicles[0].store, columns=csv_columns))
        rotor_columns = series_columns(vehicles[0], _plot_columns)
        if rotor_columns is not None:
            track_rows[0].append(plot_row(vehicles[0].store, columns=rotor_columns))
    if cfg.options["plot"] and cfg.options["csv"]:
        write_plot_csv(
            path.parent / "plot.csv",
            cfg.title,
            csv_columns,
            [[row[column] for column in csv_columns] for row in plot_rows],
        )
    labels = track_names(
        [
            getattr(vehicle, "name", "") or getattr(vehicle, "type", "") or f"vehicle {slot + 1}"
            for slot, vehicle in enumerate(vehicles)
        ]
    )
    tracks = []
    for slot, _vehicle in enumerate(vehicles):
        if not track_rows[slot]:
            continue
        columns = list(track_rows[slot][0].keys())
        tracks.append(
            {
                "name": labels[slot],
                "columns": columns,
                "rows": track_rows[slot],
                "modules": column_modules(vehicles[slot].store, columns),
            }
        )
    return RunResult(
        plot_rows=plot_rows,
        tracks=tracks,
        column_modules=column_modules(vehicles[0].store, csv_columns) if vehicles else {},
    )


def _plot_columns(vehicle):
    if type(vehicle) is Cruise3:
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
