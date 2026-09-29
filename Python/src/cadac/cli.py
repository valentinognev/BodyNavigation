import argparse
from dataclasses import dataclass, field
from pathlib import Path

from cadac.constants import EPS
from cadac.io.deck import load_deck
from cadac.stoch import seed
from cadac.io.plot import PLOT_COLUMNS, column_modules, flagged_plot_columns, plot_row, write_plot_csv
from cadac.io.scenario import load_scenario
from cadac.io.doc import document_input, open_doc_stream, write_vehicle_doc
from cadac.io.stat import merge_stat_files, open_stat_streams, write_stat_data
from cadac.io.merge import merge_plot_files, open_plot_streams, write_plot_data
from cadac.io.tabout import (
    open_tabout_stream,
    scrn_stems,
    write_tabout_banner,
    write_tabout_data,
)
from cadac.io.scrn import (
    default_scrn_stream,
    write_scrn_banner,
    write_scrn_data,
)
from cadac.io.comscrn import open_comscrn_stream, write_comscrn_data
from cadac.io.traj import (
    open_traj_stream,
    packets_from_vehicles,
    write_traj_banner,
    write_traj_data,
)
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
from cadac.vehicles.flat5.sraam5.vehicle import Sraam5
from cadac.vehicles.flat6.sraam6.target import Sraam6Target
from cadac.vehicles.flat6.sraam6.vehicle import Sraam6Missile
from cadac.vehicles.round3.rocket3.vehicle import Rocket3

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
    ("sraam5", "SRAAM5"),
    ("rocket3", "ROCKET3"),
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
register_family_type("sraam5", "SRAAM5", Sraam5)
register_family_type("rocket3", "ROCKET3", Rocket3)


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


def make_run_progress(end_time, on_progress=None):
    last_bucket = None

    def consider(sim_time, *, final=False):
        nonlocal last_bucket
        if on_progress is None:
            return
        if end_time <= 0:
            if last_bucket is not None:
                return
            last_bucket = 100
            on_progress(1.0)
            return
        fraction = min(1.0, max(0.0, sim_time / end_time))
        bucket = int(fraction * 100)
        if not final and last_bucket is not None and bucket == last_bucket:
            return
        last_bucket = bucket
        on_progress(fraction)

    return consider


def make_plot_on_step(
    plot_rows, plot_step, nveh, columns_fn=None, tracks=None, ploti_streams=None
):
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
            if ploti_streams is not None:
                slot = ctx.vehicle_slot
                stream = ploti_streams[slot] if slot < len(ploti_streams) else None
                if stream is not None:
                    # Rotor::plot_data ignores merge; timed rows always use real time
                    write_plot_data(stream, vehicle.store, merge=False)
                    stream.flush()
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


def _wrap_event_epoch(vehicle):
    """Mirror C++ event(): set vehicle.event_epoch from EventEngine.evaluate result."""
    engine = getattr(vehicle, "events", None)
    if engine is None:
        vehicle.event_epoch = False
        return
    orig = engine.evaluate

    def evaluate(store, _orig=orig, _vehicle=vehicle):
        fired = _orig(store)
        _vehicle.event_epoch = bool(fired)
        return fired

    engine.evaluate = evaluate
    vehicle.event_epoch = False


def _has_scrn_outputs(vehicle) -> bool:
    store = getattr(vehicle, "store", None)
    return store is not None and bool(scrn_stems(store))


def _run_scenario_once(
    path,
    cfg,
    *,
    on_progress=None,
    nmc=0,
    stat_state=None,
    plot_state=None,
    tabout_state=None,
    traj_state=None,
    comscrn_state=None,
):
    """One Monte Carlo / deterministic iteration (vehicles rebuilt each pass)."""
    try:
        int_step = float(cfg.timing["int_step"])
    except KeyError as exc:
        raise ValueError(f"{path}: int_step missing") from exc
    if int_step <= 0:
        raise ValueError(f"{path}: int_step {int_step}")
    plot_step = float(cfg.timing.get("plot_step", int_step))
    scrn_step = float(cfg.timing.get("scrn_step", int_step))
    traj_step = float(cfg.timing.get("traj_step", int_step))
    com_step = float(cfg.timing.get("com_step", int_step))
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

    # C++ y_doc: Vehicle::document → doc.asc once per type; document_input once at nmc==0
    if bool(cfg.options.get("doc")) and nmc == 0:
        _, doc_stream = open_doc_stream(path.parent)
        documented_types: set[str] = set()
        all_entries = []
        try:
            for vehicle in vehicles:
                vtype = getattr(vehicle, "type", "")
                if not vtype or vtype in documented_types:
                    continue
                documented_types.add(vtype)
                all_entries.extend(
                    write_vehicle_doc(doc_stream, vtype, cfg.title, vehicle.store)
                )
            document_input(path.parent, all_entries)
        finally:
            doc_stream.close()

    want_stat = bool(cfg.options.get("stat"))
    stati_write_term = [True] * len(vehicles)
    if want_stat:
        for vehicle in vehicles:
            _wrap_event_epoch(vehicle)
        if stat_state is not None and stat_state.get("streams") is None:
            paths, streams = open_stat_streams(path.parent, cfg.title, vehicles)
            stat_state["paths"] = paths
            stat_state["streams"] = streams

    want_plot_asc = bool(cfg.options.get("plot"))
    if want_plot_asc and plot_state is not None and plot_state.get("streams") is None:
        paths, streams = open_plot_streams(path.parent, cfg.title, vehicles)
        plot_state["paths"] = paths
        plot_state["streams"] = streams

    want_scrn = bool(cfg.options.get("scrn"))
    scrn_stream = default_scrn_stream() if want_scrn else None
    want_tabout = bool(cfg.options.get("tabout"))
    tabout_stream = None
    scrn_time = 0.0
    if want_scrn and scrn_stream is not None:
        # C++: one_screen_banner once, then scrn_data for each vehicle with outputs
        banner_written = False
        for vehicle in vehicles:
            if not _has_scrn_outputs(vehicle):
                continue
            if not banner_written:
                write_scrn_banner(
                    scrn_stream,
                    getattr(vehicle, "type", ""),
                    vehicle.store,
                )
                banner_written = True
            write_scrn_data(
                scrn_stream, getattr(vehicle, "name", ""), vehicle.store
            )
        scrn_stream.flush()
    if want_tabout and tabout_state is not None:
        if tabout_state.get("stream") is None:
            _tabout_path, stream = open_tabout_stream(path.parent)
            tabout_state["path"] = _tabout_path
            tabout_state["stream"] = stream
        tabout_stream = tabout_state["stream"]
        banner_written = False
        nmonte = int(getattr(cfg, "nmonte", 0))
        for vehicle in vehicles:
            if not _has_scrn_outputs(vehicle):
                continue
            if not banner_written:
                write_tabout_banner(
                    tabout_stream,
                    cfg.title,
                    getattr(vehicle, "type", ""),
                    vehicle.store,
                    nmonte=nmonte,
                    nmc=nmc,
                )
                banner_written = True
            write_tabout_data(
                tabout_stream, getattr(vehicle, "name", ""), vehicle.store
            )
        tabout_stream.flush()

    want_traj = bool(cfg.options.get("traj"))
    traj_stream = None
    traj_time = 0.0
    if want_traj and traj_state is not None:
        if traj_state.get("stream") is None:
            _traj_path, stream = open_traj_stream(path.parent)
            traj_state["path"] = _traj_path
            traj_state["stream"] = stream
        traj_stream = traj_state["stream"]
        init_packets = packets_from_vehicles(vehicles)
        if traj_state.get("banner_written") is not True:
            write_traj_banner(traj_stream, cfg.title, init_packets)
            traj_state["banner_written"] = True
        write_traj_data(traj_stream, init_packets, merge=False)
        traj_stream.flush()

    want_comscrn = bool(cfg.options.get("comscrn"))
    comscrn_stream = None
    com_time = 0.0
    if want_comscrn and comscrn_state is not None:
        if comscrn_state.get("stream") is None:
            _comscrn_path, stream = open_comscrn_stream(path.parent)
            comscrn_state["path"] = _comscrn_path
            comscrn_state["stream"] = stream
        comscrn_stream = comscrn_state["stream"]
        # C++ writes combus to screen at time=0 after module initialization
        write_comscrn_data(comscrn_stream, packets_from_vehicles(vehicles), 0.0)
        comscrn_stream.flush()

    plot_rows = []
    nveh = len(vehicles)
    csv_columns = _plot_columns(vehicles[0]) if vehicles else list(PLOT_COLUMNS)
    track_rows = [[] for _ in vehicles]
    ploti_streams = (
        (plot_state.get("streams") or [])
        if want_plot_asc and plot_state is not None
        else None
    )
    plot_on_step = make_plot_on_step(
        plot_rows, plot_step, nveh, tracks=track_rows, ploti_streams=ploti_streams
    )
    consider_progress = make_run_progress(cfg.end_time, on_progress)

    def on_step(vehicle, ctx):
        nonlocal scrn_time, traj_time, com_time
        plot_on_step(vehicle, ctx)
        if want_scrn or want_tabout:
            # C++: fabs(scrn_time-sim_time)<(int_step/2+EPS); shared clock for
            # y_scrn / y_tabout; advance on last vehicle index even when that
            # slot is an empty stub (AIM5 AIRCRAFT3 last).
            if abs(scrn_time - ctx.sim_time) < (ctx.int_step / 2 + EPS):
                if _has_scrn_outputs(vehicle):
                    if want_scrn and scrn_stream is not None:
                        write_scrn_data(
                            scrn_stream,
                            getattr(vehicle, "name", ""),
                            vehicle.store,
                        )
                        scrn_stream.flush()
                    if want_tabout and tabout_stream is not None:
                        write_tabout_data(
                            tabout_stream,
                            getattr(vehicle, "name", ""),
                            vehicle.store,
                        )
                        tabout_stream.flush()
                if ctx.vehicle_slot == nveh - 1:
                    scrn_time += scrn_step * (1.0 + ctx.out_fact)
        if want_traj and traj_stream is not None and ctx.combus is not None:
            # C++ writes traj after the full vehicle loop for this sim_time.
            if ctx.vehicle_slot == nveh - 1:
                if abs(traj_time - ctx.sim_time) < (ctx.int_step / 2 + EPS):
                    write_traj_data(traj_stream, list(ctx.combus), merge=False)
                    traj_stream.flush()
                    traj_time += traj_step * (1.0 + ctx.out_fact)
        if want_comscrn and comscrn_stream is not None and ctx.combus is not None:
            # C++ writes comscrn after the full vehicle loop for this sim_time.
            if ctx.vehicle_slot == nveh - 1:
                if abs(com_time - ctx.sim_time) < (ctx.int_step / 2 + EPS):
                    write_comscrn_data(
                        comscrn_stream, list(ctx.combus), ctx.sim_time
                    )
                    comscrn_stream.flush()
                    com_time += com_step * (1.0 + ctx.out_fact)
        if want_stat and stat_state is not None:
            slot = ctx.vehicle_slot
            streams = stat_state.get("streams") or []
            stream = streams[slot] if slot < len(streams) else None
            if stream is not None:
                # C++: write on event_epoch, and once when status drops to 0
                status = 1
                if ctx.combus is not None and slot < len(ctx.combus):
                    status = ctx.combus[slot].status
                if getattr(vehicle, "event_epoch", False):
                    write_stat_data(stream, vehicle.store, nmc, slot)
                if status == 0 and stati_write_term[slot]:
                    stati_write_term[slot] = False
                    write_stat_data(stream, vehicle.store, nmc, slot)
                stream.flush()
        if on_progress is None:
            return
        if nveh and ctx.vehicle_slot != nveh - 1:
            return
        consider_progress(ctx.sim_time, final=ctx.sim_time > cfg.end_time)

    run_loop(
        vehicles,
        modules_by_vehicle,
        module_order,
        cfg.end_time,
        int_step,
        on_step=on_step,
        nmonte=int(getattr(cfg, "nmonte", 0)),
    )
    if want_traj and traj_stream is not None:
        # C++ post-loop: traj_merge=true writes time=-1.0 endblock
        final_packets = packets_from_vehicles(vehicles)
        write_traj_data(traj_stream, final_packets, merge=True)
        traj_stream.flush()
    if want_plot_asc and ploti_streams is not None:
        # C++ post-loop: plot_merge=true; Rotor::plot_data ignores merge
        for slot, vehicle in enumerate(vehicles):
            stream = ploti_streams[slot] if slot < len(ploti_streams) else None
            if stream is None:
                continue
            honor_merge = type(vehicle) is not Rotor
            write_plot_data(stream, vehicle.store, merge=honor_merge)
            stream.flush()
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


def run_scenario(path, *, on_progress=None):
    # C++ execution.cpp Monte Carlo loop:
    #   do { if(!nmc) srand(iseed); ... execute ...; nmc++; } while(nmc < nmonte);
    # nmonte==0 → one pass (means); iseed is not bumped between MC runs.
    path = Path(path)
    cfg = load_scenario(path)
    nmonte = int(getattr(cfg, "nmonte", 0))
    nmc = 0
    result = None
    # C++ opens stati.asc / ploti.asc / tabout.asc / traj.asc once (nmc==0) and appends across MC.
    # comscrn.asc is the Python text artifact for C++ cout comscrn_data dumps.
    stat_state = {"paths": None, "streams": None} if cfg.options.get("stat") else None
    plot_state = (
        {"paths": None, "streams": None} if cfg.options.get("plot") else None
    )
    tabout_state = (
        {"path": None, "stream": None} if cfg.options.get("tabout") else None
    )
    traj_state = (
        {"path": None, "stream": None, "banner_written": False}
        if cfg.options.get("traj")
        else None
    )
    comscrn_state = (
        {"path": None, "stream": None} if cfg.options.get("comscrn") else None
    )
    try:
        while True:
            if nmc == 0:
                seed(cfg.iseed)
            result = _run_scenario_once(
                path,
                cfg,
                on_progress=on_progress,
                nmc=nmc,
                stat_state=stat_state,
                plot_state=plot_state,
                tabout_state=tabout_state,
                traj_state=traj_state,
                comscrn_state=comscrn_state,
            )
            nmc += 1
            if not (nmc < nmonte):
                break
        if (
            plot_state is not None
            and cfg.options.get("merge")
            and plot_state.get("paths")
        ):
            plot_paths = [p for p in plot_state["paths"] if p is not None]
            if plot_paths:
                merge_plot_files(plot_paths, path.parent / "plot.asc", cfg.title)
        if (
            stat_state is not None
            and cfg.options.get("merge")
            and stat_state.get("paths")
        ):
            missile_paths = [p for p in stat_state["paths"] if p is not None]
            if missile_paths:
                merge_stat_files(missile_paths, path.parent / "stat.asc", cfg.title)
    finally:
        if plot_state is not None and plot_state.get("streams"):
            for stream in plot_state["streams"]:
                if stream is not None:
                    stream.close()
        if stat_state is not None and stat_state.get("streams"):
            for stream in stat_state["streams"]:
                if stream is not None:
                    stream.close()
        if tabout_state is not None and tabout_state.get("stream") is not None:
            tabout_state["stream"].close()
        if traj_state is not None and traj_state.get("stream") is not None:
            traj_state["stream"].close()
        if comscrn_state is not None and comscrn_state.get("stream") is not None:
            comscrn_state["stream"].close()
    return result


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
