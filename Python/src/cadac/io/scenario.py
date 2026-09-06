from dataclasses import dataclass
from pathlib import Path

from cadac.io.jsonc import loads
from cadac.kernel.events import EventSpec

OPTION_KEYS = (
    "scrn",
    "events",
    "plot",
    "doc",
    "csv",
    "tabout",
    "merge",
    "comscrn",
    "traj",
)


@dataclass
class ModuleSpec:
    name: str
    phases: list[str]


@dataclass
class VehicleSpec:
    type: str
    name: str
    aero_deck: Path | None
    prop_deck: Path | None
    params: dict
    events: list[EventSpec]
    family: str | None = None


@dataclass
class RunConfig:
    title: str
    options: dict[str, bool]
    modules: list[ModuleSpec]
    timing: dict[str, float]
    end_time: float
    vehicles: list[VehicleSpec]
    family: str | None = None


def _options(raw: dict, path: Path) -> dict[str, bool]:
    given = raw.get("options") or {}
    unknown = [key for key in given if key not in OPTION_KEYS]
    if unknown:
        raise ValueError(f"{path}: unknown option key {unknown[0]!r}")
    return {key: bool(given.get(key, False)) for key in OPTION_KEYS}


def _deck_path(parent: Path, value) -> Path | None:
    if value is None:
        return None
    return parent / Path(value)


def _vehicle(parent: Path, raw: dict, scenario_family: str | None = None) -> VehicleSpec:
    events = [
        EventSpec(when=event["when"], set=event["set"])
        for event in raw.get("events") or []
    ]
    return VehicleSpec(
        type=raw["type"],
        name=raw["name"],
        aero_deck=_deck_path(parent, raw.get("aero_deck")),
        prop_deck=_deck_path(parent, raw.get("prop_deck")),
        params=dict(raw.get("params") or {}),
        events=events,
        family=raw.get("family") or scenario_family,
    )


def load_scenario(path) -> RunConfig:
    path = Path(path)
    data = loads(path.read_text(encoding="utf-8"))
    parent = path.parent
    scenario_family = data.get("family")
    return RunConfig(
        title=data["title"],
        options=_options(data, path),
        modules=[
            ModuleSpec(name=module["name"], phases=list(module["phases"]))
            for module in data.get("modules") or []
        ],
        timing={key: float(value) for key, value in (data.get("timing") or {}).items()},
        end_time=float(data["end_time"]),
        vehicles=[_vehicle(parent, vehicle, scenario_family) for vehicle in data["vehicles"]],
        family=scenario_family,
    )
