import json
import math
from pathlib import Path

import pytest

from cadac import run_scenario
from cadac.cli import _VEHICLE_FAMILIES, _VEHICLE_TYPES, _build_vehicle, _resolve_vehicle
from cadac.io.jsonc import loads
from cadac.io.scenario import VehicleSpec, load_scenario
from cadac.io.translate import deck_asc_to_jsonc, translate_scenario_asc
from cadac.kernel.executive import SimContext, run_loop
from cadac.kernel.state import Field
from cadac.vehicles.hyper5.target import Target3

ROOT = Path(__file__).resolve().parents[3]
AGM6_ASC = ROOT / "CADAC_Simulations/AGM6_250217/AGM6"
CASES = Path(__file__).resolve().parents[2] / "cases" / "agm6"
FREE_FLIGHT_ASC = AGM6_ASC / "input_3_1 AGM6 Free Flight.asc"

MODULE_ORDER = [
    "environment",
    "kinematics",
    "aerodynamics",
    "propulsion",
    "forces",
    "euler",
    "newton",
    "ins",
    "datalink",
    "sensor",
    "guidance",
    "control",
    "actuator",
    "intercept",
]


def _freeflight_short(tmp_path: Path) -> Path:
    translate_scenario_asc(FREE_FLIGHT_ASC, tmp_path, family="agm6")
    deck_asc_to_jsonc(
        AGM6_ASC / "AGM6_aero_deck.asc", tmp_path / "AGM6_aero_deck.jsonc"
    )
    src = tmp_path / "input_3_1 AGM6 Free Flight.jsonc"
    path = tmp_path / "input_freeflight.jsonc"
    data = loads(src.read_text(encoding="utf-8"))
    data["end_time"] = 0.1
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def _jsonc(tmp_path: Path, name: str, vehicles: list, **extra) -> Path:
    payload = {
        "title": "t",
        "options": extra.get("options", {}),
        "modules": extra.get("modules", []),
        "timing": extra.get("timing", {"int_step": 0.001, "plot_step": 0.1}),
        "end_time": extra.get("end_time", 0),
        "vehicles": vehicles,
    }
    if "family" in extra:
        payload["family"] = extra["family"]
    path = tmp_path / name
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8", newline="\n")
    return path


def test_agm6_type_health_and_module_order():
    from cadac.vehicles.agm6.vehicle import Agm6Missile

    vehicle = Agm6Missile("AGM6", None)
    assert vehicle.type == "MISSILE6"
    assert vehicle.health == 1
    assert [module.name for module in vehicle.modules] == MODULE_ORDER


def test_agm6_define_registers_modules_omitted_from_freeflight_asc():
    from cadac.vehicles.agm6.vehicle import Agm6Missile

    vehicle = Agm6Missile("AGM6", None)
    vehicle.define()
    assert "mins" in vehicle.store.names()
    assert "mnav" in vehicle.store.names()
    assert "mseek" in vehicle.store.names()
    assert "mguid" in vehicle.store.names()
    assert "maut" in vehicle.store.names()
    assert "mact" in vehicle.store.names()
    assert "mterm" in vehicle.store.names()


def test_agm6_define_skips_existing_field():
    from cadac.vehicles.agm6.vehicle import Agm6Missile

    vehicle = Agm6Missile("AGM6", None)
    vehicle.store.define(Field("hbe", 42.0, "real", "out", "pre", ("plot",)))
    vehicle.define()
    assert vehicle.store.get("hbe") == 42.0
    assert vehicle.store.field("hbe").module == "pre"


def test_agm6_com_names_from_com_flags():
    from cadac.vehicles.agm6.vehicle import Agm6Missile

    vehicle = Agm6Missile("AGM6", None)
    vehicle.define()
    assert "SBEL" in vehicle.com_names
    assert "hbe" not in vehicle.com_names


def test_agm6_environment_receives_weather_deck():
    from cadac.vehicles.agm6.vehicle import Agm6Missile

    sentinel = object()
    vehicle = Agm6Missile("AGM6", None, weather_deck=sentinel)
    assert vehicle.modules[0].name == "environment"
    assert vehicle.modules[0].weather_deck is sentinel


def test_family_agm6_types_registered_not_global():
    from cadac.vehicles.agm6.aircraft import Agm6Aircraft
    from cadac.vehicles.agm6.target import Agm6Target
    from cadac.vehicles.agm6.vehicle import Agm6Missile

    assert _VEHICLE_FAMILIES[("agm6", "MISSILE6")] is Agm6Missile
    assert _VEHICLE_FAMILIES[("agm6", "TARGET3")] is Agm6Target
    assert _VEHICLE_FAMILIES[("agm6", "AIRCRAFT3")] is Agm6Aircraft
    assert _resolve_vehicle("agm6", "MISSILE6") is Agm6Missile
    assert "MISSILE6" not in _VEHICLE_TYPES
    assert "AIRCRAFT3" not in _VEHICLE_TYPES
    assert _VEHICLE_TYPES["TARGET3"] is Target3


def test_no_family_missile6_still_unknown():
    with pytest.raises(ValueError, match="MISSILE6"):
        _resolve_vehicle(None, "MISSILE6")


def test_run_scenario_freeflight_hbe_and_health(tmp_path: Path):
    path = _freeflight_short(tmp_path)
    result = run_scenario(path)
    row = next(r for r in result.plot_rows if r["time"] == pytest.approx(0.1))
    assert math.isfinite(row["hbe"])
    assert 6900.0 < row["hbe"] < 7100.0

    cfg = load_scenario(path)
    vehicle = _build_vehicle(path, cfg.vehicles[0])
    vehicle.define()
    for name, value in cfg.vehicles[0].params.items():
        vehicle.store.set(name, value)
    int_step = float(cfg.timing["int_step"])
    phases = {module.name: module.phases for module in cfg.modules}
    ctx = SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )
    for module in vehicle.modules:
        if "init" in phases.get(module.name, ()):
            module.initialize(vehicle, ctx)
    module_order = [module.name for module in cfg.modules if "exec" in module.phases]
    run_loop(
        [vehicle],
        {vehicle: vehicle.modules},
        module_order,
        0.1,
        int_step,
    )
    assert vehicle.health == 1


def test_committed_freeflight_case():
    cfg = load_scenario(CASES / "input_freeflight.jsonc")
    assert cfg.family == "agm6"
    assert cfg.end_time == 30
    vehicle = cfg.vehicles[0]
    assert vehicle.family == "agm6"
    assert vehicle.type == "MISSILE6"
    assert vehicle.name == "AGM6"
    assert vehicle.aero_deck == CASES / "AGM6_aero_deck.jsonc"
    assert vehicle.aero_deck.is_file()
    assert vehicle.prop_deck is None
    assert vehicle.weather_deck is None
    assert vehicle.params["alpha0x"] == 3
    assert vehicle.params["mprop"] == 1
    assert vehicle.params["sbel3"] == -7000
    assert vehicle.params["dvbe"] == 293
    assert "mair" not in vehicle.params


def test_agm6_missile6_requires_aero_deck(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "agm6.jsonc",
        [{"type": "MISSILE6", "name": "m", "params": {}, "events": []}],
        family="agm6",
    )
    with pytest.raises(ValueError, match=r"MISSILE6 requires aero_deck"):
        run_scenario(path)


def test_agm6_missile6_prop_deck_optional(tmp_path: Path):
    path = _freeflight_short(tmp_path)
    data = loads(path.read_text(encoding="utf-8"))
    data["vehicles"][0].pop("prop_deck", None)
    data["end_time"] = 0
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    cfg = load_scenario(path)
    assert cfg.vehicles[0].prop_deck is None
    vehicle = _build_vehicle(path, cfg.vehicles[0])
    from cadac.vehicles.agm6.vehicle import Agm6Missile

    assert isinstance(vehicle, Agm6Missile)


def test_agm6_target3_aircraft3_no_decks():
    from cadac.vehicles.agm6.aircraft import Agm6Aircraft
    from cadac.vehicles.agm6.target import Agm6Target

    dummy = Path("scenario.jsonc")
    target = _build_vehicle(
        dummy,
        VehicleSpec(
            type="TARGET3",
            name="t1",
            aero_deck=None,
            prop_deck=None,
            params={},
            events=[],
            family="agm6",
        ),
    )
    aircraft = _build_vehicle(
        dummy,
        VehicleSpec(
            type="AIRCRAFT3",
            name="a1",
            aero_deck=None,
            prop_deck=None,
            params={},
            events=[],
            family="agm6",
        ),
    )
    assert isinstance(target, Agm6Target)
    assert isinstance(aircraft, Agm6Aircraft)


def test_hyper5_target3_without_family_still_constructs():
    vehicle = _build_vehicle(
        Path("scenario.jsonc"),
        VehicleSpec(
            type="TARGET3",
            name="Truck_t1",
            aero_deck=None,
            prop_deck=None,
            params={},
            events=[],
            family=None,
        ),
    )
    assert isinstance(vehicle, Target3)


def test_unknown_vehicle_type_aim5_still_raises(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "unknown.jsonc",
        [{"type": "AIM5", "name": "p", "params": {}, "events": []}],
    )
    with pytest.raises(ValueError, match="AIM5"):
        run_scenario(path)


def test_no_family_missile6_run_scenario_raises(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "missile6.jsonc",
        [{"type": "MISSILE6", "name": "m", "params": {}, "events": []}],
    )
    with pytest.raises(ValueError, match="MISSILE6"):
        run_scenario(path)
