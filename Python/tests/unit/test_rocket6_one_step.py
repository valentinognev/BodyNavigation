import json
import math
from pathlib import Path

import pytest

from cadac import run_scenario
from cadac.cli import _VEHICLE_TYPES, _build_vehicle
from cadac.io.jsonc import loads
from cadac.io.scenario import OPTION_KEYS, load_scenario
from cadac.io.translate import deck_asc_to_jsonc, translate_scenario_asc
from cadac.kernel.state import Field

ROOT = Path(__file__).resolve().parents[3]
ROCKET6_ASC = ROOT / "CADAC_Simulations/ROCKET6_250122/ROCKET6"
HYPER6_CASES = Path(__file__).resolve().parents[2] / "cases" / "hyper6"
CASES = Path(__file__).resolve().parents[2] / "cases" / "rocket6"

MODULE_ORDER = [
    "kinematics",
    "environment",
    "propulsion",
    "aerodynamics",
    "gps",
    "startrack",
    "ins",
    "guidance",
    "control",
    "rcs",
    "tvc",
    "forces",
    "newton",
    "euler",
    "intercept",
]


def _strip_unknown_options(data: dict) -> dict:
    options = data.get("options") or {}
    data["options"] = {key: value for key, value in options.items() if key in OPTION_KEYS}
    return data


def _insertion_tenth_second(tmp_path: Path) -> Path:
    translate_scenario_asc(ROCKET6_ASC / "input.asc", tmp_path, family="rocket6")
    deck_asc_to_jsonc(
        ROCKET6_ASC / "aero_deck_SLV.asc", tmp_path / "aero_deck_SLV.jsonc"
    )
    deck_asc_to_jsonc(
        ROCKET6_ASC / "weather_deck_Wallops.asc",
        tmp_path / "weather_deck_Wallops.jsonc",
    )
    path = tmp_path / "input.jsonc"
    data = _strip_unknown_options(loads(path.read_text(encoding="utf-8")))
    data["end_time"] = 0.1
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def _jsonc(tmp_path: Path, name: str, vehicles: list, **extra) -> Path:
    payload = {
        "title": "t",
        "options": extra.get("options", {}),
        "modules": extra.get("modules", []),
        "timing": extra.get("timing", {"int_step": 0.01, "plot_step": 0.1}),
        "end_time": extra.get("end_time", 0),
        "vehicles": vehicles,
    }
    if "family" in extra:
        payload["family"] = extra["family"]
    path = tmp_path / name
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8", newline="\n")
    return path


def test_rocket6_type_family_health_and_module_order():
    from cadac.vehicles.rocket6.vehicle import Rocket6

    vehicle = Rocket6("SLV", None)
    assert vehicle.type == "HYPER6"
    assert vehicle.family == "rocket6"
    assert vehicle.health == 1
    assert [module.name for module in vehicle.modules] == MODULE_ORDER
    env = next(module for module in vehicle.modules if module.name == "environment")
    assert env.weather_deck is None


def test_rocket6_environment_uses_weather_deck():
    from cadac.vehicles.rocket6.vehicle import Rocket6

    vehicle = Rocket6("SLV", None, weather_deck="weather")
    env = next(module for module in vehicle.modules if module.name == "environment")
    assert env.weather_deck == "weather"


def test_rocket6_define_skips_existing_field():
    from cadac.vehicles.rocket6.vehicle import Rocket6

    vehicle = Rocket6("SLV", None)
    vehicle.store.define(Field("alt", 42.0, "real", "out", "pre", ("plot",)))
    vehicle.define()
    assert vehicle.store.get("alt") == 42.0
    assert vehicle.store.field("alt").module == "pre"


def test_insertion_tenth_second_time_and_alt(tmp_path: Path):
    result = run_scenario(_insertion_tenth_second(tmp_path))
    row = next(r for r in result.plot_rows if r["time"] == pytest.approx(0.1))
    assert row["time"] == pytest.approx(0.1)
    assert math.isfinite(row["alt"])
    assert 50.0 < row["alt"] < 250.0
    for value in row.values():
        assert math.isfinite(value)


def test_family_jsonc_hyper6_runs_rocket6(tmp_path: Path):
    from cadac.vehicles.rocket6.vehicle import Rocket6

    path = _insertion_tenth_second(tmp_path)
    spec = load_scenario(path).vehicles[0]
    assert spec.family == "rocket6"
    assert spec.type == "HYPER6"
    assert spec.prop_deck is None
    vehicle = _build_vehicle(path, spec)
    assert isinstance(vehicle, Rocket6)
    assert vehicle.type == "HYPER6"
    assert vehicle.family == "rocket6"


def test_no_family_hyper6_still_hyper6():
    from cadac.vehicles.hyper6.vehicle import Hyper6

    assert _VEHICLE_TYPES["HYPER6"] is Hyper6
    path = HYPER6_CASES / "input_climb.jsonc"
    spec = load_scenario(path).vehicles[0]
    assert spec.family is None
    assert spec.type == "HYPER6"
    vehicle = _build_vehicle(path, spec)
    assert isinstance(vehicle, Hyper6)
    assert getattr(vehicle, "family", None) != "rocket6"


def test_unknown_vehicle_type_aim5_still_raises(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "unknown.jsonc",
        [{"type": "AIM5", "name": "p", "params": {}, "events": []}],
    )
    with pytest.raises(ValueError, match="AIM5"):
        run_scenario(path)


def test_committed_insertion_case():
    cfg = load_scenario(CASES / "input.jsonc")
    assert cfg.end_time == 190
    assert cfg.vehicles[0].type == "HYPER6"
    assert cfg.vehicles[0].family == "rocket6"
    assert cfg.vehicles[0].name == "SLV"
    assert cfg.vehicles[0].aero_deck == CASES / "aero_deck_SLV.jsonc"
    assert cfg.vehicles[0].weather_deck == CASES / "weather_deck_Wallops.jsonc"
    assert cfg.vehicles[0].prop_deck is None
    assert cfg.vehicles[0].aero_deck.is_file()
    assert cfg.vehicles[0].weather_deck.is_file()
    assert (CASES / "aero_deck_SLV.jsonc").is_file()
    assert (CASES / "weather_deck_Wallops.jsonc").is_file()
    assert cfg.vehicles[0].params["alt"] == 100
    assert cfg.vehicles[0].params["mair"] == 0


def test_rocket6_requires_aero_deck_not_prop(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "rocket6.jsonc",
        [{"type": "HYPER6", "name": "SLV", "params": {}, "events": []}],
        family="rocket6",
    )
    with pytest.raises(ValueError, match="aero_deck") as excinfo:
        run_scenario(path)
    assert "and prop_deck" not in str(excinfo.value)
