import json
import math
from pathlib import Path

import pytest

from cadac import run_scenario
from cadac.io.jsonc import loads
from cadac.io.scenario import load_scenario
from cadac.io.translate import deck_asc_to_jsonc, translate_scenario_asc
from cadac.kernel.state import Field

ROOT = Path(__file__).resolve().parents[3]
HYPER6_ASC = ROOT / "CADAC_Simulations/HYPER6_250125/HYPER6"
CASES = Path(__file__).resolve().parents[2] / "cases" / "hyper6"
ALT0 = 10000.0

MODULE_ORDER = [
    "kinematics",
    "environment",
    "aerodynamics",
    "propulsion",
    "ins",
    "guidance",
    "control",
    "actuator",
    "forces",
    "newton",
    "euler",
]


def _climb_tenth_second(tmp_path: Path) -> Path:
    translate_scenario_asc(HYPER6_ASC / "input_climb.asc", tmp_path)
    deck_asc_to_jsonc(
        HYPER6_ASC / "ghame6_aero_deck.asc", tmp_path / "ghame6_aero_deck.jsonc"
    )
    deck_asc_to_jsonc(
        HYPER6_ASC / "ghame6_prop_deck.asc", tmp_path / "ghame6_prop_deck.jsonc"
    )
    path = tmp_path / "input_climb.jsonc"
    data = loads(path.read_text(encoding="utf-8"))
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
    path = tmp_path / name
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8", newline="\n")
    return path


def test_hyper6_type_health_and_module_order():
    from cadac.vehicles.hyper6.vehicle import Hyper6

    vehicle = Hyper6("Hypersonic", None, None)
    assert vehicle.type == "HYPER6"
    assert vehicle.health == 1
    assert [module.name for module in vehicle.modules] == MODULE_ORDER


def test_hyper6_define_registers_guidance_when_omitted_from_modules():
    from cadac.vehicles.hyper6.vehicle import Hyper6

    vehicle = Hyper6("Hypersonic", None, None)
    vehicle.define()
    assert "mguide" in vehicle.store.names()


def test_hyper6_define_skips_existing_field():
    from cadac.vehicles.hyper6.vehicle import Hyper6

    vehicle = Hyper6("Hypersonic", None, None)
    vehicle.store.define(Field("alt", 42.0, "real", "out", "pre", ("plot",)))
    vehicle.define()
    assert vehicle.store.get("alt") == 42.0
    assert vehicle.store.field("alt").module == "pre"


def test_climb_tenth_second_time_and_alt(tmp_path: Path):
    result = run_scenario(_climb_tenth_second(tmp_path))
    row = next(r for r in result.plot_rows if r["time"] == pytest.approx(0.1))
    assert row["time"] == pytest.approx(0.1)
    assert math.isfinite(row["alt"])
    assert 9900.0 < row["alt"] < 10100.0
    for value in row.values():
        assert math.isfinite(value)


def test_hyper6_plot_rows_use_flagged_not_cruise3_columns(tmp_path: Path):
    result = run_scenario(_climb_tenth_second(tmp_path))
    assert result.plot_rows
    row = result.plot_rows[0]
    assert "time" in row
    assert "alt" in row
    assert "lonx" in row
    assert "latx" in row
    assert "psivgx" not in row
    assert "SBEG1" not in row
    assert "hbe" not in row
    assert "SBEL1" not in row
    assert "mach" not in row


def test_committed_climb_case_is_hyper6_60s():
    cfg = load_scenario(CASES / "input_climb.jsonc")
    assert cfg.end_time == 60
    assert cfg.vehicles[0].type == "HYPER6"
    assert cfg.vehicles[0].aero_deck == CASES / "ghame6_aero_deck.jsonc"
    assert cfg.vehicles[0].prop_deck == CASES / "ghame6_prop_deck.jsonc"
    assert cfg.vehicles[0].aero_deck.is_file()
    assert cfg.vehicles[0].prop_deck.is_file()
    assert cfg.vehicles[0].params["thtvdcomx"] == 0
    assert cfg.vehicles[0].events[0].when == {"time": {">": 10}}
    assert cfg.vehicles[0].events[0].set == {"thtvdcomx": 10}


def test_hyper6_requires_aero_and_prop_deck(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "hyper6.jsonc",
        [{"type": "HYPER6", "name": "h", "params": {}, "events": []}],
    )
    with pytest.raises(ValueError, match="aero_deck"):
        run_scenario(path)


def test_unknown_vehicle_type_aim5_still_raises(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "unknown.jsonc",
        [{"type": "AIM5", "name": "p", "params": {}, "events": []}],
    )
    with pytest.raises(ValueError, match="AIM5"):
        run_scenario(path)
