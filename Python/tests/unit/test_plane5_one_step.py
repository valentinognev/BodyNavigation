import json
import math
from pathlib import Path

import pytest

from cadac import run_scenario
from cadac.io.jsonc import loads
from cadac.io.scenario import load_scenario
from cadac.io.translate import deck_asc_to_jsonc, translate_scenario_asc

ROOT = Path(__file__).resolve().parents[3]
FALCON5 = ROOT / "CADAC_Simulations/FALCON5_250116/FALCON5"
CASES = Path(__file__).resolve().parents[2] / "cases" / "falcon5"
ALT0 = 3500.0
ALTCOM = 3000.0


def _turning_one_second(tmp_path: Path) -> Path:
    translate_scenario_asc(FALCON5 / "input_turning_to_IP.asc", tmp_path)
    deck_asc_to_jsonc(FALCON5 / "Falcon5_aero_deck.asc", tmp_path / "Falcon5_aero_deck.jsonc")
    deck_asc_to_jsonc(FALCON5 / "Falcon5_prop_deck.asc", tmp_path / "Falcon5_prop_deck.jsonc")
    path = tmp_path / "input_turning_to_IP.jsonc"
    data = loads(path.read_text(encoding="utf-8"))
    data["end_time"] = 1.0
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def test_plane5_type_health_and_module_order():
    from cadac.vehicles.plane5.vehicle import Plane5

    vehicle = Plane5("FALCON5", None, None)
    assert vehicle.type == "PLANE"
    assert vehicle.health == 1
    assert [module.name for module in vehicle.modules] == [
        "environment",
        "kinematics",
        "aerodynamics",
        "propulsion",
        "guidance",
        "control",
        "forces",
        "newton",
        "intercept",
    ]


def test_turning_one_second_time_and_alt(tmp_path: Path):
    result = run_scenario(_turning_one_second(tmp_path))
    row = next(r for r in result.plot_rows if r["time"] >= 1.0)
    assert row["time"] == pytest.approx(1.0)
    assert math.isfinite(row["alt"])
    assert row["alt"] < ALT0
    assert row["alt"] > ALTCOM - 500.0


def test_plane_plot_rows_have_time_alt_not_hyper3_columns(tmp_path: Path):
    result = run_scenario(_turning_one_second(tmp_path))
    assert result.plot_rows
    row = result.plot_rows[0]
    assert "time" in row
    assert "alt" in row
    assert "FSPV1" in row
    assert "SBEL1" in row
    assert "lonx" not in row
    assert "latx" not in row
    assert "SBEG1" not in row
    assert "psivgx" not in row


def test_committed_turning_case_is_plane_160s():
    cfg = load_scenario(CASES / "input_turning_to_IP.jsonc")
    assert cfg.end_time == 160
    assert cfg.vehicles[0].type == "PLANE"
    assert cfg.vehicles[0].aero_deck == CASES / "Falcon5_aero_deck.jsonc"
    assert cfg.vehicles[0].prop_deck == CASES / "Falcon5_prop_deck.jsonc"
    assert cfg.vehicles[0].aero_deck.is_file()
    assert cfg.vehicles[0].prop_deck.is_file()


def test_plane_requires_aero_and_prop_deck(tmp_path: Path):
    path = tmp_path / "plane.jsonc"
    path.write_text(
        '{ "title": "t", "options": {}, "modules": [], '
        '"timing": { "int_step": 0.01, "plot_step": 0.2 }, "end_time": 0, '
        '"vehicles": [ { "type": "PLANE", "name": "p", "params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="aero_deck"):
        run_scenario(path)


def test_unknown_vehicle_type_still_raises(tmp_path: Path):
    path = tmp_path / "unknown.jsonc"
    path.write_text(
        '{ "title": "t", "options": {}, "modules": [], '
        '"timing": { "int_step": 0.01, "plot_step": 0.2 }, "end_time": 0, '
        '"vehicles": [ { "type": "HYPER6", "name": "p", "params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="HYPER6"):
        run_scenario(path)
