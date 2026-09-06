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
FALCON6 = ROOT / "CADAC_Simulations/FALCON6_250201/FALCON6"
CASES = Path(__file__).resolve().parents[2] / "cases" / "falcon6"
HBE0 = 1000.0


def _gamma_tenth_second(tmp_path: Path) -> Path:
    translate_scenario_asc(FALCON6 / "input_gamma.asc", tmp_path)
    deck_asc_to_jsonc(FALCON6 / "f16_aero_deck.asc", tmp_path / "f16_aero_deck.jsonc")
    deck_asc_to_jsonc(FALCON6 / "f16_prop_deck.asc", tmp_path / "f16_prop_deck.jsonc")
    path = tmp_path / "input_gamma.jsonc"
    data = loads(path.read_text(encoding="utf-8"))
    data["end_time"] = 0.1
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def test_plane6_type_health_and_module_order():
    from cadac.vehicles.plane6.vehicle import Plane6

    vehicle = Plane6("F16", None, None)
    assert vehicle.type == "PLANE6"
    assert vehicle.health == 1
    assert [module.name for module in vehicle.modules] == [
        "environment",
        "kinematics",
        "aerodynamics",
        "propulsion",
        "guidance",
        "forces",
        "control",
        "actuator",
        "euler",
        "newton",
    ]


def test_plane6_define_registers_guidance_when_omitted_from_modules():
    from cadac.vehicles.plane6.vehicle import Plane6

    vehicle = Plane6("F16", None, None)
    vehicle.define()
    assert "mguid" in vehicle.store.names()


def test_plane6_define_skips_existing_field():
    from cadac.vehicles.plane6.vehicle import Plane6

    vehicle = Plane6("F16", None, None)
    vehicle.store.define(Field("hbe", 42.0, "real", "out", "pre", ("plot",)))
    vehicle.define()
    assert vehicle.store.get("hbe") == 42.0
    assert vehicle.store.field("hbe").module == "pre"


def test_gamma_tenth_second_time_and_hbe(tmp_path: Path):
    result = run_scenario(_gamma_tenth_second(tmp_path))
    row = next(r for r in result.plot_rows if r["time"] >= 0.1)
    assert row["time"] == pytest.approx(0.1)
    assert math.isfinite(row["hbe"])
    assert 900.0 < row["hbe"] < 1100.0
    for value in row.values():
        assert math.isfinite(value)


def test_plane6_plot_rows_use_flagged_not_cruise3_columns(tmp_path: Path):
    result = run_scenario(_gamma_tenth_second(tmp_path))
    assert result.plot_rows
    row = result.plot_rows[0]
    assert "time" in row
    assert "hbe" in row
    assert "SBEL1" in row
    assert "lonx" not in row
    assert "latx" not in row
    assert "SBEG1" not in row
    assert "psivgx" not in row
    assert "alt" not in row


def test_committed_gamma_case_is_plane6_20s():
    cfg = load_scenario(CASES / "input_gamma.jsonc")
    assert cfg.end_time == 20
    assert cfg.vehicles[0].type == "PLANE6"
    assert cfg.vehicles[0].aero_deck == CASES / "f16_aero_deck.jsonc"
    assert cfg.vehicles[0].prop_deck == CASES / "f16_prop_deck.jsonc"
    assert cfg.vehicles[0].aero_deck.is_file()
    assert cfg.vehicles[0].prop_deck.is_file()


def test_plane6_requires_aero_and_prop_deck(tmp_path: Path):
    path = tmp_path / "plane6.jsonc"
    path.write_text(
        '{ "title": "t", "options": {}, "modules": [], '
        '"timing": { "int_step": 0.01, "plot_step": 0.2 }, "end_time": 0, '
        '"vehicles": [ { "type": "PLANE6", "name": "p", "params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="aero_deck"):
        run_scenario(path)
