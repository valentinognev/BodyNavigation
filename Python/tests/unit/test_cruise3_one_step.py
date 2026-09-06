import json
from pathlib import Path

import pytest

from cadac import run_scenario
from cadac.io.jsonc import loads
from cadac.io.scenario import load_scenario
from cadac.io.translate import deck_asc_to_jsonc, translate_scenario_asc

ROOT = Path(__file__).resolve().parents[3]
HYPER3 = ROOT / "CADAC_Simulations/HYPER3_250114/HYPER3"
CASES = Path(__file__).resolve().parents[2] / "cases" / "hyper3"
MASS0 = 136077


def _climb_one_second(tmp_path: Path) -> Path:
    translate_scenario_asc(HYPER3 / "input_climb.asc", tmp_path)
    deck_asc_to_jsonc(HYPER3 / "ghame3_aero_deck.asc", tmp_path / "ghame3_aero_deck.jsonc")
    deck_asc_to_jsonc(HYPER3 / "ghame3_prop_deck.asc", tmp_path / "ghame3_prop_deck.jsonc")
    path = tmp_path / "input_climb.jsonc"
    data = loads(path.read_text(encoding="utf-8"))
    data["end_time"] = 1.0
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def test_climb_one_second_time_mass_alt(tmp_path: Path):
    result = run_scenario(_climb_one_second(tmp_path))
    row = next(r for r in result.plot_rows if r["time"] >= 1.0)
    assert row["time"] >= 1.0
    assert row["mass"] < MASS0
    assert row["alt"] > 2999


def test_plot_rows_include_plot_flagged_names(tmp_path: Path):
    result = run_scenario(_climb_one_second(tmp_path))
    assert result.plot_rows
    row = result.plot_rows[0]
    assert "time" in row
    assert "alt" in row
    assert "mass" in row
    assert "FSPV1" in row
    assert "SBEG1" in row
    assert "sbeg1" not in row


def test_plot_csv_written_when_plot_and_csv(tmp_path: Path):
    path = _climb_one_second(tmp_path)
    run_scenario(path)
    csv_path = path.parent / "plot.csv"
    raw = csv_path.read_bytes()
    assert b"\r" not in raw
    lines = raw.decode("utf-8").split("\n")
    columns = [col for col in lines[2].split(",") if col]
    assert columns == [
        "time",
        "FSPV1",
        "FSPV2",
        "FSPV3",
        "pdynmc",
        "mach",
        "lonx",
        "latx",
        "alt",
        "dvbe",
        "psivgx",
        "thtvgx",
        "SBEG1",
        "SBEG2",
        "SBEG3",
        "VBEG1",
        "VBEG2",
        "VBEG3",
        "throttle",
        "mass",
        "thrust",
        "fmassr",
        "cl_ov_cd",
    ]


def test_unknown_vehicle_type_raises(tmp_path: Path):
    path = tmp_path / "plane.jsonc"
    path.write_text(
        '{ "title": "t", "options": {}, "modules": [], '
        '"timing": { "int_step": 0.01, "plot_step": 0.2 }, "end_time": 0, '
        '"vehicles": [ { "type": "NO_SUCH_TYPE", "name": "p", "params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="NO_SUCH_TYPE"):
        run_scenario(path)


def _timing_scenario(tmp_path: Path, name: str, timing: str) -> Path:
    path = tmp_path / name
    path.write_text(
        '{ "title": "t", "options": {}, "modules": [], '
        f'"timing": {timing}, "end_time": 1, '
        '"vehicles": [ { "type": "CRUISE3", "name": "v", "params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_int_step_zero_raises(tmp_path: Path):
    path = _timing_scenario(tmp_path, "zero.jsonc", '{ "int_step": 0 }')
    with pytest.raises(ValueError, match=r": int_step"):
        run_scenario(path)


def test_int_step_negative_raises(tmp_path: Path):
    path = _timing_scenario(tmp_path, "neg.jsonc", '{ "int_step": -0.01 }')
    with pytest.raises(ValueError, match=r": int_step"):
        run_scenario(path)


def test_missing_int_step_raises(tmp_path: Path):
    path = _timing_scenario(tmp_path, "missing.jsonc", "{}")
    with pytest.raises(ValueError, match=r": int_step"):
        run_scenario(path)


def test_committed_climb_case_is_cruise3_90s():
    cfg = load_scenario(CASES / "input_climb.jsonc")
    assert cfg.end_time == 90
    assert cfg.vehicles[0].type == "CRUISE3"
    assert cfg.vehicles[0].aero_deck == CASES / "ghame3_aero_deck.jsonc"
    assert cfg.vehicles[0].prop_deck == CASES / "ghame3_prop_deck.jsonc"
    assert cfg.vehicles[0].aero_deck.is_file()
    assert cfg.vehicles[0].prop_deck.is_file()
