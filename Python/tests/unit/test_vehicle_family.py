from pathlib import Path

import pytest

from cadac import run_scenario
from cadac.io.scenario import load_scenario
from cadac.io.translate import translate_scenario_asc

ROOT = Path(__file__).resolve().parents[3]
HYPER3 = ROOT / "CADAC_Simulations/HYPER3_250114/HYPER3"


def _jsonc(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


def test_vehicle_family_none_when_omitted(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "none.jsonc",
        '{ "title": "t", "options": {}, "modules": [], '
        '"timing": { "int_step": 0.01 }, "end_time": 0, '
        '"vehicles": [ { "type": "CRUISE3", "name": "c", "params": {}, "events": [] } ] }',
    )
    cfg = load_scenario(path)
    assert cfg.vehicles[0].family is None
    assert cfg.vehicles[0].type == "CRUISE3"


def test_vehicle_family_from_vehicle_key(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "veh.jsonc",
        '{ "title": "t", "options": {}, "modules": [], '
        '"timing": { "int_step": 0.01 }, "end_time": 0, '
        '"vehicles": [ { "family": "aim5", "type": "CRUISE3", "name": "c", '
        '"params": {}, "events": [] } ] }',
    )
    assert load_scenario(path).vehicles[0].family == "aim5"


def test_scenario_family_applies_when_vehicle_omits_it(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "scen.jsonc",
        '{ "title": "t", "family": "aim5", "options": {}, "modules": [], '
        '"timing": { "int_step": 0.01 }, "end_time": 0, '
        '"vehicles": [ { "type": "CRUISE3", "name": "c", "params": {}, "events": [] } ] }',
    )
    assert load_scenario(path).vehicles[0].family == "aim5"


def test_vehicle_family_overrides_scenario_family(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "ov.jsonc",
        '{ "title": "t", "family": "aim5", "options": {}, "modules": [], '
        '"timing": { "int_step": 0.01 }, "end_time": 0, '
        '"vehicles": [ { "family": "hyper5", "type": "CRUISE3", "name": "c", '
        '"params": {}, "events": [] } ] }',
    )
    assert load_scenario(path).vehicles[0].family == "hyper5"


def test_family_set_does_not_fall_through_to_global_types(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "nofall.jsonc",
        '{ "title": "t", "options": {}, "modules": [], '
        '"timing": { "int_step": 0.01 }, "end_time": 0, '
        '"vehicles": [ { "family": "aim5", "type": "CRUISE3", "name": "c", '
        '"params": {}, "events": [] } ] }',
    )
    with pytest.raises(ValueError, match=r"unknown vehicle type 'CRUISE3' for family 'aim5'"):
        run_scenario(path)


def test_family_none_unknown_no_such_type_raises(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "unknown.jsonc",
        '{ "title": "t", "options": {}, "modules": [], '
        '"timing": { "int_step": 0.01 }, "end_time": 0, '
        '"vehicles": [ { "type": "NO_SUCH_TYPE", "name": "r", "params": {}, "events": [] } ] }',
    )
    with pytest.raises(ValueError, match="NO_SUCH_TYPE"):
        run_scenario(path)


def test_translate_family_none_omits_family_key(tmp_path: Path):
    translate_scenario_asc(HYPER3 / "input_climb.asc", tmp_path)
    raw = (tmp_path / "input_climb.jsonc").read_text(encoding="utf-8")
    assert '"family"' not in raw


def test_translate_family_aim5_writes_on_each_vehicle(tmp_path: Path):
    src = tmp_path / "two.asc"
    src.write_text(
        "TITLE two\nOPTIONS y_scrn\nMODULES\n\tenvironment\tdef,exec\nEND\n"
        "TIMING\n\tint_step 0.01\nEND\nVEHICLES 2\n"
        "\tAIM5 Missile\n\t\tsael1  0\n\tEND\n"
        "\tAIRCRAFT3 Target\n\t\tsael1  1\n\tEND\n"
        "ENDTIME 10\nSTOP\n",
        encoding="utf-8",
        newline="\n",
    )
    translate_scenario_asc(src, tmp_path, family="aim5")
    cfg = load_scenario(tmp_path / "two.jsonc")
    assert [v.family for v in cfg.vehicles] == ["aim5", "aim5"]
    assert [v.type for v in cfg.vehicles] == ["AIM5", "AIRCRAFT3"]
