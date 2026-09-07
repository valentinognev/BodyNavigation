import json
from pathlib import Path

import pytest

from cadac import run_scenario
from cadac.io.scenario import load_scenario
from cadac.io.translate import translate_scenario_asc
from cadac.cli import _VEHICLE_FAMILIES, _VEHICLE_TYPES


def test_scenario_family_applies_when_vehicle_omits_it(tmp_path: Path):
    path = tmp_path / "scen.jsonc"
    path.write_text(
        '{ "title": "t", "family": "magsix", "options": {}, "modules": [], '
        '"timing": {"int_step": 0.01}, "end_time": 0, '
        '"vehicles": [ { "type": "CRUISE3", "name": "v", '
        '"params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    assert load_scenario(path).vehicles[0].family == "magsix"


def test_vehicle_family_overrides_scenario_family(tmp_path: Path):
    path = tmp_path / "ov.jsonc"
    path.write_text(
        '{ "title": "t", "family": "aim5", "options": {}, "modules": [], '
        '"timing": {"int_step": 0.01}, "end_time": 0, '
        '"vehicles": [ { "family": "magsix", "type": "CRUISE3", "name": "v", '
        '"params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    assert load_scenario(path).vehicles[0].family == "magsix"


def test_translate_family_magsix_stamps_vehicles(tmp_path: Path):
    src = tmp_path / "in.asc"
    src.write_text(
        "TITLE t\nOPTIONS n_scrn\nMODULES\n\tenvironment\tdef,exec\nEND\n"
        "TIMING\n\tint_step 0.01\nEND\nVEHICLES 1\n\tROTOR RECT.MR1\n"
        "\thbe 1000\n\tEND\nEND\nENDTIME 0.35\nSTOP\n",
        encoding="utf-8",
        newline="\n",
    )
    translate_scenario_asc(src, tmp_path, family="magsix")
    data = json.loads((tmp_path / "in.jsonc").read_text(encoding="utf-8"))
    assert data["vehicles"][0]["family"] == "magsix"
    assert data["vehicles"][0]["type"] == "ROTOR"


def test_family_set_error_includes_type_and_family(tmp_path: Path):
    path = tmp_path / "nope.jsonc"
    path.write_text(
        '{ "title": "t", "options": {}, "modules": [], '
        '"timing": {"int_step": 0.01}, "end_time": 0, '
        '"vehicles": [ { "type": "CRUISE3", "name": "v", '
        '"family": "magsix", "params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    assert "CRUISE3" in _VEHICLE_TYPES
    assert ("magsix", "CRUISE3") not in _VEHICLE_FAMILIES
    with pytest.raises(ValueError, match=r"CRUISE3.*magsix|magsix.*CRUISE3"):
        run_scenario(path)
