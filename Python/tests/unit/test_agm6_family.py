import json
from pathlib import Path

import pytest

from cadac.cli import _VEHICLE_TYPES, _resolve_vehicle, run_scenario
from cadac.io.scenario import load_scenario
from cadac.io.translate import translate_scenario_asc
from cadac.vehicles.plane6.vehicle import Plane6


def test_no_family_uses_global_types():
    assert _resolve_vehicle(None, "PLANE6") is Plane6


def test_family_set_does_not_fall_back_to_global():
    with pytest.raises(ValueError, match="PLANE6"):
        _resolve_vehicle("agm6", "PLANE6")


def test_no_family_missile6_still_unknown():
    with pytest.raises(ValueError, match="MISSILE6"):
        _resolve_vehicle(None, "MISSILE6")


def test_load_scenario_family_defaults_onto_vehicle(tmp_path):
    path = tmp_path / "s.jsonc"
    path.write_text(
        json.dumps(
            {
                "title": "t",
                "family": "agm6",
                "options": {},
                "modules": [],
                "timing": {"int_step": 0.1},
                "end_time": 0.0,
                "vehicles": [{"type": "PLANE6", "name": "p", "params": {}, "events": []}],
            }
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    cfg = load_scenario(path)
    assert cfg.family == "agm6"
    assert cfg.vehicles[0].family == "agm6"


def test_vehicle_family_wins_over_scenario(tmp_path):
    path = tmp_path / "s.jsonc"
    path.write_text(
        json.dumps(
            {
                "title": "t",
                "family": "agm6",
                "options": {},
                "modules": [],
                "timing": {"int_step": 0.1},
                "end_time": 0.0,
                "vehicles": [
                    {
                        "type": "PLANE6",
                        "name": "p",
                        "family": "other",
                        "params": {},
                        "events": [],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    cfg = load_scenario(path)
    assert cfg.family == "agm6"
    assert cfg.vehicles[0].family == "other"


def test_translate_writes_family(tmp_path):
    src = tmp_path / "input.asc"
    src.write_text(
        "TITLE t\nOPTIONS y_plot\nMODULES\nenvironment def,exec\nEND\n"
        "TIMING\nint_step 0.1\nEND\nVEHICLES 1\nPLANE6 p\nEND\nENDTIME 1\nSTOP\n",
        encoding="utf-8",
        newline="\n",
    )
    translate_scenario_asc(src, tmp_path, family="agm6")
    data = json.loads((tmp_path / "input.jsonc").read_text(encoding="utf-8"))
    assert data["family"] == "agm6"


def test_global_unknown_type_still_aim5(tmp_path):
    path = tmp_path / "s.jsonc"
    path.write_text(
        json.dumps(
            {
                "title": "t",
                "options": {},
                "modules": [],
                "timing": {"int_step": 0.1},
                "end_time": 0.0,
                "vehicles": [{"type": "AIM5", "name": "p", "params": {}, "events": []}],
            }
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="AIM5"):
        run_scenario(path)


def test_hyper5_target3_without_family_still_global():
    assert "TARGET3" in _VEHICLE_TYPES
