import json
from pathlib import Path

import pytest

from cadac.cli import _build_vehicle, run_scenario
from cadac.io.scenario import load_scenario
from cadac.vehicles.round3.hyper5.target import Target3


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _minimal(type_name: str, *, scenario_family=None, vehicle_family=None) -> dict:
    vehicle = {"type": type_name, "name": "v", "params": {}, "events": []}
    if vehicle_family is not None:
        vehicle["family"] = vehicle_family
    body = {
        "title": "family",
        "options": {},
        "modules": [{"name": "environment", "phases": ["def", "exec"]}],
        "timing": {"int_step": 0.1},
        "end_time": 0.0,
        "vehicles": [vehicle],
    }
    if scenario_family is not None:
        body["family"] = scenario_family
    return body


def test_omitted_family_target3_is_hyper5(tmp_path: Path):
    path = _write(tmp_path / "g.jsonc", _minimal("TARGET3"))
    cfg = load_scenario(path)
    assert cfg.vehicles[0].family is None
    vehicle = _build_vehicle(path, cfg.vehicles[0])
    assert type(vehicle) is Target3


def test_scenario_family_applies_when_vehicle_omits_it(tmp_path: Path):
    path = _write(tmp_path / "s.jsonc", _minimal("TARGET3", scenario_family="sraam6"))
    assert load_scenario(path).vehicles[0].family == "sraam6"


def test_vehicle_family_overrides_scenario_family(tmp_path: Path):
    path = _write(
        tmp_path / "s.jsonc",
        _minimal("TARGET3", scenario_family="sraam6", vehicle_family="hyper5"),
    )
    assert load_scenario(path).vehicles[0].family == "hyper5"


def test_family_sraam6_target3_constructs_sraam6_target(tmp_path: Path):
    from cadac.vehicles.flat6.sraam6.target import Sraam6Target

    path = _write(tmp_path / "s.jsonc", _minimal("TARGET3", vehicle_family="sraam6"))
    cfg = load_scenario(path)
    assert cfg.vehicles[0].family == "sraam6"
    vehicle = _build_vehicle(path, cfg.vehicles[0])
    assert type(vehicle) is Sraam6Target


def test_family_sraam6_does_not_fall_through_to_global_target3(tmp_path: Path):
    path = _write(tmp_path / "s.jsonc", _minimal("CRUISE3", vehicle_family="sraam6"))
    cfg = load_scenario(path)
    assert cfg.vehicles[0].family == "sraam6"
    with pytest.raises(ValueError, match="CRUISE3"):
        run_scenario(path)
