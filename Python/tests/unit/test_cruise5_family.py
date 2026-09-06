import json
from pathlib import Path

import pytest

from cadac import run_scenario
from cadac.cli import _plot_columns
from cadac.io.plot import PLOT_COLUMNS, flagged_plot_columns
from cadac.io.scenario import load_scenario
from cadac.io.translate import translate_scenario_asc
from cadac.kernel.state import Field, StateStore

ROOT = Path(__file__).resolve().parents[3]
HYPER3 = ROOT / "CADAC_Simulations/HYPER3_250114/HYPER3"
CRUISE5 = ROOT / "CADAC_Simulations/CRUISE5_250115/CRUISE5"


def test_load_scenario_family_defaults_none(tmp_path):
    src = tmp_path / "n.jsonc"
    src.write_text(
        '{ "title": "t", "options": {}, "modules": [], "timing": {"int_step": 0.05}, '
        '"end_time": 1, "vehicles": [ { "type": "CRUISE3", "name": "v", '
        '"params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    cfg = load_scenario(src)
    assert cfg.vehicles[0].family is None
    assert cfg.vehicles[0].type == "CRUISE3"


def test_load_scenario_reads_family_key(tmp_path):
    src = tmp_path / "f.jsonc"
    src.write_text(
        '{ "title": "t", "options": {}, "modules": [], "timing": {"int_step": 0.05}, '
        '"end_time": 1, "vehicles": [ { "family": "cruise5", "type": "CRUISE3", '
        '"name": "UAV", "params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    cfg = load_scenario(src)
    assert cfg.vehicles[0].family == "cruise5"
    assert cfg.vehicles[0].type == "CRUISE3"
    assert cfg.vehicles[0].name == "UAV"


def test_scenario_family_applies_when_vehicle_omits_it(tmp_path):
    src = tmp_path / "s.jsonc"
    src.write_text(
        '{ "title": "t", "family": "cruise5", "options": {}, "modules": [], '
        '"timing": {"int_step": 0.05}, "end_time": 1, '
        '"vehicles": [ { "type": "CRUISE3", "name": "UAV", '
        '"params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    cfg = load_scenario(src)
    assert cfg.vehicles[0].family == "cruise5"


def test_vehicle_family_overrides_scenario_family(tmp_path):
    src = tmp_path / "o.jsonc"
    src.write_text(
        '{ "title": "t", "family": "aim5", "options": {}, "modules": [], '
        '"timing": {"int_step": 0.05}, "end_time": 1, '
        '"vehicles": [ { "family": "cruise5", "type": "CRUISE3", "name": "UAV", '
        '"params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    cfg = load_scenario(src)
    assert cfg.vehicles[0].family == "cruise5"


def test_translate_without_family_omits_key(tmp_path):
    translate_scenario_asc(HYPER3 / "input_climb.asc", tmp_path)
    raw = json.loads((tmp_path / "input_climb.jsonc").read_text(encoding="utf-8"))
    assert "family" not in raw["vehicles"][0]
    cfg = load_scenario(tmp_path / "input_climb.jsonc")
    assert cfg.vehicles[0].family is None


def test_translate_family_cruise5_writes_on_all_vehicles(tmp_path):
    translate_scenario_asc(CRUISE5 / "input_1.asc", tmp_path, family="cruise5")
    raw = json.loads((tmp_path / "input_1.jsonc").read_text(encoding="utf-8"))
    assert len(raw["vehicles"]) == 3
    for vehicle in raw["vehicles"]:
        assert vehicle["family"] == "cruise5"
    types = [v["type"] for v in raw["vehicles"]]
    assert types == ["CRUISE3", "TARGET3", "SATELLITE3"]
    cfg = load_scenario(tmp_path / "input_1.jsonc")
    assert [v.family for v in cfg.vehicles] == ["cruise5", "cruise5", "cruise5"]
    assert cfg.vehicles[0].params["mprop"] == 4
    assert cfg.vehicles[0].params["mcontrol"] == 46
    assert cfg.vehicles[0].params["mguidance"] == 30
    assert cfg.end_time == 410


def test_family_set_does_not_fall_through_to_global_cruise3(tmp_path):
    path = tmp_path / "uav.jsonc"
    path.write_text(
        '{ "title": "t", "options": {}, "modules": [], "timing": {"int_step": 0.05}, '
        '"end_time": 0, "vehicles": [ { "family": "cruise5", "type": "CRUISE3", '
        '"name": "UAV", "aero_deck": "a.jsonc", "prop_deck": "p.jsonc", '
        '"params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    (tmp_path / "a.jsonc").write_text(
        '{ "title": "a", "tables": [] }\n', encoding="utf-8", newline="\n"
    )
    (tmp_path / "p.jsonc").write_text(
        '{ "title": "p", "tables": [] }\n', encoding="utf-8", newline="\n"
    )
    with pytest.raises(ValueError, match="CRUISE3") as excinfo:
        run_scenario(path)
    assert "cruise5" in str(excinfo.value)


def test_family_none_hyper5_target3_still_works(tmp_path):
    path = tmp_path / "t.jsonc"
    path.write_text(
        '{ "title": "t", "options": {}, "modules": [ '
        '{ "name": "environment", "phases": ["def", "init", "exec"] }, '
        '{ "name": "newton", "phases": ["def", "init", "exec"] }, '
        '{ "name": "forces", "phases": ["def", "exec"] }, '
        '{ "name": "intercept", "phases": ["def", "exec"] } ], '
        '"timing": {"int_step": 0.05}, "end_time": 0, '
        '"vehicles": [ { "type": "TARGET3", "name": "Truck_t1", '
        '"params": { "lonx": 0, "latx": 0, "alt": 100, "dvbe": 1 }, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    result = run_scenario(path)
    assert result.plot_rows is not None


def test_plot_columns_non_cruise3_class_with_type_cruise3_uses_flagged():
    class Other:
        type = "CRUISE3"

        def __init__(self):
            self.store = StateStore()
            self.store.define(
                Field("alt", 1.0, "real", "out", "newton", ("plot",))
            )

    other = Other()
    cols = _plot_columns(other)
    assert cols == flagged_plot_columns(other.store)
    assert cols != list(PLOT_COLUMNS)
