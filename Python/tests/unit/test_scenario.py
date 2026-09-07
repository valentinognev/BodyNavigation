from pathlib import Path

import pytest

from cadac.io.scenario import load_scenario
from cadac.kernel.events import EventSpec

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "minimal_cruise3.jsonc"


def test_load_minimal_cruise3_with_time_event():
    cfg = load_scenario(FIXTURE)
    assert cfg.title == "minimal CRUISE3"
    assert cfg.options["scrn"] is True
    assert cfg.options["events"] is False
    assert cfg.options["plot"] is False
    assert cfg.options["doc"] is False
    assert cfg.options["csv"] is False
    assert cfg.options["tabout"] is False
    assert cfg.options["merge"] is False
    assert cfg.options["comscrn"] is False
    assert cfg.options["traj"] is False
    assert cfg.end_time == 90.0
    assert cfg.timing["int_step"] == 0.01
    assert len(cfg.modules) == 1
    assert cfg.modules[0].name == "environment"
    assert cfg.modules[0].phases == ["def", "init", "exec"]
    assert len(cfg.vehicles) == 1
    v = cfg.vehicles[0]
    assert v.type == "CRUISE3"
    assert v.name == "HYPER3"
    assert v.aero_deck == FIXTURE.parent / "ghame3_aero_deck.jsonc"
    assert v.prop_deck == FIXTURE.parent / "ghame3_prop_deck.jsonc"
    assert v.params == {"lonx": -80.55}
    assert v.events == [EventSpec(when={"time": {">": 10}}, set={"mprop": 2})]


def test_unknown_option_key_raises(tmp_path: Path):
    p = tmp_path / "bad.jsonc"
    p.write_text(
        '{ "title": "t", "options": { "scrn": true, "nope": true }, '
        '"modules": [], "timing": {}, "end_time": 1, '
        '"vehicles": [ { "type": "CRUISE3", "name": "v", "params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="nope"):
        load_scenario(p)


def test_extra_params_keys_stay(tmp_path: Path):
    p = tmp_path / "extra.jsonc"
    p.write_text(
        '{ "title": "t", "options": {}, "modules": [], "timing": {}, "end_time": 1, '
        '"vehicles": [ { "type": "CRUISE3", "name": "v", '
        '"params": { "lonx": -80.55, "custom": 1 }, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    cfg = load_scenario(p)
    assert cfg.vehicles[0].params == {"lonx": -80.55, "custom": 1}
