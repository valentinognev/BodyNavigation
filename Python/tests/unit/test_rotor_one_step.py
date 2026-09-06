import json
from pathlib import Path

import pytest

from cadac import run_scenario
from cadac.io.jsonc import loads
from cadac.io.scenario import load_scenario
from cadac.io.translate import translate_scenario_asc
from cadac.kernel.state import Field

ROOT = Path(__file__).resolve().parents[3]
MAGSIX_ASC = ROOT / "CADAC_Simulations/MAGSIX_231111/MAGSIX"
CASES = Path(__file__).resolve().parents[2] / "cases" / "magsix"
MODULE_ORDER = ["environment", "trajectory", "attitude"]


def _attitude_smoke(tmp_path: Path) -> Path:
    translate_scenario_asc(MAGSIX_ASC / "input.asc", tmp_path, family="magsix")
    path = tmp_path / "input.jsonc"
    data = loads(path.read_text(encoding="utf-8"))
    data["end_time"] = 0.01
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def test_rotor_type_health_and_module_order():
    from cadac.vehicles.rotor.vehicle import Rotor

    vehicle = Rotor("RECT.MR1")
    assert vehicle.type == "ROTOR"
    assert vehicle.health == 1
    assert [module.name for module in vehicle.modules] == MODULE_ORDER


def test_rotor_define_skips_existing_field():
    from cadac.vehicles.rotor.vehicle import Rotor

    vehicle = Rotor("RECT.MR1")
    vehicle.store.define(Field("hbe", 42.0, "real", "out", "pre", ("plot",)))
    vehicle.define()
    assert vehicle.store.get("hbe") == 42.0
    assert vehicle.store.field("hbe").module == "pre"


def test_attitude_smoke_hbe_and_phix(tmp_path: Path):
    result = run_scenario(_attitude_smoke(tmp_path))
    assert result.plot_rows
    row = result.plot_rows[-1]
    assert 990.0 < row["hbe"] < 1010.0
    assert abs(row["phix"]) < 180.0
    for value in row.values():
        assert value == value  # finite


def test_committed_attitude_case_is_rect_mr1():
    cfg = load_scenario(CASES / "input.jsonc")
    assert cfg.end_time == 0.35
    assert cfg.timing["int_step"] == 0.0001
    assert cfg.timing["plot_step"] == 0.005
    v = cfg.vehicles[0]
    assert v.type == "ROTOR"
    assert v.family == "magsix"
    assert v.name == "RECT.MR1"
    assert v.aero_deck is None
    assert v.prop_deck is None
    assert v.params["hbe"] == 1000
    assert v.params["dvbe"] == 16.6
    assert v.params["omega_rpm"] == 850
    assert v.params["thtvlx"] == -77
    assert v.params["nonlinear"] == 0
    assert v.params["phix"] == 3
    assert v.params["rrx"] == -40
    assert v.params["moi_spin"] == 0.004
    assert v.params["moi_trans"] == 0.0268


def test_rotor_does_not_require_decks(tmp_path: Path):
    path = tmp_path / "rotor.jsonc"
    path.write_text(
        '{ "title": "t", "options": {}, "modules": [], '
        '"timing": {"int_step": 0.0001}, "end_time": 0, '
        '"vehicles": [ { "type": "ROTOR", "name": "r", '
        '"params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    run_scenario(path)


def test_family_magsix_rotor_runs(tmp_path: Path):
    result = run_scenario(_attitude_smoke(tmp_path))
    assert result.plot_rows[0]["hbe"] == pytest.approx(1000.0, rel=1e-3)


def test_unknown_type_no_such_type_still_raises(tmp_path: Path):
    path = tmp_path / "unknown.jsonc"
    path.write_text(
        '{ "title": "t", "options": {}, "modules": [], '
        '"timing": {"int_step": 0.0001}, "end_time": 0, '
        '"vehicles": [ { "type": "NO_SUCH_TYPE", "name": "x", '
        '"params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="NO_SUCH_TYPE"):
        run_scenario(path)
