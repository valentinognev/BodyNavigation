import json
import math
from pathlib import Path

import pytest

from cadac import run_scenario
from cadac.io.jsonc import loads
from cadac.io.translate import deck_asc_to_jsonc, translate_scenario_asc
from cadac.vehicles.flat6.sraam6.target import Sraam6Target
from cadac.vehicles.flat6.sraam6.vehicle import Sraam6Missile

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "CADAC_Simulations/SRAAM6_250130/SRAAM6"
CASES = Path(__file__).resolve().parents[2] / "cases" / "sraam6"


def test_missile_module_order_includes_tvc():
    vehicle = Sraam6Missile("Missile", None, None)
    assert vehicle.type == "MISSILE6"
    names = [m.name for m in vehicle.modules]
    assert names[:4] == ["environment", "kinematics", "aerodynamics", "propulsion"]
    assert "tvc" in names
    assert "newton" in names


def test_target_constructor_has_no_decks():
    vehicle = Sraam6Target("Target aircraft")
    assert vehicle.type == "TARGET3"


def test_com_names_from_com_outputs():
    missile = Sraam6Missile("Missile", None, None)
    missile.define()
    for name in ("time", "vmach", "SBEL", "VBEL", "mseek"):
        assert name in missile.com_names
        assert "com" in missile.store.field(name).outputs
    target = Sraam6Target("Target aircraft")
    target.define()
    for name in ("dvae", "SAEL", "VAEL", "psial", "thtal"):
        assert name in target.com_names
        assert "com" in target.store.field(name).outputs


def test_1v1_tenth_second_hbe(tmp_path: Path):
    translate_scenario_asc(SRC / "input_1v1.asc", tmp_path, family="sraam6")
    deck_asc_to_jsonc(SRC / "sraam6_aero_deck.asc", tmp_path / "sraam6_aero_deck.jsonc")
    deck_asc_to_jsonc(SRC / "sraam6_prop_deck.asc", tmp_path / "sraam6_prop_deck.jsonc")
    path = tmp_path / "input_1v1.jsonc"
    data = loads(path.read_text(encoding="utf-8"))
    data["end_time"] = 0.1
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    result = run_scenario(path)
    row = next(r for r in result.plot_rows if r["time"] >= 0.1)
    assert math.isfinite(row["hbe"])
    assert 4500.0 < row["hbe"] < 5500.0


def test_1v1_tenth_second_target_sael_and_health(tmp_path: Path):
    from cadac.cli import _build_vehicle
    from cadac.io.scenario import load_scenario
    from cadac.kernel.executive import SimContext, run_loop

    translate_scenario_asc(SRC / "input_1v1.asc", tmp_path, family="sraam6")
    deck_asc_to_jsonc(SRC / "sraam6_aero_deck.asc", tmp_path / "sraam6_aero_deck.jsonc")
    deck_asc_to_jsonc(SRC / "sraam6_prop_deck.asc", tmp_path / "sraam6_prop_deck.jsonc")
    path = tmp_path / "input_1v1.jsonc"
    data = loads(path.read_text(encoding="utf-8"))
    data["end_time"] = 0.1
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    cfg = load_scenario(path)
    vehicles = []
    for spec in cfg.vehicles:
        vehicle = _build_vehicle(path, spec)
        vehicle.define()
        for name, value in spec.params.items():
            vehicle.store.set(name, value)
        ctx = SimContext(0.0, 0.001, 0.0, 0.0, None, len(vehicles))
        for module in vehicle.modules:
            module.initialize(vehicle, ctx)
        vehicles.append(vehicle)
    run_loop(
        vehicles,
        {v: v.modules for v in vehicles},
        [m["name"] for m in data["modules"]],
        0.1,
        0.001,
    )
    assert vehicles[0].health == 1
    assert vehicles[1].health == 1
    sael = vehicles[1].store.get("SAEL")
    assert all(math.isfinite(float(x)) for x in sael)


def test_family_sraam6_cruise3_does_not_fall_through(tmp_path: Path):
    path = tmp_path / "c.jsonc"
    path.write_text(
        json.dumps(
            {
                "title": "nf",
                "options": {},
                "modules": [],
                "timing": {"int_step": 0.1},
                "end_time": 0.0,
                "vehicles": [
                    {
                        "family": "sraam6",
                        "type": "CRUISE3",
                        "name": "c",
                        "params": {},
                        "events": [],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="CRUISE3"):
        run_scenario(path)


def test_family_sraam6_target3_tiny_jsonc_needs_no_decks(tmp_path: Path):
    path = tmp_path / "t.jsonc"
    path.write_text(
        json.dumps(
            {
                "title": "tgt",
                "options": {},
                "modules": [],
                "timing": {"int_step": 0.1},
                "end_time": 0.0,
                "vehicles": [
                    {
                        "family": "sraam6",
                        "type": "TARGET3",
                        "name": "t",
                        "params": {},
                        "events": [],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    run_scenario(path)


def test_family_sraam6_missile6_without_decks_raises(tmp_path: Path):
    path = tmp_path / "m.jsonc"
    path.write_text(
        json.dumps(
            {
                "title": "m",
                "options": {},
                "modules": [],
                "timing": {"int_step": 0.1},
                "end_time": 0.0,
                "vehicles": [
                    {
                        "family": "sraam6",
                        "type": "MISSILE6",
                        "name": "m",
                        "params": {},
                        "events": [],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="aero_deck"):
        run_scenario(path)


def test_translated_jsonc_vehicles_have_family_sraam6():
    data = loads((CASES / "input_1v1.jsonc").read_text(encoding="utf-8"))
    assert data["end_time"] == 12
    assert [v["family"] for v in data["vehicles"]] == ["sraam6", "sraam6"]
