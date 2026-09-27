import json
import math
from pathlib import Path

import pytest

from cadac import run_scenario
from cadac.cli import _build_vehicle
from cadac.io.asc_deck import parse_asc_deck
from cadac.io.jsonc import loads
from cadac.io.scenario import load_scenario
from cadac.io.translate import deck_asc_to_jsonc, translate_scenario_asc
from cadac.kernel.executive import SimContext, run_loop
from cadac.tables.lookup import Datadeck

ROOT = Path(__file__).resolve().parents[3]
AIM5_ASC = ROOT / "CADAC_Simulations/AIM5_250114/AIM5"
CASES = Path(__file__).resolve().parents[2] / "cases" / "aim5"

MODULE_ORDER = [
    "environment",
    "kinematics",
    "aerodynamics",
    "propulsion",
    "seeker",
    "guidance",
    "control",
    "forces",
    "newton",
    "intercept",
]
AIRCRAFT_MODULE_ORDER = [
    "environment",
    "kinematics",
    "guidance",
    "control",
    "forces",
    "newton",
]
INIT_NAMES = {"kinematics", "control", "newton"}


def _aero_deck():
    _, tables = parse_asc_deck(AIM5_ASC / "aim5_aero_deck.asc")
    return Datadeck.from_tables(tables)


def _prop_deck():
    _, tables = parse_asc_deck(AIM5_ASC / "aim5_prop_deck.asc")
    return Datadeck.from_tables(tables)


def _jsonc(tmp_path: Path, name: str, vehicles: list, **extra) -> Path:
    payload = {
        "title": "t",
        "options": extra.get("options", {}),
        "modules": extra.get("modules", []),
        "timing": extra.get("timing", {"int_step": 0.002, "plot_step": 0.02}),
        "end_time": extra.get("end_time", 0),
        "vehicles": vehicles,
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8", newline="\n")
    return path


def _translate_decks(dst: Path) -> None:
    deck_asc_to_jsonc(AIM5_ASC / "aim5_aero_deck.asc", dst / "aim5_aero_deck.jsonc")
    deck_asc_to_jsonc(AIM5_ASC / "aim5_prop_deck.asc", dst / "aim5_prop_deck.jsonc")


def _hori_short(tmp_path: Path) -> Path:
    translate_scenario_asc(AIM5_ASC / "input_hori.asc", tmp_path, family="aim5")
    _translate_decks(tmp_path)
    path = tmp_path / "input_hori.jsonc"
    data = loads(path.read_text(encoding="utf-8"))
    data["end_time"] = 0.01
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def _prepare_from_scenario(path: Path):
    cfg = load_scenario(path)
    int_step = float(cfg.timing["int_step"])
    phases = {module.name: module.phases for module in cfg.modules}
    vehicles = []
    for spec in cfg.vehicles:
        vehicle = _build_vehicle(path, spec)
        vehicle.define()
        for name, value in spec.params.items():
            vehicle.store.set(name, value)
        ctx = SimContext(
            sim_time=0.0,
            int_step=int_step,
            event_time=0.0,
            out_fact=0.0,
            combus=None,
            vehicle_slot=len(vehicles),
        )
        for module in vehicle.modules:
            if "init" in phases.get(module.name, ()):
                module.initialize(vehicle, ctx)
        vehicles.append(vehicle)
    return cfg, vehicles


def test_aim5_type_health_and_module_order():
    from cadac.vehicles.flat3.aim5.vehicle import Aim5

    vehicle = Aim5("Missile", None, None)
    assert vehicle.type == "AIM5"
    assert vehicle.health == 1
    assert [module.name for module in vehicle.modules] == MODULE_ORDER


def test_aircraft3_constructor_has_no_aero_deck_required():
    from cadac.vehicles.flat3.aim5.aircraft import Aim5Aircraft

    vehicle = Aim5Aircraft("Target")
    assert vehicle.type == "AIRCRAFT3"
    assert vehicle.health == 1
    assert not hasattr(vehicle, "aero_deck")
    assert [module.name for module in vehicle.modules] == AIRCRAFT_MODULE_ORDER


def test_sael2_applied_after_init_sbel_east():
    from cadac.vehicles.flat3.aim5.vehicle import Aim5

    vehicle = Aim5("Missile", _aero_deck(), _prop_deck())
    vehicle.define()
    vehicle.store.set("sael1", 0.0)
    vehicle.store.set("sael2", -9000.0)
    vehicle.store.set("sael3", -10000.0)
    vehicle.store.set("dvae", 269.0)
    ctx = SimContext(
        sim_time=0.0,
        int_step=0.002,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )
    for module in vehicle.modules:
        if module.name in INIT_NAMES:
            module.initialize(vehicle, ctx)
    assert vehicle.store.get("SBEL")[1] == pytest.approx(-9000)
    assert vehicle.store.get("dvbe") == pytest.approx(269)


def test_committed_hori_case():
    cfg = load_scenario(CASES / "input_hori.jsonc")
    assert cfg.end_time == 10
    missile, target = cfg.vehicles
    assert missile.family == "aim5"
    assert missile.type == "AIM5"
    assert missile.name == "Missile"
    assert missile.aero_deck == CASES / "aim5_aero_deck.jsonc"
    assert missile.prop_deck == CASES / "aim5_prop_deck.jsonc"
    assert missile.aero_deck.is_file()
    assert missile.prop_deck.is_file()
    assert missile.params["sael2"] == -9000
    assert missile.params["dvae"] == 269
    assert missile.params["mprop"] == 1
    assert missile.params["mseek"] == 1
    assert missile.params["mguid"] == 1
    assert missile.params["gnav"] == 4
    assert "tgt_num" not in missile.params
    assert target.family == "aim5"
    assert target.type == "AIRCRAFT3"
    assert target.name == "Target"
    assert target.aero_deck is None
    assert target.prop_deck is None
    assert target.params["acft_option"] == 0


def test_run_scenario_hori_smoke_alt_and_health(tmp_path: Path):
    path = _hori_short(tmp_path)
    result = run_scenario(path)
    assert result.plot_rows
    row = result.plot_rows[0]
    assert math.isfinite(row["alt"])
    assert 9900.0 < row["alt"] < 10100.0

    cfg, vehicles = _prepare_from_scenario(path)
    missile, target = vehicles
    assert missile.store.get("SBEL")[1] == pytest.approx(-9000)
    int_step = float(cfg.timing["int_step"])
    module_order = [
        module.name for module in cfg.modules if "exec" in module.phases
    ]
    run_loop(
        vehicles,
        {vehicle: vehicle.modules for vehicle in vehicles},
        module_order,
        0.01,
        int_step,
    )
    assert math.isfinite(missile.store.get("alt"))
    assert missile.health == 1
    assert target.health == 1


def test_family_none_aim5_with_decks_constructs(tmp_path: Path):
    _translate_decks(tmp_path)
    path = _jsonc(
        tmp_path,
        "aim5.jsonc",
        [
            {
                "type": "AIM5",
                "name": "Missile",
                "aero_deck": "aim5_aero_deck.jsonc",
                "prop_deck": "aim5_prop_deck.jsonc",
                "params": {},
                "events": [],
            }
        ],
    )
    spec = load_scenario(path).vehicles[0]
    assert spec.family is None
    vehicle = _build_vehicle(path, spec)
    assert vehicle.type == "AIM5"


def test_family_none_aircraft3_raises_unknown_type(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "aircraft3.jsonc",
        [{"type": "AIRCRAFT3", "name": "Target", "params": {}, "events": []}],
    )
    with pytest.raises(ValueError, match=r"unknown vehicle type 'AIRCRAFT3'"):
        run_scenario(path)


def test_unknown_vehicle_type_still_no_such_type(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "unknown.jsonc",
        [{"type": "NO_SUCH_TYPE", "name": "p", "params": {}, "events": []}],
    )
    with pytest.raises(ValueError, match="NO_SUCH_TYPE"):
        run_scenario(path)
