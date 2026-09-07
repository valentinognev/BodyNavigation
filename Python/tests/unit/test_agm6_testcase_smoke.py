import json
import math
import shutil
from pathlib import Path

import pytest

from cadac import run_scenario
from cadac.cli import _VEHICLE_TYPES, _build_vehicle
from cadac.io.jsonc import loads
from cadac.io.scenario import OPTION_KEYS, load_scenario
from cadac.io.translate import deck_asc_to_jsonc, translate_scenario_asc
from cadac.kernel.events import EventSpec
from cadac.kernel.executive import SimContext, run_loop
from cadac.vehicles.agm6.target import Agm6Target
from cadac.vehicles.hyper5.target import Target3

ROOT = Path(__file__).resolve().parents[3]
AGM6_ASC = ROOT / "CADAC_Simulations/AGM6_250217/AGM6"
CASES = Path(__file__).resolve().parents[2] / "cases" / "agm6"
TEST_CASE_ASC = AGM6_ASC / "input_2_1 AGM6 Test Case.asc"
CASE = CASES / "input_testcase.jsonc"


def _keep_known_options(data: dict) -> dict:
    data = dict(data)
    data["options"] = {
        key: value for key, value in (data.get("options") or {}).items() if key in OPTION_KEYS
    }
    return data


def _translate_testcase(dst: Path) -> Path:
    translate_scenario_asc(TEST_CASE_ASC, dst, family="agm6")
    aero_src = CASES / "AGM6_aero_deck.jsonc"
    if aero_src.is_file():
        shutil.copy(aero_src, dst / "AGM6_aero_deck.jsonc")
    else:
        deck_asc_to_jsonc(AGM6_ASC / "AGM6_aero_deck.asc", dst / "AGM6_aero_deck.jsonc")
    weather_src = CASES / "weather_deck.jsonc"
    if weather_src.is_file():
        shutil.copy(weather_src, dst / "weather_deck.jsonc")
    else:
        deck_asc_to_jsonc(AGM6_ASC / "weather_deck.asc", dst / "weather_deck.jsonc")
    src = dst / f"{TEST_CASE_ASC.stem}.jsonc"
    data = _keep_known_options(json.loads(src.read_text(encoding="utf-8")))
    src.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    return src


def _testcase_short(tmp_path: Path) -> Path:
    data = _keep_known_options(loads(CASE.read_text(encoding="utf-8")))
    data["end_time"] = 0.05
    path = tmp_path / "input_testcase.jsonc"
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    shutil.copy(CASES / "AGM6_aero_deck.jsonc", tmp_path / "AGM6_aero_deck.jsonc")
    shutil.copy(CASES / "weather_deck.jsonc", tmp_path / "weather_deck.jsonc")
    return path


def test_translate_testcase_three_vehicles_weather_events_stoch(tmp_path: Path):
    translate_scenario_asc(TEST_CASE_ASC, tmp_path, family="agm6")
    data = json.loads((tmp_path / f"{TEST_CASE_ASC.stem}.jsonc").read_text(encoding="utf-8"))
    assert data["family"] == "agm6"
    vehicles = data["vehicles"]
    assert [vehicle["type"] for vehicle in vehicles] == [
        "MISSILE6",
        "TARGET3",
        "AIRCRAFT3",
    ]
    missile = vehicles[0]
    assert missile["weather_deck"] == "weather_deck.jsonc"
    params = missile["params"]
    assert params["mair"] == 212
    assert params["biasal"] == 0
    assert params["randal"] == 0
    assert params["dvae"] == 5
    assert params["biast"] == 0
    assert params["randt"] == 0
    assert "GAUSS" not in params
    assert "MARKOV" not in params
    assert "RAYL" not in params
    events = missile["events"]
    assert events[0]["when"] == {"time": {">": 3}}
    assert events[0]["set"]["mnav"] == 3
    assert events[0]["set"]["mseek"] == 2
    assert events[1]["when"] == {"mseek": {"=": 4}}
    assert events[1]["set"]["mguid"] == 6

    path = _translate_testcase(tmp_path)
    cfg = load_scenario(path)
    assert [vehicle.type for vehicle in cfg.vehicles] == [
        "MISSILE6",
        "TARGET3",
        "AIRCRAFT3",
    ]
    spec = cfg.vehicles[0]
    assert spec.weather_deck == tmp_path / "weather_deck.jsonc"
    assert spec.weather_deck.is_file()
    assert spec.params["mair"] == 212
    assert spec.events[0] == EventSpec(
        when={"time": {">": 3}},
        set={
            "mnav": 3,
            "mprop": 1,
            "mseek": 2,
            "mguid": 30,
            "grav_bias": 1.5,
            "gnav": 3,
        },
    )
    assert spec.events[1] == EventSpec(
        when={"mseek": {"=": 4}},
        set={"mguid": 6, "gnav": 3},
    )


def test_committed_testcase_case():
    cfg = load_scenario(CASE)
    assert cfg.family == "agm6"
    assert [vehicle.type for vehicle in cfg.vehicles] == [
        "MISSILE6",
        "TARGET3",
        "AIRCRAFT3",
    ]
    missile, target, aircraft = cfg.vehicles
    assert missile.family == "agm6"
    assert missile.type == "MISSILE6"
    assert missile.weather_deck == CASES / "weather_deck.jsonc"
    assert missile.weather_deck.is_file()
    assert missile.aero_deck == CASES / "AGM6_aero_deck.jsonc"
    assert missile.aero_deck.is_file()
    assert missile.params["mair"] == 212
    assert missile.params["biasal"] == 0
    assert missile.params["randal"] == 2
    assert missile.params["dvae"] == 5
    assert "GAUSS" not in missile.params
    assert missile.events[0].when == {"time": {">": 3}}
    assert missile.events[1].when == {"mseek": {"=": 4}}
    assert target.type == "TARGET3"
    assert aircraft.type == "AIRCRAFT3"


def test_family_only_target3_is_agm6_not_hyper5():
    cfg = load_scenario(CASE)
    target = _build_vehicle(CASE, cfg.vehicles[1])
    assert isinstance(target, Agm6Target)
    assert not isinstance(target, Target3)
    assert _VEHICLE_TYPES["TARGET3"] is Target3


def test_run_scenario_testcase_three_vehicles_hbe_and_health(tmp_path: Path):
    path = _testcase_short(tmp_path)
    cfg = load_scenario(path)
    assert len(cfg.vehicles) == 3
    result = run_scenario(path)
    assert result.plot_rows
    row = result.plot_rows[0]
    assert math.isfinite(row["hbe"])

    vehicles = []
    modules_by_vehicle = {}
    phases = {module.name: module.phases for module in cfg.modules}
    int_step = float(cfg.timing["int_step"])
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
        modules_by_vehicle[vehicle] = vehicle.modules
    run_loop(
        vehicles,
        modules_by_vehicle,
        [module.name for module in cfg.modules if "exec" in module.phases],
        0.05,
        int_step,
    )
    assert all(vehicle.health == 1 for vehicle in vehicles)
    assert math.isfinite(vehicles[0].store.get("hbe"))
