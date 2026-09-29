import json
import math
import shutil
from pathlib import Path

import pytest

from cadac import run_scenario
from cadac.cli import _VEHICLE_FAMILIES, _VEHICLE_TYPES, _build_vehicle
from cadac.io.jsonc import loads
from cadac.io.scenario import VehicleSpec, load_scenario
from cadac.kernel.executive import SimContext, run_loop
from cadac.vehicles.round3.hyper3.vehicle import Cruise3
from cadac.vehicles.round3.cruise5.satellite import Cruise5Satellite
from cadac.vehicles.round3.cruise5.target import Cruise5Target
from cadac.vehicles.round3.cruise5.vehicle import Cruise5
from cadac.vehicles.round3.hyper5.target import Target3

CASES = Path(__file__).resolve().parents[2] / "cases" / "cruise5"

WP_FLAG_WHEN = {"wp_flag": {"=": -1}}

MODULE_ORDER = [
    "environment",
    "aerodynamics",
    "propulsion",
    "forces",
    "newton",
    "gps",
    "targeting",
    "seeker",
    "ins",
    "guidance",
    "control",
    "intercept",
]


def _jsonc(tmp_path: Path, name: str, vehicles: list, **extra) -> Path:
    payload = {
        "title": "t",
        "options": extra.get("options", {}),
        "modules": extra.get("modules", []),
        "timing": extra.get("timing", {"int_step": 0.05, "plot_step": 0.05}),
        "end_time": extra.get("end_time", 0),
        "vehicles": vehicles,
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8", newline="\n")
    return path


def _short_input1(tmp_path: Path) -> Path:
    shutil.copy(CASES / "input_1.jsonc", tmp_path / "input_1.jsonc")
    shutil.copy(CASES / "cruise3_aero_deck.jsonc", tmp_path / "cruise3_aero_deck.jsonc")
    shutil.copy(CASES / "cruise3_prop_deck.jsonc", tmp_path / "cruise3_prop_deck.jsonc")
    path = tmp_path / "input_1.jsonc"
    data = loads(path.read_text(encoding="utf-8"))
    data["end_time"] = 0.05
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def test_family_pairs_do_not_overwrite_globals():
    assert _VEHICLE_TYPES["CRUISE3"] is Cruise3
    assert _VEHICLE_TYPES["TARGET3"] is Target3
    assert _VEHICLE_FAMILIES[("cruise5", "CRUISE3")] is Cruise5
    assert _VEHICLE_FAMILIES[("cruise5", "TARGET3")] is Cruise5Target
    assert _VEHICLE_FAMILIES[("cruise5", "SATELLITE3")] is Cruise5Satellite


def test_translated_input1_three_wp_flag_events():
    cfg = load_scenario(CASES / "input_1.jsonc")
    assert cfg.end_time == 410
    assert cfg.vehicles[0].family == "cruise5"
    assert cfg.vehicles[0].params["gfthm"] == 893620
    events = cfg.vehicles[0].events
    assert len(events) == 3
    assert events[0].when == WP_FLAG_WHEN
    assert events[0].set == {
        "wp_lonx": 15.25, "wp_latx": 35.54, "psifgx": 90, "altcom": 5000,
    }
    assert events[1].when == WP_FLAG_WHEN
    assert events[1].set == {
        "wp_lonx": 15.43, "wp_latx": 35.44, "psifgx": 180, "altcom": 2000,
    }
    assert events[2].when == WP_FLAG_WHEN
    assert events[2].set == {
        "mtargeting": 1, "del_radius": 5000, "mguidance": 43,
        "point_gain": 1, "line_gain": 1, "nl_gain_fact": 0.6,
        "decrement": 1000, "thtfgx": -50, "mcontrol": 44,
    }


def test_translated_input1_params_decks_and_family():
    cfg = load_scenario(CASES / "input_1.jsonc")
    uav, tank, sat = cfg.vehicles
    assert [v.family for v in cfg.vehicles] == ["cruise5", "cruise5", "cruise5"]
    assert uav.params["mprop"] == 4
    assert uav.params["mcontrol"] == 46
    assert uav.params["mguidance"] == 30
    assert uav.params["mseeker"] == 0
    assert uav.params["alt"] == 7000
    assert uav.params["lonx"] == 14.7
    assert tank.params["lonx"] == 15.4
    assert sat.params["alt"] == 500000
    assert tank.aero_deck is None
    assert tank.prop_deck is None
    assert sat.aero_deck is None
    assert sat.prop_deck is None
    assert uav.aero_deck == CASES / "cruise3_aero_deck.jsonc"
    assert uav.prop_deck == CASES / "cruise3_prop_deck.jsonc"
    assert uav.aero_deck.is_file()
    assert uav.prop_deck.is_file()


def test_cruise5_type_health_and_module_order():
    vehicle = Cruise5("UAV", None, None)
    assert vehicle.type == "CRUISE3"
    assert vehicle.health == 1
    assert [module.name for module in vehicle.modules] == MODULE_ORDER


def test_family_none_cruise3_builds_hyper3(tmp_path):
    aero = tmp_path / "a.jsonc"
    prop = tmp_path / "p.jsonc"
    aero.write_text('{ "title": "a", "tables": [] }\n', encoding="utf-8", newline="\n")
    prop.write_text('{ "title": "p", "tables": [] }\n', encoding="utf-8", newline="\n")
    spec = VehicleSpec(
        type="CRUISE3",
        name="v",
        aero_deck=aero,
        prop_deck=prop,
        params={},
        events=[],
        family=None,
    )
    vehicle = _build_vehicle(tmp_path / "x.jsonc", spec)
    assert type(vehicle) is Cruise3


def test_family_cruise5_plane_raises(tmp_path):
    path = _jsonc(
        tmp_path,
        "plane.jsonc",
        [
            {
                "family": "cruise5",
                "type": "PLANE",
                "name": "p",
                "params": {},
                "events": [],
            }
        ],
    )
    with pytest.raises(ValueError, match="PLANE") as excinfo:
        run_scenario(path)
    assert "cruise5" in str(excinfo.value)


def test_smoke_one_step_alt_finite_and_health(tmp_path):
    path = _short_input1(tmp_path)
    result = run_scenario(path)
    assert result.plot_rows
    assert math.isfinite(result.plot_rows[0]["alt"])

    cfg = load_scenario(path)
    vehicles = []
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
    module_order = [module.name for module in cfg.modules if "exec" in module.phases]
    run_loop(
        vehicles,
        {vehicle: vehicle.modules for vehicle in vehicles},
        module_order,
        0.05,
        int_step,
    )
    assert type(vehicles[0]) is Cruise5
    assert type(vehicles[1]) is Cruise5Target
    assert type(vehicles[2]) is Cruise5Satellite
    assert math.isfinite(vehicles[0].store.get("alt"))
    assert all(vehicle.health == 1 for vehicle in vehicles)
