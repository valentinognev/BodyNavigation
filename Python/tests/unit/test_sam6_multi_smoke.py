import json
import math
from pathlib import Path

import numpy as np
import pytest

from cadac import run_scenario
from cadac.cli import _build_vehicle
from cadac.io.scenario import load_scenario
from cadac.io.translate import deck_asc_to_jsonc
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext, _publish, run_loop

ROOT = Path(__file__).resolve().parents[3]
SRBM_AERO_ASC = ROOT / "CADAC_Simulations/SAM6_250217/SAM6/SRBM_aero_deck.asc"

AIRCRAFT_MODULES = [
    {"name": "environment", "phases": ["def", "exec"]},
    {"name": "kinematics", "phases": ["def", "init", "exec"]},
    {"name": "guidance", "phases": ["def", "exec"]},
    {"name": "control", "phases": ["def", "exec"]},
    {"name": "forces", "phases": ["def", "exec"]},
    {"name": "newton", "phases": ["def", "init", "exec"]},
]
ROCKET_MODULES = [
    {"name": "environment", "phases": ["def", "exec"]},
    {"name": "kinematics", "phases": ["def", "init", "exec"]},
    {"name": "aerodynamics", "phases": ["def", "init", "exec"]},
    {"name": "propulsion", "phases": ["def", "exec"]},
    {"name": "sensor", "phases": ["def", "exec"]},
    {"name": "guidance", "phases": ["def", "exec"]},
    {"name": "control", "phases": ["def", "exec"]},
    {"name": "forces", "phases": ["def", "exec"]},
    {"name": "newton", "phases": ["def", "init", "exec"]},
    {"name": "intercept", "phases": ["def", "exec"]},
]
RADAR_AIRCRAFT_MODULES = [
    {"name": "environment", "phases": ["def", "exec"]},
    {"name": "kinematics", "phases": ["def", "init", "exec"]},
    {"name": "guidance", "phases": ["def", "exec"]},
    {"name": "control", "phases": ["def", "exec"]},
    {"name": "forces", "phases": ["def", "exec"]},
    {"name": "newton", "phases": ["def", "init", "exec"]},
    {"name": "sensor", "phases": ["def", "exec"]},
]

AIRCRAFT_RF1 = {
    "family": "sam6",
    "type": "AIRCRAFT3",
    "name": "AC1",
    "params": {
        "sael1": 0,
        "sael2": -30e3,
        "sael3": -10e3,
        "dvae": 250,
        "psivlx": 90,
        "acft_option": 0,
    },
    "events": [],
}
ROCKET_SRBM_PARAMS = {
    "sael1": 1000,
    "sael2": -250e3,
    "sael3": 0,
    "thtvlx": 85,
    "mprop": 1,
    "maut": 1,
    "psivlx": 90,
    "dvae": 10,
    "alpha_t0x": 5,
    "beta_t0x": 0,
    "alpmax": 40,
    "alt_endo": 30000,
    "ancomx_bias": 0.5,
}
RADAR_FLAT0 = {
    "family": "sam6",
    "type": "RADAR0",
    "name": "Radar",
    "params": {
        "mtrack": 2,
        "srel1": 0,
        "srel2": 0,
        "srel3": 0,
        "lethal_rng": 20e3,
        "track_step": 0.01,
    },
    "events": [],
}

TIMING = {"int_step": 0.001, "plot_step": 0.1}
END_TIME = 0.1
LETHAL_RNG = 20e3
RF1_RANGE = math.sqrt(30e3**2 + 10e3**2)


def _write_jsonc(tmp_path: Path, name: str, modules, vehicles) -> Path:
    payload = {
        "title": name,
        "options": {},
        "modules": modules,
        "timing": TIMING,
        "end_time": END_TIME,
        "vehicles": vehicles,
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def _run(path: Path):
    cfg = load_scenario(path)
    int_step = float(cfg.timing["int_step"])
    phases = {module.name: module.phases for module in cfg.modules}
    module_order = [module.name for module in cfg.modules if "exec" in module.phases]
    vehicles = []
    modules_by_vehicle = {}
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
        module_order,
        cfg.end_time,
        int_step,
    )
    return vehicles


def _packets(vehicles):
    combus = [
        Packet(name=vehicle.name, type=vehicle.type, status=vehicle.health, vars={})
        for vehicle in vehicles
    ]
    for slot, vehicle in enumerate(vehicles):
        _publish(combus, slot, vehicle)
    return combus


def _by_type(items, type_name):
    return next(item for item in items if item.type == type_name)


def test_aircraft3_rf1_one_tenth_s_alt_health(tmp_path: Path):
    path = _write_jsonc(tmp_path, "aircraft.jsonc", AIRCRAFT_MODULES, [AIRCRAFT_RF1])
    cfg = load_scenario(path)
    assert cfg.end_time == END_TIME
    spec = cfg.vehicles[0]
    assert spec.family == "sam6"
    assert spec.type == "AIRCRAFT3"
    assert spec.aero_deck is None
    assert spec.prop_deck is None
    assert spec.params["sael1"] == 0
    assert spec.params["sael2"] == -30e3
    assert spec.params["sael3"] == -10e3
    assert spec.params["dvae"] == 250
    assert spec.params["psivlx"] == 90
    assert spec.params["acft_option"] == 0

    run_scenario(path)
    vehicle = _by_type(_run(path), "AIRCRAFT3")
    assert vehicle.health == 1
    assert float(vehicle.store.get("alt")) == pytest.approx(10000.0, abs=50.0)


def test_rocket5_srbm_one_tenth_s_health(tmp_path: Path):
    deck_asc_to_jsonc(SRBM_AERO_ASC, tmp_path / "SRBM_aero_deck.jsonc")
    rocket = {
        "family": "sam6",
        "type": "ROCKET5",
        "name": "SRBM",
        "aero_deck": "SRBM_aero_deck.jsonc",
        "params": ROCKET_SRBM_PARAMS,
        "events": [],
    }
    path = _write_jsonc(tmp_path, "rocket.jsonc", ROCKET_MODULES, [rocket])
    cfg = load_scenario(path)
    spec = cfg.vehicles[0]
    assert spec.family == "sam6"
    assert spec.type == "ROCKET5"
    assert spec.aero_deck == tmp_path / "SRBM_aero_deck.jsonc"
    assert spec.aero_deck.is_file()
    assert spec.prop_deck is None
    assert spec.params["sael1"] == 1000
    assert spec.params["sael2"] == -250e3
    assert spec.params["sael3"] == 0
    assert spec.params["thtvlx"] == 85
    assert spec.params["mprop"] == 1
    assert spec.params["maut"] == 1

    run_scenario(path)
    vehicle = _by_type(_run(path), "ROCKET5")
    assert vehicle.health == 1
    assert float(vehicle.store.get("alt")) == pytest.approx(0.0, abs=50.0)


def test_radar0_mtrack2_far_aircraft_lnch_delay_stays_zero(tmp_path: Path):
    path = _write_jsonc(
        tmp_path,
        "radar_aircraft.jsonc",
        RADAR_AIRCRAFT_MODULES,
        [AIRCRAFT_RF1, RADAR_FLAT0],
    )
    cfg = load_scenario(path)
    assert cfg.end_time == END_TIME
    aircraft_spec = _by_type(cfg.vehicles, "AIRCRAFT3")
    radar_spec = _by_type(cfg.vehicles, "RADAR0")
    assert aircraft_spec.family == "sam6"
    assert radar_spec.family == "sam6"
    assert aircraft_spec.aero_deck is None
    assert aircraft_spec.prop_deck is None
    assert radar_spec.sam_deck is None
    assert radar_spec.srmb_deck is None
    assert radar_spec.params["mtrack"] == 2
    assert radar_spec.params["srel1"] == 0
    assert radar_spec.params["srel2"] == 0
    assert radar_spec.params["srel3"] == 0
    assert radar_spec.params["lethal_rng"] == LETHAL_RNG
    assert RF1_RANGE == pytest.approx(31.6e3, rel=0.01)
    assert RF1_RANGE > LETHAL_RNG

    run_scenario(path)
    vehicles = _run(path)
    packets = _packets(vehicles)
    aircraft = _by_type(vehicles, "AIRCRAFT3")
    radar = _by_type(vehicles, "RADAR0")
    radar_pkt = _by_type(packets, "RADAR0")
    aircraft_pkt = _by_type(packets, "AIRCRAFT3")
    assert aircraft.health == 1
    assert radar.health == 1
    assert aircraft_pkt.status == 1
    assert radar_pkt.status == 1
    srel = np.asarray(radar.store.get("SREL"), dtype=float)
    assert all(math.isfinite(float(comp)) for comp in srel)
    sael = np.asarray(aircraft.store.get("SAEL"), dtype=float)
    range_m = float(np.linalg.norm(sael - srel))
    assert range_m > LETHAL_RNG
    assert radar_pkt.vars["lnch_delay_m1"] == 0
    assert radar_pkt.vars["lnch_delay_m1"] != 9999
