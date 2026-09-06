import json
import math
from pathlib import Path

import pytest

from cadac import run_scenario
from cadac.io.asc_deck import parse_asc_deck
from cadac.io.jsonc import loads
from cadac.io.scenario import load_scenario
from cadac.io.translate import deck_asc_to_jsonc, translate_scenario_asc
from cadac.kernel.executive import SimContext, run_loop
from cadac.tables.lookup import Datadeck
from cadac.vehicles.hyper5.target import Target3

ROOT = Path(__file__).resolve().parents[3]
HYPER5_ASC = ROOT / "CADAC_Simulations/HYPER5_250113/HYPER5"
CASES = Path(__file__).resolve().parents[2] / "cases" / "hyper5"

MODULE_ORDER = [
    "environment",
    "aerodynamics",
    "propulsion",
    "forces",
    "newton",
    "seeker",
    "guidance",
    "control",
    "intercept",
    "targeting",
]
EXEC_ORDER = MODULE_ORDER[:-1]
INIT_NAMES = {"environment", "propulsion", "newton"}

HYPER_PARAMS = {
    "lonx": -106.28,
    "latx": 33.35,
    "alt": 2400.0,
    "psivgx": 0.0,
    "thtvgx": -10.5,
    "dvbe": 254.0,
    "alphax": -1.5,
    "phimvx": 0.0,
    "area": 11.6986,
    "alpposlimx": 6.0,
    "alpneglimx": -4.0,
    "mprop": 0,
    "mass0": 1352.0,
    "mcontrol": 44,
    "gcp": 2.0,
    "allimx": 1.0,
    "philimx": 70.0,
    "tphi": 1.0,
    "anposlimx": 2.0,
    "anneglimx": -2.0,
    "gacp": 10.0,
    "ta": 0.8,
    "mseeker": 1,
    "acq_range": 6000.0,
    "mguidance": 66,
    "pronav_gain": 3.5,
    "bias": 5.0,
}
TARGET_PARAMS = {
    "lonx": -106.28,
    "latx": 33.4,
    "alt": 1200.0,
}


def _aero_deck():
    _, tables = parse_asc_deck(HYPER5_ASC / "hyper5_aero_deck.asc")
    return Datadeck.from_tables(tables)


def _prepare(vehicle, params, int_step=0.05):
    vehicle.define()
    for name, value in params.items():
        vehicle.store.set(name, value)
    ctx = SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )
    for module in vehicle.modules:
        if module.name in INIT_NAMES:
            module.initialize(vehicle, ctx)


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


def _demo_short(tmp_path: Path) -> Path:
    translate_scenario_asc(HYPER5_ASC / "input.asc", tmp_path)
    deck_asc_to_jsonc(
        HYPER5_ASC / "hyper5_aero_deck.asc", tmp_path / "hyper5_aero_deck.jsonc"
    )
    path = tmp_path / "input.jsonc"
    data = loads(path.read_text(encoding="utf-8"))
    data["end_time"] = 0.05
    data["timing"]["plot_step"] = 0.05
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def test_hyper5_type_health_and_module_order():
    from cadac.vehicles.hyper5.vehicle import Hyper5

    vehicle = Hyper5("RR3X", None, None)
    assert vehicle.type == "HYPER5"
    assert vehicle.health == 1
    assert [module.name for module in vehicle.modules] == MODULE_ORDER


def test_hyper5_define_registers_targeting_when_omitted_from_modules():
    from cadac.vehicles.hyper5.vehicle import Hyper5

    vehicle = Hyper5("RR3X", None, None)
    vehicle.define()
    assert "mtargeting" in vehicle.store.names()


def test_hyper5_target3_run_loop_one_step():
    from cadac.vehicles.hyper5.vehicle import Hyper5

    hyper = Hyper5("RR3X", _aero_deck(), None)
    target = Target3("Truck_t1")
    _prepare(hyper, HYPER_PARAMS)
    _prepare(target, TARGET_PARAMS)
    run_loop(
        vehicles=[hyper, target],
        modules_by_vehicle={hyper: hyper.modules, target: target.modules},
        module_order=EXEC_ORDER,
        end_time=0.05,
        int_step=0.05,
    )
    assert math.isfinite(hyper.store.get("alt"))
    assert math.isfinite(target.store.get("alt"))
    assert hyper.health == 1
    assert target.health == 1


def test_run_scenario_one_step_alt_finite(tmp_path: Path):
    result = run_scenario(_demo_short(tmp_path))
    assert result.plot_rows
    row = result.plot_rows[0]
    assert math.isfinite(row["alt"])
    assert "time" in row
    assert "throttle" not in row
    assert "SBEL1" not in row


def test_committed_demo_47_case():
    cfg = load_scenario(CASES / "input.jsonc")
    assert cfg.end_time == 25
    hyper, target = cfg.vehicles
    assert hyper.type == "HYPER5"
    assert hyper.name == "RR3X"
    assert hyper.aero_deck == CASES / "hyper5_aero_deck.jsonc"
    assert hyper.aero_deck.is_file()
    assert hyper.prop_deck is None
    assert hyper.params["mprop"] == 0
    assert hyper.params["mcontrol"] == 44
    assert hyper.params["mguidance"] == 66
    assert hyper.params["mseeker"] == 1
    assert hyper.params["acq_range"] == 6000
    assert target.type == "TARGET3"
    assert target.name == "Truck_t1"
    assert target.aero_deck is None
    assert target.prop_deck is None


def test_hyper5_requires_aero_deck(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "hyper5.jsonc",
        [{"type": "HYPER5", "name": "h", "params": {"mprop": 0}, "events": []}],
    )
    with pytest.raises(ValueError, match=r"HYPER5 requires aero_deck"):
        run_scenario(path)


def test_hyper5_requires_prop_deck_when_mprop_nonzero(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "hyper5_prop.jsonc",
        [
            {
                "type": "HYPER5",
                "name": "h",
                "aero_deck": "missing.jsonc",
                "params": {"mprop": 1},
                "events": [],
            }
        ],
    )
    with pytest.raises(ValueError, match=r"HYPER5 requires prop_deck"):
        run_scenario(path)


def test_target3_run_scenario_without_decks(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "target3.jsonc",
        [{"type": "TARGET3", "name": "Truck_t1", "params": {}, "events": []}],
    )
    result = run_scenario(path)
    assert result.plot_rows is not None


def test_satellite3_run_scenario_without_decks(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "satellite3.jsonc",
        [{"type": "SATELLITE3", "name": "s1", "params": {}, "events": []}],
    )
    result = run_scenario(path)
    assert result.plot_rows is not None


def test_unknown_vehicle_type_still_aim5(tmp_path: Path):
    path = _jsonc(
        tmp_path,
        "unknown.jsonc",
        [{"type": "NO_SUCH_TYPE", "name": "p", "params": {}, "events": []}],
    )
    with pytest.raises(ValueError, match="NO_SUCH_TYPE"):
        run_scenario(path)
