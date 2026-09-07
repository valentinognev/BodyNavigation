import json
import math
import shutil
from pathlib import Path

import pytest

from cadac import run_scenario
from cadac.cli import _build_vehicle
from cadac.io.jsonc import loads
from cadac.io.scenario import load_scenario
from cadac.kernel.events import EventSpec
from cadac.kernel.executive import SimContext, run_loop

CASES = Path(__file__).resolve().parents[2] / "cases" / "sam6"
AUTOPILOT = CASES / "input_SAM_autopilot.jsonc"

EXPECTED_EVENTS = [
    EventSpec(
        when={"time": {">": 5}},
        set={"maut": 3, "wacl_bias": -0.3, "pacl_bias": 0, "zacl_bias": 0},
    ),
    EventSpec(when={"time": {">": 10}}, set={"ancomx_test": 1}),
    EventSpec(when={"time": {">": 13}}, set={"ancomx_test": 0}),
    EventSpec(when={"time": {">": 16}}, set={"ancomx_test": 1}),
    EventSpec(when={"time": {">": 19}}, set={"ancomx_test": 0}),
    EventSpec(when={"time": {">": 22}}, set={"ancomx_test": 1}),
    EventSpec(when={"time": {">": 25}}, set={"ancomx_test": 0}),
]


def _autopilot_short(tmp_path: Path) -> Path:
    for name in (
        "input_SAM_autopilot.jsonc",
        "SAM_aero_deck.jsonc",
        "SAM_prop_deck.jsonc",
    ):
        shutil.copy(CASES / name, tmp_path / name)
    path = tmp_path / "input_SAM_autopilot.jsonc"
    data = loads(path.read_text(encoding="utf-8"))
    data["end_time"] = 0.05
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def _run_loop_short(path: Path):
    cfg = load_scenario(path)
    spec = cfg.vehicles[0]
    vehicle = _build_vehicle(path, spec)
    vehicle.define()
    for name, value in spec.params.items():
        vehicle.store.set(name, value)
    int_step = float(cfg.timing["int_step"])
    phases = {module.name: module.phases for module in cfg.modules}
    ctx = SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )
    for module in vehicle.modules:
        if "init" in phases.get(module.name, ()):
            module.initialize(vehicle, ctx)
    module_order = [module.name for module in cfg.modules if "exec" in module.phases]
    run_loop(
        [vehicle],
        {vehicle: vehicle.modules},
        module_order,
        0.05,
        int_step,
    )
    return vehicle


def test_committed_autopilot_case_is_sam6_missile6_30s():
    data = loads(AUTOPILOT.read_text(encoding="utf-8"))
    assert data["end_time"] == 30
    assert all(vehicle["family"] == "sam6" for vehicle in data["vehicles"])
    assert all(vehicle["type"] != "RADAR0" for vehicle in data["vehicles"])

    cfg = load_scenario(AUTOPILOT)
    assert cfg.end_time == 30
    assert len(cfg.vehicles) == 1
    vehicle = cfg.vehicles[0]
    assert vehicle.family == "sam6"
    assert vehicle.type == "MISSILE6"
    assert vehicle.name == "SAM"
    assert vehicle.params["mact"] == 2
    assert vehicle.params["mins"] == 1
    assert vehicle.params["maut"] == 2
    assert vehicle.aero_deck == CASES / "SAM_aero_deck.jsonc"
    assert vehicle.prop_deck == CASES / "SAM_prop_deck.jsonc"
    assert vehicle.aero_deck.is_file()
    assert vehicle.prop_deck.is_file()
    assert vehicle.sam_deck is None
    assert vehicle.srmb_deck is None
    assert vehicle.events == EXPECTED_EVENTS


def test_autopilot_one_step_hbe_alt_health(tmp_path: Path):
    path = _autopilot_short(tmp_path)
    result = run_scenario(path)
    row = next(r for r in result.plot_rows if r["time"] == pytest.approx(0.05))
    assert math.isfinite(row["hbe"])
    assert math.isfinite(row["alt"])
    assert row["msl_time"] >= 0

    vehicle = _run_loop_short(path)
    assert vehicle.health == 1
    assert vehicle.store.get("msl_time") >= 0
    assert math.isfinite(float(vehicle.store.get("hbe")))
    assert math.isfinite(float(vehicle.store.get("alt")))
