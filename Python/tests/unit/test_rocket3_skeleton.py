from pathlib import Path

from cadac.cli import _resolve_vehicle
from cadac.io.scenario import load_scenario
from cadac.vehicles.round3.rocket3.vehicle import Rocket3

CASE = Path(__file__).resolve().parents[2] / "cases" / "rocket3" / "inlaunch.jsonc"

EXPECTED_MODULES = [
    "environment",
    "propulsion",
    "aerodynamics",
    "forces",
    "newton",
]


def test_rocket3_family_registers_and_load_scenario():
    assert _resolve_vehicle("rocket3", "ROCKET3") is Rocket3
    cfg = load_scenario(CASE)
    assert cfg.family == "rocket3"
    assert len(cfg.vehicles) == 1
    vehicle_spec = cfg.vehicles[0]
    assert vehicle_spec.family == "rocket3"
    assert vehicle_spec.type == "ROCKET3"
    assert [module.name for module in cfg.modules] == EXPECTED_MODULES
    vehicle = Rocket3("m", [])
    vehicle.define()
    assert [module.name for module in vehicle.modules] == EXPECTED_MODULES
