from pathlib import Path

from cadac.cli import _resolve_vehicle
from cadac.io.scenario import load_scenario
from cadac.vehicles.flat5.sraam5.vehicle import Sraam5

CASE = Path(__file__).resolve().parents[2] / "cases" / "sraam5" / "inlar1.jsonc"

EXPECTED_MODULES = [
    "target",
    "environment",
    "seeker",
    "ai_radar",
    "ins",
    "guidance",
    "control",
    "aerodynamics",
    "propulsion",
    "forces",
    "newton",
    "rotations",
    "intercept",
]


def test_sraam5_family_registers_and_load_scenario():
    assert _resolve_vehicle("sraam5", "SRAAM5") is Sraam5
    cfg = load_scenario(CASE)
    assert cfg.family == "sraam5"
    assert len(cfg.vehicles) == 1
    vehicle_spec = cfg.vehicles[0]
    assert vehicle_spec.family == "sraam5"
    assert vehicle_spec.type == "SRAAM5"
    assert [module.name for module in cfg.modules] == EXPECTED_MODULES
    vehicle = Sraam5("m", [])
    vehicle.define()
    assert [module.name for module in vehicle.modules] == EXPECTED_MODULES
