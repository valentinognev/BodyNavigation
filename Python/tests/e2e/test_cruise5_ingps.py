"""CRUISE5 INGPS e2e — python-golden stub until GPS+INS trajectory golden exists."""

from pathlib import Path

import pytest

from cadac.io.scenario import load_scenario

CASE = Path(__file__).resolve().parents[2] / "cases" / "cruise5" / "ingps.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "cruise5" / "ingps.plot.csv"


def test_cruise5_ingps_load_scenario_skeleton():
    cfg = load_scenario(CASE)
    assert cfg.family == "cruise5"
    uav = cfg.vehicles[0]
    assert uav.type == "CRUISE3"
    assert uav.params.get("mgps") == 0
    assert uav.params.get("dtimgps") == 1.0
    assert uav.params.get("azgex1") == 60.0
    assert "gps" in [m.name for m in cfg.modules]
    assert cfg.end_time == 0.0


def test_cruise5_ingps_golden_deferred():
    if not GOLDEN.is_file():
        pytest.skip("python-golden deferred until CRUISE5 ingps golden plot.csv exists")
