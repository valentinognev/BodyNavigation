"""ROCKET3 inlaunch e2e — python-golden stub until Task 89 fills modules."""

from pathlib import Path

import pytest

from cadac.io.scenario import load_scenario

CASE = Path(__file__).resolve().parents[2] / "cases" / "rocket3" / "inlaunch.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "rocket3" / "inlaunch.plot.csv"


def test_rocket3_inlaunch_load_scenario_skeleton():
    cfg = load_scenario(CASE)
    assert cfg.family == "rocket3"
    assert cfg.vehicles[0].type == "ROCKET3"
    assert cfg.end_time == 0.0


def test_rocket3_inlaunch_golden_deferred_until_modules_exist():
    if not GOLDEN.is_file():
        pytest.skip(
            "python-golden deferred until ROCKET3 modules exist (after Task 89)"
        )
