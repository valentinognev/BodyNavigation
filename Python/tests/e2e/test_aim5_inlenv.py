"""AIM5 inlenv e2e — python-golden stub until trajectory golden exists."""

from pathlib import Path

import pytest

from cadac.io.scenario import load_scenario

CASE = Path(__file__).resolve().parents[2] / "cases" / "aim5" / "inlenv.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "aim5" / "inlenv.plot.csv"


def test_aim5_inlenv_load_scenario_skeleton():
    cfg = load_scenario(CASE)
    assert cfg.family == "aim5"
    assert cfg.vehicles[0].type == "AIM5"
    assert cfg.vehicles[0].params.get("mtarg") == 1
    assert cfg.end_time == 0.0


def test_aim5_inlenv_golden_deferred():
    if not GOLDEN.is_file():
        pytest.skip("python-golden deferred until AIM5 inlenv golden plot.csv exists")
