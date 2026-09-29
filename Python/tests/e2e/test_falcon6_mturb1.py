"""FALCON6 Dryden MTURB=1 e2e — python-golden (Task 80)."""

from pathlib import Path

import pytest

from cadac.io.scenario import load_scenario

CASE = Path(__file__).resolve().parents[2] / "cases" / "falcon6" / "inpitch_mturb1.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "falcon6" / "inpitch_mturb1.plot.csv"


def test_falcon6_dryden_load_scenario():
    cfg = load_scenario(CASE)
    vehicle = cfg.vehicles[0]
    assert vehicle.type == "PLANE6"
    assert vehicle.params["mair"] == 100
    assert vehicle.params["turb_length"] == 100
    assert vehicle.params["turb_sigma"] == 0.5
    assert cfg.end_time == pytest.approx(0.2)


def test_falcon6_dryden_golden_deferred():
    """Trajectory golden after unit G2TURB lock; harvest optional."""
    if not GOLDEN.is_file():
        pytest.skip("python-golden deferred until inpitch_mturb1.plot.csv is harvested")
