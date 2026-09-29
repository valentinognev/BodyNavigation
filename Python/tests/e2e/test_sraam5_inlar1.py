"""SRAAM5 inlar1 e2e — python-golden stub until Task 64 fills modules."""

from pathlib import Path

import pytest

from cadac.io.scenario import load_scenario

CASE = Path(__file__).resolve().parents[2] / "cases" / "sraam5" / "inlar1.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "sraam5" / "inlar1.plot.csv"


def test_sraam5_inlar1_load_scenario_skeleton():
    cfg = load_scenario(CASE)
    assert cfg.family == "sraam5"
    assert cfg.vehicles[0].type == "SRAAM5"
    assert cfg.end_time == 0.0


def test_sraam5_inlar1_golden_deferred_until_modules_exist():
    if not GOLDEN.is_file():
        pytest.skip(
            "python-golden deferred until SRAAM5 modules exist (after Task 64)"
        )
