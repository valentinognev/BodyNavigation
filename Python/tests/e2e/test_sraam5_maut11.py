"""SRAAM5 inlar1_maut11 e2e — MAUT=11 α/β hold derived case (python-golden)."""

from pathlib import Path

import pytest

from cadac.io.scenario import load_scenario

CASE = Path(__file__).resolve().parents[2] / "cases" / "sraam5" / "inlar1_maut11.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "sraam5" / "maut11.plot.csv"


def test_sraam5_maut11_load_scenario():
    cfg = load_scenario(CASE)
    assert cfg.family == "sraam5"
    assert cfg.vehicles[0].type == "SRAAM5"
    params = cfg.vehicles[0].params
    assert params["maut"] == 11
    assert params["mturn"] == 0
    assert params["alphac"] == pytest.approx(0.12)
    assert params["betac"] == pytest.approx(-0.07)
    assert cfg.end_time == 0.0


def test_sraam5_maut11_golden_deferred_until_runnable():
    if not GOLDEN.is_file():
        pytest.skip(
            "python-golden deferred until SRAAM5 inlar1_maut11 run is unblocked"
        )
