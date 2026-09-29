"""SRAAM5 inlar1_mturn1 e2e — MTURN=1 bank-to-turn derived case (python-golden)."""

from pathlib import Path

import pytest

from cadac.io.scenario import load_scenario

CASE = Path(__file__).resolve().parents[2] / "cases" / "sraam5" / "inlar1_mturn1.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "sraam5" / "mturn1.plot.csv"


def test_sraam5_mturn1_load_scenario():
    cfg = load_scenario(CASE)
    assert cfg.family == "sraam5"
    assert cfg.vehicles[0].type == "SRAAM5"
    assert cfg.vehicles[0].params["mturn"] == 1
    assert cfg.vehicles[0].params["maut"] == 44
    assert cfg.end_time == 0.0


def test_sraam5_mturn1_golden_deferred_until_runnable():
    if not GOLDEN.is_file():
        pytest.skip(
            "python-golden deferred until SRAAM5 inlar1_mturn1 run is unblocked"
        )
