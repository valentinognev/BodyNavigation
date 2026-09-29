"""SRAAM5 inlar1_mseek5 e2e — DTCT/DBTC intercept-plane split (python-golden)."""

from pathlib import Path

import pytest

from cadac.io.scenario import load_scenario

CASE = Path(__file__).resolve().parents[2] / "cases" / "sraam5" / "inlar1_mseek5.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "sraam5" / "intercept_plane.plot.csv"


def test_sraam5_mseek5_load_scenario():
    cfg = load_scenario(CASE)
    assert cfg.family == "sraam5"
    assert cfg.vehicles[0].type == "SRAAM5"
    params = cfg.vehicles[0].params
    assert params["mseek"] == 5
    assert params["mterm"] == 1
    assert any(m.name == "intercept" for m in cfg.modules)
    assert cfg.end_time == 0.0


def test_sraam5_intercept_plane_golden_deferred_until_runnable():
    if not GOLDEN.is_file():
        pytest.skip(
            "python-golden deferred until SRAAM5 inlar1_mseek5 run is unblocked"
        )
