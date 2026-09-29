"""SRAAM5 AI radar S2 e2e — python-golden (load + deferred trajectory)."""

from pathlib import Path

import pytest

from cadac.io.scenario import load_scenario

CASE = Path(__file__).resolve().parents[2] / "cases" / "sraam5" / "inlar1_ntag.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "sraam5" / "inlar1_ntag.plot.csv"


def test_sraam5_ai_radar_ntag_load_scenario():
    cfg = load_scenario(CASE)
    assert cfg.family == "sraam5"
    assert cfg.vehicles[0].type == "SRAAM5"
    params = cfg.vehicles[0].params
    assert params["ntag"] == 1
    assert params["dtimtu"] == pytest.approx(0.1)
    assert params["dtimup"] == pytest.approx(1.0)
    assert "ai_radar" in [m.name for m in cfg.modules]


def test_sraam5_ai_radar_golden_deferred_until_runnable():
    if not GOLDEN.is_file():
        pytest.skip(
            "python-golden deferred until SRAAM5 full trajectory run is available"
        )
