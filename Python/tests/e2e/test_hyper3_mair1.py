"""HYPER3 MAIR=1 tabular atmosphere e2e — python-golden."""

from pathlib import Path

import pytest

from cadac.io.scenario import load_scenario

CASE = Path(__file__).resolve().parents[2] / "cases" / "hyper3" / "input_mair1.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "hyper3" / "mair1.plot.csv"


def test_hyper3_mair1_load_scenario():
    cfg = load_scenario(CASE)
    vehicle = cfg.vehicles[0]
    assert vehicle.type == "CRUISE3"
    assert vehicle.params["mair"] == 1
    assert vehicle.weather_deck is not None
    assert vehicle.weather_deck.name == "weather_deck_Wallops.jsonc"
    assert vehicle.weather_deck.is_file()
    assert cfg.end_time == pytest.approx(0.2)


def test_hyper3_mair1_golden_deferred_until_cli_weather():
    """Trajectory golden needs run_scenario wiring weather_deck into Cruise3."""
    if not GOLDEN.is_file():
        pytest.skip(
            "python-golden deferred until cli wires weather_deck to Cruise3 (MAIR=1)"
        )
