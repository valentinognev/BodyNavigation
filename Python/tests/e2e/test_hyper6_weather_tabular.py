"""HYPER6 GHAME6 tabular weather (MATMO=3, MWIND=3) e2e — python-golden."""

from pathlib import Path

import pytest

from cadac.io.scenario import load_scenario

CASE = Path(__file__).resolve().parents[2] / "cases" / "hyper6" / "inclimb_mair033.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "hyper6" / "weather_tabular.plot.csv"


def test_hyper6_weather_tabular_load_scenario():
    cfg = load_scenario(CASE)
    vehicle = cfg.vehicles[0]
    assert vehicle.type == "HYPER6"
    assert vehicle.params["mair"] == 33
    assert vehicle.weather_deck is not None
    assert vehicle.weather_deck.name == "weather_deck_Wallops.jsonc"
    assert vehicle.weather_deck.is_file()
    assert cfg.end_time == pytest.approx(0.2)


def test_hyper6_weather_tabular_golden_deferred_until_cli_weather():
    """Trajectory golden needs run_scenario wiring weather_deck into Hyper6."""
    if not GOLDEN.is_file():
        pytest.skip(
            "python-golden deferred until cli wires weather_deck to Hyper6 "
            "(MATMO=3/MWIND=3); vehicle.py/cli untouched in Task 83"
        )
