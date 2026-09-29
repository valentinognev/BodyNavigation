"""FALCON6 tabular weather (MATMO=3, MWIND=3) e2e — python-golden."""

from pathlib import Path

import pytest

from cadac.io.scenario import load_scenario

CASE = Path(__file__).resolve().parents[2] / "cases" / "falcon6" / "inpitch_mair33.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "falcon6" / "weather_tabular.plot.csv"


def test_falcon6_weather_tabular_load_scenario():
    cfg = load_scenario(CASE)
    vehicle = cfg.vehicles[0]
    assert vehicle.type == "PLANE6"
    assert vehicle.params["mair"] == 33
    assert vehicle.weather_deck is not None
    assert vehicle.weather_deck.name == "weather_deck.jsonc"
    assert vehicle.weather_deck.is_file()
    assert cfg.end_time == pytest.approx(0.2)


def test_falcon6_weather_tabular_golden_deferred_until_cli_weather():
    """Trajectory golden needs run_scenario wiring weather_deck into Flat6."""
    if not GOLDEN.is_file():
        pytest.skip(
            "python-golden deferred until cli wires weather_deck to Flat6Environment "
            "(MATMO=3/MWIND=3); vehicle.py/cli untouched in Task 81"
        )
