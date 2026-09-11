from pathlib import Path

from cadac.io.catalog import SKIP_STEMS, classify_asc

ROOT = Path(__file__).resolve().parents[2].parent  # BodyNavigation
HYPER3 = ROOT / "CADAC_Simulations/HYPER3_250114/HYPER3"
SEC104 = ROOT / "CADAC_Simulations/HYPER6 Input Problems for Sec 10_4"


def test_skip_stems():
    assert "readme" in SKIP_STEMS
    assert classify_asc(HYPER3 / "readme.asc") == "skip"
    assert classify_asc(HYPER3 / "input_copy.asc") == "skip"


def test_scenario_climb():
    assert classify_asc(HYPER3 / "input_climb.asc") == "scenario"


def test_deck_aero():
    assert classify_asc(HYPER3 / "ghame3_aero_deck.asc") == "deck"


def test_sec104_is_scenario():
    assert classify_asc(SEC104 / "10_1_1_input_aero.asc") == "scenario"
