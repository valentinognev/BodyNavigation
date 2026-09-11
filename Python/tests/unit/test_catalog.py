from pathlib import Path

from cadac.io import catalog
from cadac.io.catalog import (
    SKIP_STEMS,
    classify_asc,
    dest_jsonc,
    family_for,
    repo_root,
    translate_one,
)

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


def test_family_map():
    assert family_for("AGM6") == "agm6"
    assert family_for("HYPER3") is None
    assert family_for("ROCKET6") == "rocket6"


def test_dest_hyper6_sec104():
    root = repo_root()
    src = root / "CADAC_Simulations/HYPER6 Input Problems for Sec 10_4/10_1_1_input_aero.asc"
    dst = dest_jsonc("HYPER6", src)
    assert dst == root / "Python/cases/hyper6/10_1_1_input_aero.jsonc"


def test_translate_one_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "cases_dir", lambda program: tmp_path)
    src = HYPER3 / "input_climb.asc"
    dest = dest_jsonc("HYPER3", src)
    dest.write_text("keep-me")
    original = dest.read_text()
    assert translate_one("HYPER3", src) == "exists"
    assert dest.read_text() == original
