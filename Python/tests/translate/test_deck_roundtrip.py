from pathlib import Path

import numpy as np

from cadac.io.asc_deck import parse_asc_deck
from cadac.io.deck import load_deck
from cadac.io.translate import deck_asc_to_jsonc

ROOT = Path(__file__).resolve().parents[3]
HYPER3 = ROOT / "CADAC_Simulations/HYPER3_250114/HYPER3"
AERO = HYPER3 / "ghame3_aero_deck.asc"
PROP = HYPER3 / "ghame3_prop_deck.asc"


def _assert_roundtrip(src: Path, dst: Path) -> None:
    deck_asc_to_jsonc(src, dst)
    _, parsed = parse_asc_deck(src)
    reloaded = load_deck(dst)
    assert len(reloaded) == len(parsed)
    for asc_t, json_t in zip(parsed, reloaded, strict=True):
        np.testing.assert_array_equal(json_t.x1, asc_t.x1)
        if asc_t.x2 is None:
            assert json_t.x2 is None
        else:
            np.testing.assert_array_equal(json_t.x2, asc_t.x2)
        np.testing.assert_array_equal(json_t.values, asc_t.values)


def test_aero_deck_roundtrip(tmp_path: Path):
    _assert_roundtrip(AERO, tmp_path / "ghame3_aero_deck.jsonc")


def test_prop_deck_roundtrip(tmp_path: Path):
    _assert_roundtrip(PROP, tmp_path / "ghame3_prop_deck.jsonc")
