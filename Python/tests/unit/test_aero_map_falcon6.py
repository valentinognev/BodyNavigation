from pathlib import Path

from cadac.aero_map import map_payload, payload_from_aid
from cadac.io.jsonc import load

_PYTHON = Path(__file__).resolve().parents[2]
F16_DECK = _PYTHON / "cases" / "falcon6" / "f16_aero_deck.jsonc"

AID = {
    "alpha": [-2.0, 0.0, 4.0],
    "CL": [0.0, 0.2, 0.6],
    "CD": [0.02, 0.02, 0.04],
    "Cm": [0.0, -0.01, -0.03],
    "MACH": 0.2,
}


def test_falcon6_polar_cannot_confirm_without_template():
    result = map_payload(None, "PLANE6", payload_from_aid(AID), None)
    assert result.can_confirm is False


def test_falcon6_template_merge_confirms():
    template = load(F16_DECK)
    result = map_payload(None, "PLANE6", payload_from_aid(AID), template)
    assert result.can_confirm is True
    assert all(r.status == "merged" for r in result.rows)
    assert result.deck["title"] == template["title"]
    cx = next(t for t in result.deck["tables"] if t["name"] == "cx_vs_elev_alpha")
    original = next(t for t in template["tables"] if t["name"] == "cx_vs_elev_alpha")
    assert cx["values"] == original["values"]
    assert cx is not original
