from pathlib import Path

from cadac.aero_map import map_payload, payload_from_aid
from cadac.io.jsonc import load

_PYTHON = Path(__file__).resolve().parents[2]
CRUISE_DECK = _PYTHON / "cases" / "cruise5" / "cruise3_aero_deck.jsonc"

AID = {
    "alpha": [-2.0, 0.0, 4.0],
    "CL": [0.0, 0.2, 0.6],
    "CD": [0.02, 0.025, 0.04],
    "Cm": [0.0, -0.01, -0.03],
    "MACH": 0.2,
}


def _table(deck, name):
    return next(t for t in deck["tables"] if t["name"] == name)


def test_cruise5_maps_cd0_cl0_cla_and_merges_ckk():
    payload = payload_from_aid(AID)
    template = load(CRUISE_DECK)
    result = map_payload("cruise5", "CRUISE3", payload, template)
    by = {r.name: r.status for r in result.rows}
    assert by["cd0_vs_mach"] == "mapped"
    assert by["cl0_vs_mach"] == "mapped"
    assert by["cla_vs_mach"] == "mapped"
    assert by["ckk_vs_mach"] == "merged"
    assert by["cla0_vs_mach"] == "merged"
    assert result.can_confirm is True

    cd0 = _table(result.deck, "cd0_vs_mach")
    assert cd0["dim"] == 1
    assert cd0["x1"] == [0.2]
    assert cd0["values"] == [0.025]

    cl0 = _table(result.deck, "cl0_vs_mach")
    assert cl0["dim"] == 1
    assert cl0["x1"] == [0.2]
    assert cl0["values"] == [0.2]

    cla = _table(result.deck, "cla_vs_mach")
    assert cla["dim"] == 1
    assert cla["x1"] == [0.2]
    assert cla["values"] == [0.1]

    merged = _table(result.deck, "ckk_vs_mach")
    original = _table(template, "ckk_vs_mach")
    assert merged["values"] == original["values"]
    assert merged is not original


def test_cruise5_cla_missing_when_alpha_span_zero():
    payload = payload_from_aid(
        {
            "alpha": [0.0],
            "CL": [0.2],
            "CD": [0.025],
            "Cm": [-0.01],
            "MACH": 0.2,
        }
    )
    result = map_payload("cruise5", "CRUISE3", payload, load(CRUISE_DECK))
    by = {r.name: r.status for r in result.rows}
    assert by["cd0_vs_mach"] == "mapped"
    assert by["cl0_vs_mach"] == "mapped"
    assert by["cla_vs_mach"] == "missing"
    assert result.can_confirm is False
    names = {t["name"] for t in result.deck["tables"]}
    assert "cla_vs_mach" not in names
