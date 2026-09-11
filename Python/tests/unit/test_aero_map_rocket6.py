from pathlib import Path

from cadac.aero_map import map_payload, payload_from_mdt_rows
from cadac.io.jsonc import load

_PYTHON = Path(__file__).resolve().parents[2]
ROCKET_DECK = _PYTHON / "cases" / "rocket6" / "aero_deck_SLV.jsonc"

GRID_ROWS = [
    {"mach": 0.8, "alpha": 0.0, "cn": 0.1, "cm": -0.01, "ca": 0.3},
    {"mach": 0.8, "alpha": 4.0, "cn": 0.5, "cm": -0.02, "ca": 0.31},
    {"mach": 1.2, "alpha": 0.0, "cn": 0.12, "cm": -0.011, "ca": 0.4},
    {"mach": 1.2, "alpha": 4.0, "cn": 0.6, "cm": -0.03, "ca": 0.41},
]


def _table(deck, name):
    return next(t for t in deck["tables"] if t["name"] == name)


def test_rocket6_maps_slv1_from_ca_cn_cm():
    payload = payload_from_mdt_rows(GRID_ROWS)
    template = load(ROCKET_DECK)
    result = map_payload("rocket6", "HYPER6", payload, template)
    by = {r.name: r.status for r in result.rows}
    assert by["ca0slv1_vs_mach"] == "mapped"
    assert by["cn0slv1_vs_mach_alpha"] == "mapped"
    assert by["clm0slv1_vs_mach_alpha"] == "mapped"
    assert by["caaslv1_vs_mach"] == "merged"
    assert by["ca0bslv1_vs_mach"] == "merged"
    assert by["clmqslv1_vs_mach"] == "merged"
    assert result.can_confirm is True

    ca = _table(result.deck, "ca0slv1_vs_mach")
    assert ca["dim"] == 1
    assert ca["x1"] == [0.8, 1.2]
    assert ca["values"] == [0.3, 0.4]

    cn = _table(result.deck, "cn0slv1_vs_mach_alpha")
    assert cn["dim"] == 2
    assert cn["x1"] == [0.8, 1.2]
    assert cn["x2"] == [0.0, 4.0]
    assert cn["values"] == [[0.1, 0.5], [0.12, 0.6]]

    cm = _table(result.deck, "clm0slv1_vs_mach_alpha")
    assert cm["values"] == [[-0.01, -0.02], [-0.011, -0.03]]

    merged = _table(result.deck, "caaslv1_vs_mach")
    original = _table(template, "caaslv1_vs_mach")
    assert merged["values"] == original["values"]
    assert merged is not original


def test_rocket6_slv3_substitutes_suffix_and_merges_slv3():
    payload = payload_from_mdt_rows(GRID_ROWS)
    template = load(ROCKET_DECK)
    result = map_payload("rocket6", "HYPER6", payload, template, slv=3)
    names = {r.name for r in result.rows}
    assert names == {
        "ca0slv3_vs_mach",
        "caaslv3_vs_mach",
        "ca0bslv3_vs_mach",
        "cn0slv3_vs_mach_alpha",
        "clm0slv3_vs_mach_alpha",
        "clmqslv3_vs_mach",
    }
    by = {r.name: r.status for r in result.rows}
    assert by["ca0slv3_vs_mach"] == "mapped"
    assert by["cn0slv3_vs_mach_alpha"] == "mapped"
    assert by["clm0slv3_vs_mach_alpha"] == "mapped"
    assert by["caaslv3_vs_mach"] == "merged"
    assert by["clmqslv3_vs_mach"] == "merged"
    assert result.can_confirm is True

    ca = _table(result.deck, "ca0slv3_vs_mach")
    assert ca["values"] == [0.3, 0.4]
    merged = _table(result.deck, "caaslv3_vs_mach")
    original = _table(template, "caaslv3_vs_mach")
    slv1 = _table(template, "caaslv1_vs_mach")
    assert merged["values"] == original["values"]
    assert merged["values"] != slv1["values"]


def test_rocket6_slv1_keeps_other_stage_tables():
    payload = payload_from_mdt_rows(GRID_ROWS)
    template = load(ROCKET_DECK)
    result = map_payload("rocket6", "HYPER6", payload, template, slv=1)
    names = {t["name"] for t in result.deck["tables"]}
    assert "ca0slv2_vs_mach" in names
    assert "ca0slv3_vs_mach" in names
    assert len(result.deck["tables"]) == len(template["tables"]) == 18
    slv2 = _table(result.deck, "ca0slv2_vs_mach")
    original = _table(template, "ca0slv2_vs_mach")
    assert slv2["values"] == original["values"]
    assert slv2 is not original
    slv3 = _table(result.deck, "ca0slv3_vs_mach")
    assert slv3 is not _table(template, "ca0slv3_vs_mach")
