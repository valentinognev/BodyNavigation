from pathlib import Path

from cadac.aero_map import map_payload, payload_from_mdt_rows
from cadac.io.jsonc import load

_PYTHON = Path(__file__).resolve().parents[2]
SAM_DECK = _PYTHON / "cases" / "sam6" / "SAM_aero_deck.jsonc"

GRID_ROWS = [
    {"mach": 0.8, "alpha": 0.0, "cn": 0.1, "cm": -0.01, "ca": 0.3},
    {"mach": 0.8, "alpha": 4.0, "cn": 0.5, "cm": -0.02, "ca": 0.31},
    {"mach": 1.2, "alpha": 0.0, "cn": 0.12, "cm": -0.011, "ca": 0.4},
    {"mach": 1.2, "alpha": 4.0, "cn": 0.6, "cm": -0.03, "ca": 0.41},
]


def _table(deck, name):
    return next(t for t in deck["tables"] if t["name"] == name)


def test_sam6_maps_cn0_clm0_ca0_3d_and_merges_rest():
    payload = payload_from_mdt_rows(GRID_ROWS)
    assert payload.axes["beta"] == []
    template = load(SAM_DECK)
    result = map_payload("sam6", "MISSILE6", payload, template)
    by = {r.name: r.status for r in result.rows}
    assert by["cn0_vs_mach,betax,alphax"] == "mapped"
    assert by["clm0_vs_mach,betax,alphax"] == "mapped"
    assert by["ca0_vs_mach,betax,alphax"] == "mapped"
    assert by["cad_vs_mach"] == "merged"
    assert by["clmq_vs_mach"] == "merged"
    assert result.can_confirm is True

    cn = _table(result.deck, "cn0_vs_mach,betax,alphax")
    assert cn["dim"] == 3
    assert cn["x1"] == [0.8, 1.2]
    assert cn["x2"] == [0.0]
    assert cn["x3"] == [0.0, 4.0]
    assert cn["values"] == [[[0.1, 0.5]], [[0.12, 0.6]]]

    cm = _table(result.deck, "clm0_vs_mach,betax,alphax")
    assert cm["dim"] == 3
    assert cm["x2"] == [0.0]
    assert cm["values"] == [[[-0.01, -0.02]], [[-0.011, -0.03]]]

    ca = _table(result.deck, "ca0_vs_mach,betax,alphax")
    assert ca["dim"] == 3
    assert ca["x1"] == [0.8, 1.2]
    assert ca["x2"] == [0.0]
    assert ca["x3"] == [0.0, 4.0]
    assert ca["values"] == [[[0.3, 0.31]], [[0.4, 0.41]]]

    merged = _table(result.deck, "cad_vs_mach")
    original = _table(template, "cad_vs_mach")
    assert merged["values"] == original["values"]
    assert merged is not original


def test_sam6_without_template_mapped_only():
    result = map_payload("sam6", "MISSILE6", payload_from_mdt_rows(GRID_ROWS), None)
    assert result.can_confirm is False
    by = {r.name: r.status for r in result.rows}
    assert by["cn0_vs_mach,betax,alphax"] == "mapped"
    assert by["cad_vs_mach"] == "missing"


def test_sam6_multi_beta_wraps_single_x2():
    rows = [
        {"mach": 0.8, "alpha": 0.0, "beta": -4.0, "cn": 0.1, "cm": -0.01, "ca": 0.3},
        {"mach": 0.8, "alpha": 4.0, "beta": 0.0, "cn": 0.5, "cm": -0.02, "ca": 0.31},
        {"mach": 1.2, "alpha": 0.0, "beta": 4.0, "cn": 0.12, "cm": -0.011, "ca": 0.4},
        {"mach": 1.2, "alpha": 4.0, "beta": 4.0, "cn": 0.6, "cm": -0.03, "ca": 0.41},
    ]
    payload = payload_from_mdt_rows(rows)
    assert payload.axes["beta"] == [-4.0, 0.0, 4.0]
    result = map_payload("sam6", "MISSILE6", payload, load(SAM_DECK))
    cn = _table(result.deck, "cn0_vs_mach,betax,alphax")
    assert cn["dim"] == 3
    assert cn["x2"] == [0.0]
    assert len(cn["x2"]) == 1
    assert cn["values"] == [[[0.1, 0.5]], [[0.12, 0.6]]]
    assert len(cn["values"][0]) == 1
    ca = _table(result.deck, "ca0_vs_mach,betax,alphax")
    assert ca["x2"] == [0.0]
    assert len(ca["values"][0]) == 1
