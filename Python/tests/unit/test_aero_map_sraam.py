from pathlib import Path

from cadac.aero_map import AeroPayload, map_payload, payload_from_mdt_rows
from cadac.io.jsonc import load

_PYTHON = Path(__file__).resolve().parents[2]
SRAAM_DECK = _PYTHON / "cases" / "sraam6" / "sraam6_aero_deck.jsonc"
AGM_DECK = _PYTHON / "cases" / "agm6" / "AGM6_aero_deck.jsonc"

GRID_ROWS = [
    {"mach": 0.8, "alpha": 0.0, "cn": 0.1, "cm": -0.01, "ca": 0.3},
    {"mach": 0.8, "alpha": 4.0, "cn": 0.5, "cm": -0.02, "ca": 0.31},
    {"mach": 1.2, "alpha": 0.0, "cn": 0.12, "cm": -0.011, "ca": 0.4},
    {"mach": 1.2, "alpha": 4.0, "cn": 0.6, "cm": -0.03, "ca": 0.41},
]


def _grid_payload():
    return payload_from_mdt_rows(GRID_ROWS)


def _table(deck, name):
    return next(t for t in deck["tables"] if t["name"] == name)


def test_sraam_maps_cn_and_merges_rest():
    payload = _grid_payload()
    template = load(SRAAM_DECK)
    result = map_payload("sraam6", "MISSILE6", payload, template)
    by = {r.name: r.status for r in result.rows}
    assert by["cn0_vs_mach_alpha"] == "mapped"
    assert by["clm0_vs_mach_alpha"] == "mapped"
    assert by["ca0_vs_mach"] == "mapped"
    assert by["clmq_vs_mach"] == "merged"
    assert result.can_confirm is True

    cn = _table(result.deck, "cn0_vs_mach_alpha")
    assert cn["dim"] == 2
    assert cn["x1"] == [0.8, 1.2]
    assert cn["x2"] == [0.0, 4.0]
    assert cn["values"] == [[0.1, 0.5], [0.12, 0.6]]

    cm = _table(result.deck, "clm0_vs_mach_alpha")
    assert cm["values"] == [[-0.01, -0.02], [-0.011, -0.03]]

    ca = _table(result.deck, "ca0_vs_mach")
    assert ca["dim"] == 1
    assert ca["x1"] == [0.8, 1.2]
    assert ca["values"] == [0.3, 0.4]

    merged = _table(result.deck, "clmq_vs_mach")
    original = _table(template, "clmq_vs_mach")
    assert merged["values"] == original["values"]
    assert merged is not original
    assert result.deck["title"] == "SRAAM6 aero deck"


def test_missing_without_template():
    payload = _grid_payload()
    result = map_payload("sraam6", "MISSILE6", payload, None)
    assert result.can_confirm is False
    assert any(r.status == "missing" for r in result.rows)
    by = {r.name: r.status for r in result.rows}
    assert by["cn0_vs_mach_alpha"] == "mapped"
    assert by["clmq_vs_mach"] == "missing"


def test_ca0_mean_when_alpha_zero_absent():
    payload = payload_from_mdt_rows(
        [
            {"mach": 0.8, "alpha": 2.0, "cn": 0.1, "cm": -0.01, "ca": 0.25},
            {"mach": 0.8, "alpha": 4.0, "cn": 0.5, "cm": -0.02, "ca": 0.75},
            {"mach": 1.2, "alpha": 2.0, "cn": 0.12, "cm": -0.011, "ca": 0.5},
            {"mach": 1.2, "alpha": 4.0, "cn": 0.6, "cm": -0.03, "ca": 1.5},
        ]
    )
    result = map_payload("sraam6", "MISSILE6", payload, load(SRAAM_DECK))
    ca = _table(result.deck, "ca0_vs_mach")
    assert ca["values"] == [0.5, 1.0]


def test_agm6_shares_sraam_payload_mapping():
    payload = _grid_payload()
    template = load(AGM_DECK)
    result = map_payload("agm6", "MISSILE6", payload, template)
    by = {r.name: r.status for r in result.rows}
    assert by["cn0_vs_mach_alpha"] == "mapped"
    assert by["clm0_vs_mach_alpha"] == "mapped"
    assert by["ca0_vs_mach"] == "mapped"
    assert by["clmq_vs_mach"] == "merged"
    assert result.can_confirm is True


def test_empty_alphas_omits_ca0():
    payload = AeroPayload(
        source="misdc",
        solver="mdt",
        axes={"mach": [0.8], "alpha": [], "beta": []},
        tables={"ca": [[0.3]], "cn": [[]], "cm": [[]]},
        ref={},
    )
    result = map_payload("sraam6", "MISSILE6", payload, load(SRAAM_DECK))
    by = {r.name: r.status for r in result.rows}
    assert by["ca0_vs_mach"] == "missing"
    assert result.can_confirm is False


def test_unknown_family_is_noop():
    payload = _grid_payload()
    template = {"title": "keep me", "tables": [{"name": "x", "dim": 1, "x1": [1.0], "values": [2.0]}]}
    result = map_payload("ghame5", "HYPER5", payload, template)
    assert result.rows == []
    assert result.can_confirm is True
    assert result.deck["title"] == "keep me"
    assert result.deck["tables"] == template["tables"]
    assert result.deck is not template
    assert result.deck["tables"] is not template["tables"]

def test_plane6_without_template_is_missing_not_noop():
    result = map_payload(None, "PLANE6", _grid_payload(), None)
    assert result.rows
    assert all(r.status == "missing" for r in result.rows)
    assert result.can_confirm is False
    assert {r.name for r in result.rows} == {
        "cx_vs_elev_alpha",
        "cxq_vs_alpha",
        "cyr_vs_alpha",
        "cyp_vs_alpha",
        "cz_vs_alpha",
        "czq_vs_alpha",
        "cl_vs_beta_alpha",
        "cldr_vs_beta_alpha",
        "clda_vs_beta_alpha",
        "clr_vs_alpha",
        "clp_vs_alpha",
        "cm_vs_elev_alpha",
        "cmq_vs_alpha",
        "cn_vs_beta_alpha",
        "cnda_vs_beta_alpha",
        "cndr_vs_beta_alpha",
        "cnr_vs_alpha",
        "cnp_vs_alpha",
    }
