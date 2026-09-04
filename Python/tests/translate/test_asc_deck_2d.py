from pathlib import Path
from cadac.io.asc_deck import parse_asc_deck

PROP = Path(__file__).resolve().parents[3] / "CADAC_Simulations/HYPER3_250114/HYPER3/ghame3_prop_deck.asc"

def test_parse_ca_vs_alpha_mach():
    _, tables = parse_asc_deck(PROP)
    t = {x.name: x for x in tables}["ca_vs_alpha_mach"]
    assert t.dim == 2
    assert t.x1.shape == (9,)
    assert t.x2.shape == (13,)
    assert t.values.shape == (9, 13)
    assert t.x1[0] == -3
    assert t.x2[0] == 0.4
    assert t.values[0, 0] == 1.09449
