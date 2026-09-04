from pathlib import Path
from cadac.io.asc_deck import parse_asc_deck

HYPER3 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/HYPER3_250114/HYPER3/ghame3_aero_deck.asc"

def test_parse_cd0_vs_mach():
    title, tables = parse_asc_deck(HYPER3)
    by = {t.name: t for t in tables}
    t = by["cd0_vs_mach"]
    assert t.dim == 1
    assert t.x1[0] == 0.4
    assert t.values[0] == 0.0340
    assert len(t.x1) == 13
