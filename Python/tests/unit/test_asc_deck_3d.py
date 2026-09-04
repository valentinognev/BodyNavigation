from pathlib import Path

from cadac.io.asc_deck import parse_asc_deck

PROP = (
    Path(__file__).resolve().parents[3]
    / "CADAC_Simulations/FALCON5_250116/FALCON5/Falcon5_prop_deck.asc"
)


def test_parse_ff_vs_thrust_alt_mach():
    _, tables = parse_asc_deck(PROP)
    t = {x.name: x for x in tables}["ff_vs_thrust_alt_mach"]
    assert t.dim == 3
    assert t.x1.shape == (6,)
    assert t.x2.shape == (2,)
    assert t.x3.shape == (4,)
    assert t.values.shape == (6, 2, 4)
    assert t.x1[0] == -13344
    assert t.x2[0] == 0
    assert t.x3[0] == 0.4
    assert t.values[0, 0, 0] == 0.0
    assert t.values[0, 1, 3] == 0.2394
    assert t.x1[-1] == 56045
    assert t.x2[-1] == 12000
    assert t.x3[-1] == 1.0
    assert t.values[5, 1, 3] == 1.7136
