from pathlib import Path

import numpy as np

from cadac.io.asc_deck import parse_asc_deck
from cadac.tables.lookup import Datadeck, Table

PROP = (
    Path(__file__).resolve().parents[3]
    / "CADAC_Simulations/FALCON5_250116/FALCON5/Falcon5_prop_deck.asc"
)


def deck():
    t = Table(
        name="w",
        dim=3,
        x1=np.array([0.0, 10.0]),
        x2=np.array([0.0, 10.0]),
        x3=np.array([0.0, 10.0]),
        values=np.array(
            [[[0.0, 10.0], [10.0, 20.0]], [[10.0, 20.0], [20.0, 30.0]]]
        ),
    )
    return Datadeck.from_tables([t])


def test_grid_corner_from_falcon5_prop():
    _, tables = parse_asc_deck(PROP)
    d = Datadeck.from_tables(tables)
    assert d.look_up("ff_vs_thrust_alt_mach", -13344.0, 0.0, 0.4) == 0.0
    assert d.look_up("ff_vs_thrust_alt_mach", 56045.0, 12000.0, 1.0) == 1.7136


def test_center():
    assert deck().look_up("w", 5.0, 5.0, 5.0) == 15.0


def test_upper_x1_constant():
    assert deck().look_up("w", 20.0, 0.0, 0.0) == 10.0


def test_lower_x1_slope():
    assert deck().look_up("w", -5.0, 0.0, 0.0) == -5.0
