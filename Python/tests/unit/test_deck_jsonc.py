import numpy as np
from pathlib import Path

import pytest

from cadac.io.deck import load_deck
from cadac.tables.lookup import Datadeck, Table


def test_load_1d_table(tmp_path: Path):
    p = tmp_path / "d.jsonc"
    p.write_text(
        '{ "title": "t", "tables": [ { "name": "cd0_vs_mach", "dim": 1, '
        '"x1": [0.4, 0.6], "values": [0.034, 0.0337] } ] }'
    )
    tables = load_deck(p)
    t = tables[0]
    assert t.name == "cd0_vs_mach"
    assert t.dim == 1
    np.testing.assert_allclose(t.x1, [0.4, 0.6])
    np.testing.assert_allclose(t.values, [0.034, 0.0337])


def test_load_1d_axes_x2_x3_none(tmp_path: Path):
    p = tmp_path / "d.jsonc"
    p.write_text(
        '{ "title": "t", "tables": [ { "name": "cd0_vs_mach", "dim": 1, '
        '"x1": [0.4, 0.6], "values": [0.034, 0.0337] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    t = load_deck(p)[0]
    assert t.x2 is None
    assert t.x3 is None
    assert t.values.shape == (2,)


def test_load_2d_table(tmp_path: Path):
    p = tmp_path / "d.jsonc"
    p.write_text(
        '{ "title": "t", "tables": [ { "name": "cla", "dim": 2, '
        '"x1": [0.4, 0.8], "x2": [0.0, 10.0], '
        '"values": [[0.1, 0.2], [0.3, 0.4]] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    t = load_deck(p)[0]
    assert t.name == "cla"
    assert t.dim == 2
    np.testing.assert_allclose(t.x1, [0.4, 0.8])
    np.testing.assert_allclose(t.x2, [0.0, 10.0])
    assert t.x3 is None
    np.testing.assert_allclose(t.values, [[0.1, 0.2], [0.3, 0.4]])
    assert t.values.shape == (2, 2)


def test_load_3d_table(tmp_path: Path):
    p = tmp_path / "d.jsonc"
    p.write_text(
        '{ "title": "t", "tables": [ { "name": "clq", "dim": 3, '
        '"x1": [0.4, 0.8], "x2": [0.0, 10.0], "x3": [-1.0, 1.0], '
        '"values": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    t = load_deck(p)[0]
    assert t.name == "clq"
    assert t.dim == 3
    np.testing.assert_allclose(t.x1, [0.4, 0.8])
    np.testing.assert_allclose(t.x2, [0.0, 10.0])
    np.testing.assert_allclose(t.x3, [-1.0, 1.0])
    np.testing.assert_allclose(
        t.values, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    )
    assert t.values.shape == (2, 2, 2)


def test_load_shape_mismatch_raises_valueerror(tmp_path: Path):
    p = tmp_path / "d.jsonc"
    p.write_text(
        '{ "title": "t", "tables": [ { "name": "bad", "dim": 1, '
        '"x1": [0.4, 0.6], "values": [0.034] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError):
        load_deck(p)


def test_load_strips_jsonc_comments(tmp_path: Path):
    p = tmp_path / "d.jsonc"
    p.write_text(
        '{ "title": "t", // deck title\n'
        '  "tables": [ { "name": "cd0_vs_mach", "dim": 1, '
        '"x1": [0.4, 0.6], "values": [0.034, 0.0337] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    t = load_deck(p)[0]
    assert t.name == "cd0_vs_mach"


def test_datadeck_from_tables_lookup():
    t = Table(
        name="cd0_vs_mach",
        dim=1,
        x1=np.array([0.4, 0.6]),
        x2=None,
        x3=None,
        values=np.array([0.034, 0.0337]),
    )
    deck = Datadeck.from_tables([t])
    got = deck.table("cd0_vs_mach")
    assert got is t


def test_datadeck_missing_table_raises_keyerror():
    t = Table(
        name="cd0_vs_mach",
        dim=1,
        x1=np.array([0.4, 0.6]),
        x2=None,
        x3=None,
        values=np.array([0.034, 0.0337]),
    )
    deck = Datadeck.from_tables([t])
    with pytest.raises(KeyError):
        deck.table("missing")
