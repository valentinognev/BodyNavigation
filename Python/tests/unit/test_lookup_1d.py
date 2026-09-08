import numpy as np
from cadac.tables.lookup import Datadeck, Table

def deck():
    t = Table(name="y", dim=1, x1=np.array([0.0, 10.0]), x2=None, x3=None, values=np.array([0.0, 10.0]))
    return Datadeck.from_tables([t])

def test_midpoint():
    assert deck().look_up("y", 5.0) == 5.0

def test_upper_constant():
    assert deck().look_up("y", 20.0) == 10.0

def test_lower_slope():
    assert deck().look_up("y", -5.0) == -5.0

def test_on_node():
    assert deck().look_up("y", 10.0) == 10.0


def test_repeated_look_up_identical():
    d = deck()
    first = d.look_up("y", 5.0)
    for _ in range(32):
        assert d.look_up("y", 5.0) == first
    assert d.look_up("y", 20.0) == 10.0
    assert d.look_up("y", 5.0) == first


def test_look_up_two_tables_do_not_share_locs():
    t1 = Table(name="y", dim=1, x1=np.array([0.0, 10.0]), x2=None, x3=None, values=np.array([0.0, 10.0]))
    t2 = Table(name="z", dim=1, x1=np.array([0.0, 2.0]), x2=None, x3=None, values=np.array([1.0, 3.0]))
    d = Datadeck.from_tables([t1, t2])
    assert d.look_up("y", 5.0) == 5.0
    assert d.look_up("z", 1.0) == 2.0
    assert d.look_up("y", 5.0) == 5.0


def test_datadeck_has_loc_cache_attribute():
    d = deck()
    d.look_up("y", 5.0)
    assert hasattr(d, "_loc_cache")
    assert "y" in d._loc_cache


def test_find_index_last_key_memo():
    d = deck()
    bp = d.table("y").x1
    loc = d.find_index(1, 5.0, bp)
    assert d._idx_last == ((id(bp), 1, 5.0), loc)
    assert d.find_index(1, 5.0, bp) == loc
    loc_hi = d.find_index(1, 20.0, bp)
    assert loc_hi == 1
    assert d._idx_last == ((id(bp), 1, 20.0), loc_hi)
