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
