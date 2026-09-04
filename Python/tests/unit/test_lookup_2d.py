import numpy as np
from cadac.tables.lookup import Datadeck, Table

def deck():
    t = Table(
        name="z", dim=2,
        x1=np.array([0.0, 10.0]), x2=np.array([0.0, 10.0]), x3=None,
        values=np.array([[0.0, 10.0], [10.0, 20.0]]),
    )
    return Datadeck.from_tables([t])

def test_center():
    assert deck().look_up("z", 5.0, 5.0) == 10.0

def test_upper_x1_constant():
    assert deck().look_up("z", 20.0, 0.0) == 10.0
