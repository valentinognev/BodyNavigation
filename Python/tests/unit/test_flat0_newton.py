import numpy as np
from cadac.eom.flat0 import Flat0Newton
from cadac.kernel.state import StateStore
from cadac.kernel.executive import SimContext


class V:
    def __init__(self):
        self.store = StateStore()


def test_name_is_newton():
    assert Flat0Newton().name == "newton"


def test_flat0_newton_defines_cpp_fields():
    v = V()
    n = Flat0Newton()
    n.define(v)
    for name in ("srel1", "srel2", "srel3"):
        field = v.store.field(name)
        assert field.type == "real"
        assert field.role == "data"
        assert field.module == "newton"
    srel = v.store.field("SREL")
    assert srel.type == "vec"
    assert srel.role == "out"
    assert srel.module == "newton"
    np.testing.assert_allclose(v.store.get("SREL"), [0.0, 0.0, 0.0], rtol=1e-12, atol=1e-14)


def test_flat0_newton_fixed_site():
    v = V()
    n = Flat0Newton()
    n.define(v)
    v.store.set("srel1", 1.0)
    v.store.set("srel2", 2.0)
    v.store.set("srel3", 3.0)
    n.initialize(v, SimContext(0.0, 0.01, 0.0, 0.0, [], 0))
    np.testing.assert_allclose(v.store.get("SREL"), [1.0, 2.0, 3.0], rtol=1e-12, atol=1e-14)
    n.execute(v, SimContext(1.0, 0.01, 0.0, 0.0, [], 0))
    np.testing.assert_allclose(v.store.get("SREL"), [1.0, 2.0, 3.0], rtol=1e-12, atol=1e-14)
