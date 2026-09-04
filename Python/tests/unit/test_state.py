import numpy as np
import pytest

from cadac.kernel.state import Field, StateStore


def test_define_get_set():
    s = StateStore()
    s.define(Field("time", 0.0, "real", "exec", "environment"))
    assert s.get("time") == 0.0
    s.set("time", 10.01)
    assert s.get("time") == 10.01
    assert s.names() == ["time"]


def test_duplicate_define_raises_valueerror():
    s = StateStore()
    s.define(Field("time", 0.0, "real", "exec", "environment"))
    with pytest.raises(ValueError):
        s.define(Field("time", 1.0, "real", "exec", "environment"))


def test_unknown_get_set_raises_keyerror():
    s = StateStore()
    with pytest.raises(KeyError):
        s.get("time")
    with pytest.raises(KeyError):
        s.set("time", 1.0)


def test_int_vs_real():
    s = StateStore()
    s.define(Field("time", 0.0, "real", "exec", "environment"))
    s.define(Field("mprop", 1, "int", "data", "propulsion"))
    assert type(s.get("mprop")) is int
    assert s.get("mprop") == 1
    s.set("mprop", 2)
    assert type(s.get("mprop")) is int
    assert s.get("mprop") == 2
    s.set("time", 10.0)
    assert type(s.get("time")) is float
    assert s.get("time") == 10.0


def test_vec_mat_shape():
    s = StateStore()
    s.define(Field("sbii", np.zeros(3), "vec", "state", "newton"))
    s.define(Field("TEI", np.eye(3), "mat", "state", "newton"))
    np.testing.assert_allclose(s.get("sbii"), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(s.get("TEI"), np.eye(3))
    s.set("sbii", np.array([1.0, 2.0, 3.0]))
    s.set("TEI", np.eye(3) * 2)
    np.testing.assert_allclose(s.get("sbii"), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(s.get("TEI"), np.eye(3) * 2)
    with pytest.raises(ValueError):
        s.set("sbii", np.zeros(4))
    with pytest.raises(ValueError):
        s.set("TEI", np.zeros((2, 2)))
    with pytest.raises(ValueError):
        s.define(Field("bad_vec", np.zeros(2), "vec", "state", "newton"))
    with pytest.raises(ValueError):
        s.define(Field("bad_mat", np.zeros((3, 2)), "mat", "state", "newton"))
