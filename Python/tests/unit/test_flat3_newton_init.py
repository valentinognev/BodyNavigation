from types import SimpleNamespace

import numpy as np

from cadac.constants import RAD
from cadac.eom.flat3 import Flat3Newton
from cadac.kernel.state import StateStore


def test_name_is_newton():
    assert Flat3Newton().name == "newton"


def test_init_newton_sbel_dvbe_alt():
    s = StateStore()
    vehicle = SimpleNamespace(store=s)
    newton = Flat3Newton()
    newton.define(vehicle)
    s.set("sbel1", 0.0)
    s.set("sbel2", 0.0)
    s.set("sbel3", -3500.0)
    s.set("dvbe", 200.0)
    s.set("psivlx", 0.0)
    s.set("thtvlx", 0.0)
    newton.initialize(vehicle, None)
    assert s.get("SBEL")[2] == -3500
    assert s.get("dvbe") == 200
    # C++ Flat3::init_newton does not load alt (flat3[36]); first exec sees 0
    assert s.get("alt") == 0.0


def test_initialize_vbel_and_tbl_level_north():
    s = StateStore()
    vehicle = SimpleNamespace(store=s)
    newton = Flat3Newton()
    newton.define(vehicle)
    s.set("sbel1", 0.0)
    s.set("sbel2", 0.0)
    s.set("sbel3", -3500.0)
    s.set("dvbe", 200.0)
    s.set("psivlx", 0.0)
    s.set("thtvlx", 0.0)
    newton.initialize(vehicle, None)
    np.testing.assert_allclose(s.get("VBEL"), [200.0, 0.0, 0.0], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(s.get("TBL"), np.eye(3), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(s.get("TVL"), np.eye(3), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(s.get("TBV"), np.eye(3), rtol=1e-12, atol=1e-14)


def test_initialize_vbel_heading_east():
    s = StateStore()
    vehicle = SimpleNamespace(store=s)
    newton = Flat3Newton()
    newton.define(vehicle)
    s.set("sbel1", 0.0)
    s.set("sbel2", 0.0)
    s.set("sbel3", -3500.0)
    s.set("dvbe", 200.0)
    s.set("psivlx", 90.0)
    s.set("thtvlx", 0.0)
    newton.initialize(vehicle, None)
    psivl = 90.0 * RAD
    expected = np.array(
        [
            200.0 * np.cos(0.0) * np.cos(psivl),
            200.0 * np.cos(0.0) * np.sin(psivl),
            200.0 * (-np.sin(0.0)),
        ]
    )
    np.testing.assert_allclose(s.get("VBEL"), expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(s.get("psivl"), psivl, rtol=1e-12, atol=1e-14)


def test_define_does_not_register_environment_fields():
    s = StateStore()
    Flat3Newton().define(SimpleNamespace(store=s))
    assert "alt" in s.names()
    for name in ("grav", "rho", "pdynmc", "mach", "vsound", "press"):
        assert name not in s.names()


def test_define_skips_existing_tbv():
    from cadac.kernel.state import Field

    s = StateStore()
    existing = ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0))
    s.define(Field("TBV", existing, "mat", "out", "control"))
    Flat3Newton().define(SimpleNamespace(store=s))
    np.testing.assert_array_equal(s.get("TBV"), np.diag([1.0, 2.0, 3.0]))
    assert s.field("TBV").module == "control"
