import numpy as np
import pytest

from cadac.constants import WEII3
from cadac.eom.round3 import Round3Newton
from cadac.kernel.executive import SimContext
from cadac.kernel.state import StateStore
from cadac.math.earth import cadsph
from cadac.math.frames import polar_from_cart


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(**kwargs):
    fields = dict(
        sim_time=0.0,
        int_step=0.1,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )
    fields.update(kwargs)
    return SimContext(**fields)


def _defined_newton():
    vehicle = _Vehicle()
    newton = Round3Newton()
    newton.define(vehicle)
    return vehicle, newton


def _hyper3_ics(store):
    store.set("lonx", -80.55)
    store.set("latx", 28.43)
    store.set("alt", 3000.0)
    store.set("psivgx", 90.0)
    store.set("thtvgx", 0.0)
    store.set("dvbe", 250.0)


def test_name_is_newton():
    assert Round3Newton().name == "newton"


def test_define_registers_newton_fields():
    vehicle, newton = _defined_newton()
    for name in (
        "psivg",
        "thtvg",
        "lonx",
        "latx",
        "alt",
        "dvbe",
        "psivgx",
        "thtvgx",
        "altx",
    ):
        assert vehicle.store.get(name) == 0.0
    for name in ("sbii", "vbii", "abii", "sbeg", "vbeg", "sb0ii"):
        np.testing.assert_array_equal(vehicle.store.get(name), np.zeros(3))
    for name in ("tgv", "tig", "tge", "weii"):
        np.testing.assert_array_equal(vehicle.store.get(name), np.zeros((3, 3)))


def test_define_does_not_register_environment_fields():
    vehicle, newton = _defined_newton()
    for name in (
        "time",
        "event_time",
        "int_step_new",
        "out_step_fact",
        "grav",
        "rho",
        "pdynmc",
        "mach",
        "vsound",
        "press",
    ):
        with pytest.raises(KeyError):
            vehicle.store.get(name)


def test_initialize_alt_dvbe_and_cadsph_altitude():
    vehicle, newton = _defined_newton()
    _hyper3_ics(vehicle.store)
    newton.initialize(vehicle, _ctx())
    assert vehicle.store.get("alt") == 3000.0
    np.testing.assert_allclose(vehicle.store.get("dvbe"), 250.0, rtol=1e-12, atol=1e-14)
    sbii = vehicle.store.get("sbii")
    assert abs(cadsph(sbii)[2] - 3000.0) < 1e-6


def test_initialize_weii_skew_sym():
    vehicle, newton = _defined_newton()
    _hyper3_ics(vehicle.store)
    newton.initialize(vehicle, _ctx())
    weii = vehicle.store.get("weii")
    expected = np.zeros((3, 3))
    expected[0, 1] = -WEII3
    expected[1, 0] = WEII3
    np.testing.assert_allclose(weii, expected, rtol=1e-12, atol=1e-14)


def test_initialize_vbeg_heading_east():
    vehicle, newton = _defined_newton()
    _hyper3_ics(vehicle.store)
    newton.initialize(vehicle, _ctx())
    vbeg = vehicle.store.get("vbeg")
    d, az, el = polar_from_cart(vbeg)
    np.testing.assert_allclose(d, 250.0, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(el, 0.0, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(az, np.pi / 2.0, rtol=1e-12, atol=1e-10)
