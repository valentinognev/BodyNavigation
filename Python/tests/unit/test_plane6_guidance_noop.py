from types import SimpleNamespace

import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.falcon6.guidance import Plane6Guidance

DEFINED = (
    "mguid",
    "line_gain",
    "nl_gain_fact",
    "decrement",
    "swel1",
    "swel2",
    "swel3",
    "psiflx",
    "thtflx",
    "dwb",
    "nl_gain",
    "VBEO",
    "VBEF",
    "dwbh",
    "SWBL",
    "turn_min",
    "wp_flag",
)
INT_DATA = ("mguid",)
INT_DIAG = ("wp_flag",)
VEC_DIAG = ("VBEO", "VBEF", "SWBL")
SCRN_PLOT_DIAG = ("nl_gain", "dwbh")
REAL_DATA = (
    "line_gain",
    "nl_gain_fact",
    "decrement",
    "swel1",
    "swel2",
    "swel3",
    "psiflx",
    "thtflx",
)
REAL_DIAG = ("dwb", "nl_gain", "dwbh", "turn_min")
EXTERNALS = (
    "time",
    "halt",
    "grav",
    "SBEL",
    "VBEL",
    "dvbe",
    "psivlx",
    "thtvlx",
    "philimx",
    "phicomx",
    "ancomx",
    "alcomx",
)
CONTROL_OWNED = ("phicomx", "ancomx", "alcomx")


def _ctx(int_step=0.001):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _ready(mguid=0):
    vehicle = SimpleNamespace(store=StateStore())
    guid = Plane6Guidance()
    guid.define(vehicle)
    vehicle.store.set("mguid", mguid)
    guid.initialize(vehicle, _ctx())
    return vehicle, guid


def test_name_is_guidance():
    assert Plane6Guidance().name == "guidance"


def test_define_registers_cpp_def_guidance_fields():
    vehicle = SimpleNamespace(store=StateStore())
    Plane6Guidance().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    for name in DEFINED:
        assert store.field(name).module == "guidance"
    for name in INT_DATA:
        assert store.get(name) == 0
        assert store.field(name).type == "int"
        assert store.field(name).role == "data"
        assert store.field(name).outputs == ("scrn",)
    for name in INT_DIAG:
        assert store.get(name) == 0
        assert store.field(name).type == "int"
        assert store.field(name).role == "diag"
        assert store.field(name).outputs == ("plot",)
    zeros = np.zeros(3)
    for name in VEC_DIAG:
        np.testing.assert_array_equal(store.get(name), zeros)
        assert store.get(name).shape == (3,)
        assert store.field(name).type == "vec"
        assert store.field(name).role == "diag"
        assert store.field(name).outputs == ()
    for name in REAL_DATA:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "data"
        assert store.field(name).outputs == ()
    for name in REAL_DIAG:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "diag"
    for name in SCRN_PLOT_DIAG:
        assert store.field(name).outputs == ("scrn", "plot")
    assert store.field("dwb").outputs == ()
    assert store.field("turn_min").outputs == ()
    for name in EXTERNALS:
        assert name not in store.names()


def test_initialize_is_pass():
    vehicle, _guid = _ready()
    store = vehicle.store
    assert store.get("mguid") == 0
    assert store.get("line_gain") == 0.0
    assert store.get("nl_gain_fact") == 0.0
    assert store.get("wp_flag") == 0
    np.testing.assert_array_equal(store.get("VBEO"), np.zeros(3))
    np.testing.assert_array_equal(store.get("SWBL"), np.zeros(3))


def test_execute_mguid_zero_does_not_raise():
    vehicle, guid = _ready(mguid=0)
    guid.execute(vehicle, _ctx())


def test_execute_mguid_zero_does_not_write_control_or_diagnostics():
    vehicle, guid = _ready(mguid=0)
    store = vehicle.store
    for name in CONTROL_OWNED:
        store.define(Field(name, 7.0, "real", "data", "control"))
    store.set("dwb", 11.0)
    store.set("nl_gain", 12.0)
    store.set("dwbh", 13.0)
    store.set("turn_min", 14.0)
    store.set("wp_flag", 1)
    sentinel = np.array([9.0, 8.0, 7.0])
    store.set("VBEO", sentinel)
    store.set("VBEF", sentinel)
    store.set("SWBL", sentinel)
    guid.execute(vehicle, _ctx())
    for name in CONTROL_OWNED:
        assert store.get(name) == 7.0
    assert store.get("dwb") == 11.0
    assert store.get("nl_gain") == 12.0
    assert store.get("dwbh") == 13.0
    assert store.get("turn_min") == 14.0
    assert store.get("wp_flag") == 1
    np.testing.assert_array_equal(store.get("VBEO"), sentinel)
    np.testing.assert_array_equal(store.get("VBEF"), sentinel)
    np.testing.assert_array_equal(store.get("SWBL"), sentinel)


def test_execute_mguid_zero_does_not_require_newton_or_control_names():
    vehicle, guid = _ready(mguid=0)
    store = vehicle.store
    for name in EXTERNALS:
        assert name not in store.names()
    guid.execute(vehicle, _ctx())
    for name in EXTERNALS:
        assert name not in store.names()


@pytest.mark.parametrize("mguid", (30, 33))
def test_execute_mguid_line_modes_raise(mguid):
    vehicle, guid = _ready(mguid=mguid)
    store = vehicle.store
    for name in CONTROL_OWNED:
        store.define(Field(name, 7.0, "real", "data", "control"))
    with pytest.raises(ValueError, match="unknown mguid"):
        guid.execute(vehicle, _ctx())
    for name in CONTROL_OWNED:
        assert store.get(name) == 7.0
    for name in EXTERNALS:
        if name not in CONTROL_OWNED:
            assert name not in store.names()
    assert store.get("wp_flag") == 0
    np.testing.assert_array_equal(store.get("SWBL"), np.zeros(3))
