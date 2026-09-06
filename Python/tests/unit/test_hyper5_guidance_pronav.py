import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.hyper5.guidance import Hyper5Guidance

# Demo 4.7 Terminal pro-nav
PRONAV_GAIN = 3.5
BIAS = 5.0
GRAV = 9.81
CLOSING_SPEED = 240.0
RANGE_GO = 5500.0
WOEB = np.array([0.02, -0.05, 0.01])
UTBB = np.array([0.95, 0.22, -0.22])
TBG = np.array(
    [
        [0.8, 0.0, 0.6],
        [0.0, 1.0, 0.0],
        [-0.6, 0.0, 0.8],
    ]
)

RTOL = 1e-12
ATOL = 1e-14

PRONAV_FIELDS = {
    "pronav_gain": ("real", "data", ()),
    "bias": ("real", "data", ()),
}

NOT_DEFINED = (
    "WOEB",
    "UTBB",
    "closing_speed",
    "range_go",
    "TBG",
    "psisbx",
    "thtsbx",
    "line_gain",
    "nl_gain_fact",
    "decrement",
    "psifgx",
    "thtfgx",
    "nl_gain",
    "VBEF",
    "philimx",
    "alcomx",
    "ancomx",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _skew(vec):
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.05,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _expected_pronav(woeb, utbb, pronav_gain, closing_speed, tbg, grav, bias):
    grav_g = np.array([0.0, 0.0, grav + bias])
    return _skew(woeb) @ utbb * (pronav_gain * closing_speed) - tbg @ grav_g


def _ready(
    pronav_gain=PRONAV_GAIN,
    bias=BIAS,
    grav=GRAV,
    closing_speed=CLOSING_SPEED,
    range_go=RANGE_GO,
    woeb=None,
    utbb=None,
    tbg=None,
    plant_unused=True,
    mguidance=66,
):
    vehicle = _Vehicle()
    guidance = Hyper5Guidance()
    guidance.define(vehicle)
    store = vehicle.store
    if woeb is None:
        woeb = WOEB
    if utbb is None:
        utbb = UTBB
    if tbg is None:
        tbg = TBG
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("TBG", tbg, "mat", "out", "control"))
    store.define(Field("WOEB", woeb, "vec", "out", "seeker"))
    store.define(Field("UTBB", utbb, "vec", "out", "seeker"))
    store.define(Field("closing_speed", closing_speed, "real", "out", "seeker"))
    if plant_unused:
        store.define(Field("range_go", range_go, "real", "out", "seeker"))
    store.set("mguidance", mguidance)
    store.set("pronav_gain", pronav_gain)
    store.set("bias", bias)
    return vehicle, guidance


def test_define_registers_pronav_gain_and_bias():
    vehicle = _Vehicle()
    Hyper5Guidance().define(vehicle)
    store = vehicle.store
    for name, (ftype, role, outputs) in PRONAV_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "guidance"
        assert field.outputs == outputs
        assert store.get(name) == 0.0


def test_define_does_not_register_seeker_tbg_or_unused_locals():
    vehicle = _Vehicle()
    Hyper5Guidance().define(vehicle)
    store = vehicle.store
    for name in NOT_DEFINED:
        with pytest.raises(KeyError):
            store.get(name)


def test_execute_mguidance_0_does_not_write():
    vehicle, guidance = _ready(mguidance=0)
    store = vehicle.store
    woeb = store.get("WOEB").copy()
    utbb = store.get("UTBB").copy()
    tbg = store.get("TBG").copy()
    guidance.execute(vehicle, _ctx())
    assert store.get("pronav_gain") == PRONAV_GAIN
    assert store.get("bias") == BIAS
    assert store.get("closing_speed") == CLOSING_SPEED
    assert store.get("range_go") == RANGE_GO
    assert store.get("mguidance") == 0
    np.testing.assert_array_equal(store.get("WOEB"), woeb)
    np.testing.assert_array_equal(store.get("UTBB"), utbb)
    np.testing.assert_array_equal(store.get("TBG"), tbg)


def test_guidance_pronav_demo_47_matches_cpp():
    vehicle, guidance = _ready()
    expected = _expected_pronav(
        WOEB, UTBB, PRONAV_GAIN, CLOSING_SPEED, TBG, GRAV, BIAS
    )
    got = guidance.guidance_pronav(vehicle)
    assert np.all(np.isfinite(got))
    assert got.shape == (3,)
    assert not np.allclose(got, 0.0)
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


def test_guidance_pronav_does_not_read_unused_cpp_locals():
    vehicle, guidance = _ready(plant_unused=False)
    store = vehicle.store
    for name in ("range_go", "psisbx", "thtsbx"):
        with pytest.raises(KeyError):
            store.get(name)
    expected = _expected_pronav(
        WOEB, UTBB, PRONAV_GAIN, CLOSING_SPEED, TBG, GRAV, BIAS
    )
    got = guidance.guidance_pronav(vehicle)
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)
