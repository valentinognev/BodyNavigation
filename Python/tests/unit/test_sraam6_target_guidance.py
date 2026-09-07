import math

import numpy as np
import pytest

from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.sraam6.target import Sraam6TargetGuidance

RTOL = 1e-12
ATOL = 1e-14

GRAV = 9.8
GTURN = 1.0
GUID_GAIN = 1.0
SAEL = np.array([10000.0, 500.0, -2000.0])
SBEL = np.array([0.0, 0.0, -5000.0])
VAEL = np.array([0.0, 250.0, 0.0])
VBEL = np.array([250.0, 0.0, 0.0])

DEFINED = ("msl_num", "tgt_option", "guid_gain", "ACOML", "gturn")
INT_DATA = ("msl_num", "tgt_option")
REAL_DATA = ("guid_gain", "gturn")
EXTERNALS = ("SAEL", "VAEL", "TVL", "grav")


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(combus=None):
    return SimContext(
        sim_time=0.0,
        int_step=0.001,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=1,
    )


def _missile_packet(*, mseek=4, sbel=SBEL, vbel=VBEL):
    return Packet(
        name="m1",
        type="MISSILE6",
        status=1,
        vars={
            "mseek": mseek,
            "SBEL": np.asarray(sbel, dtype=float),
            "VBEL": np.asarray(vbel, dtype=float),
        },
    )


def _target_packet():
    return Packet(
        name="t1",
        type="TARGET3",
        status=1,
        vars={"SAEL": SAEL.copy(), "VAEL": VAEL.copy()},
    )


def _plant_externals(store, *, sael=SAEL, vael=VAEL, tvl=None, grav=GRAV):
    if tvl is None:
        tvl = np.eye(3)
    store.define(Field("SAEL", sael, "vec", "state", "newton"))
    store.define(Field("VAEL", vael, "vec", "state", "newton"))
    store.define(Field("TVL", tvl, "mat", "out", "newton"))
    store.define(Field("grav", grav, "real", "out", "environment"))


def _ready(
    *,
    tgt_option=1,
    gturn=GTURN,
    guid_gain=GUID_GAIN,
    msl_num=1,
    grav=GRAV,
    tvl=None,
    sael=SAEL,
    vael=VAEL,
    combus=None,
    plant=True,
):
    vehicle = _Vehicle()
    guidance = Sraam6TargetGuidance()
    guidance.define(vehicle)
    if plant:
        _plant_externals(vehicle.store, sael=sael, vael=vael, tvl=tvl, grav=grav)
    store = vehicle.store
    store.set("tgt_option", tgt_option)
    store.set("gturn", gturn)
    store.set("guid_gain", guid_gain)
    store.set("msl_num", msl_num)
    guidance.initialize(vehicle, _ctx(combus))
    return vehicle, guidance, _ctx(combus)


def test_name_is_guidance():
    assert Sraam6TargetGuidance().name == "guidance"


def test_define_registers_def_guidance_fields():
    vehicle = _Vehicle()
    Sraam6TargetGuidance().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    for name in INT_DATA:
        assert store.field(name).type == "int"
        assert store.field(name).role == "data"
        assert store.field(name).module == "guidance"
        assert store.get(name) == 0
    for name in REAL_DATA:
        assert store.field(name).type == "real"
        assert store.field(name).role == "data"
        assert store.field(name).module == "guidance"
        assert store.get(name) == 0.0
    assert store.field("ACOML").type == "vec"
    assert store.field("ACOML").role == "out"
    assert store.field("ACOML").module == "guidance"
    np.testing.assert_allclose(store.get("ACOML"), np.zeros(3), rtol=RTOL, atol=ATOL)
    for name in EXTERNALS:
        assert name not in store.names()


def test_option_1_gturn_identity_tvl():
    vehicle, guidance, ctx = _ready(tgt_option=1, gturn=GTURN, grav=GRAV, tvl=np.eye(3))
    guidance.execute(vehicle, ctx)
    want = np.array([0.0, GTURN * GRAV, -GRAV])
    np.testing.assert_allclose(vehicle.store.get("ACOML"), want, rtol=RTOL, atol=ATOL)


def test_option_0_gravity_bias():
    vehicle, guidance, ctx = _ready(tgt_option=0, grav=GRAV)
    guidance.execute(vehicle, ctx)
    want = np.array([0.0, 0.0, -GRAV])
    np.testing.assert_allclose(vehicle.store.get("ACOML"), want, rtol=RTOL, atol=ATOL)


def test_option_3_raises():
    vehicle, guidance, ctx = _ready(tgt_option=3)
    with pytest.raises(ValueError, match="tgt_option"):
        guidance.execute(vehicle, ctx)


def test_option_2_mseek_4_finite_acoml():
    combus = [_target_packet(), _missile_packet(mseek=4)]
    vehicle, guidance, ctx = _ready(tgt_option=2, msl_num=1, combus=combus)
    guidance.execute(vehicle, ctx)
    acoml = vehicle.store.get("ACOML")
    assert np.all(np.isfinite(acoml))
    # SAEL-SBEL=(10000,500,3000); VAEL=(0,250,0); VBEL=(250,0,0)
    # ||VAEL×VBEL||=250*250; UVBEL=(1,0,0); UVAEL=(0,1,0)
    # EPSL=UVAEL×UVBEL=(0,0,-1); EPSL×UVAEL=(1,0,0)
    dab = math.sqrt(10000.0**2 + 500.0**2 + 3000.0**2)
    gain = GUID_GAIN * (250.0 * 250.0) / dab
    want = np.array([gain, 0.0, -GRAV])
    np.testing.assert_allclose(acoml, want, rtol=RTOL, atol=ATOL)


def test_option_2_seeker_not_mode_4_leaves_acoml_zero():
    combus = [_target_packet(), _missile_packet(mseek=2)]
    vehicle, guidance, ctx = _ready(tgt_option=2, msl_num=1, combus=combus)
    vehicle.store.set("ACOML", (1.0, 2.0, 3.0))
    guidance.execute(vehicle, ctx)
    np.testing.assert_allclose(
        vehicle.store.get("ACOML"), np.zeros(3), rtol=RTOL, atol=ATOL
    )
