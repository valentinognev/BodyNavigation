import numpy as np
import pytest
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.cruise5.intercept import Cruise5Intercept

def _vehicle():
    v = type("V", (), {})()
    v.store = StateStore()
    v.health = 1
    return v

def _ctx(status=1):
    pkt = Packet(name="UAV", type="CRUISE3", status=status, vars={})
    return SimContext(0.0, 0.05, 0.0, 0.0, [pkt], 0)

def test_ground_impact_kills():
    v = _vehicle()
    mod = Cruise5Intercept()
    mod.define(v)
    for name, value, ftype in (
        ("alt", -1.0, "real"), ("time", 1.0, "real"), ("dvbe", 200.0, "real"),
        ("psivgx", 90.0, "real"), ("thtvgx", 0.0, "real"),
        ("sbeg", np.zeros(3), "vec"), ("mguidance", 30, "int"),
        ("wp_lonx", 14.9, "real"), ("wp_latx", 35.4, "real"), ("wp_alt", 0.0, "real"),
        ("SWBG", np.array([10.0, 20.0, 30.0]), "vec"), ("wp_flag", 0, "int"),
        ("mseeker", 0, "int"), ("range_go", 0.0, "real"),
        ("STBG", np.zeros(3), "vec"), ("closing_speed", 0.0, "real"),
        ("targ_com_slot", 0, "int"),
    ):
        if name not in v.store.names():
            v.store.define(Field(name, value, ftype, "data", "test"))
    ctx = _ctx()
    mod.execute(v, ctx)
    assert v.health == 0
    assert ctx.combus[0].status == 0
    assert v.store.get("write") == 0

def test_mguidance_43_alt_below_wp_alt_kills():
    v = _vehicle()
    mod = Cruise5Intercept()
    mod.define(v)
    swbg = np.array([3.0, 4.0, 0.0])
    for name, value, ftype in (
        ("alt", 50.0, "real"), ("time", 1.0, "real"), ("dvbe", 200.0, "real"),
        ("psivgx", 90.0, "real"), ("thtvgx", -50.0, "real"),
        ("sbeg", np.zeros(3), "vec"), ("mguidance", 43, "int"),
        ("wp_lonx", 15.4, "real"), ("wp_latx", 35.3, "real"), ("wp_alt", 100.0, "real"),
        ("SWBG", swbg, "vec"), ("wp_flag", 0, "int"),
        ("mseeker", 0, "int"), ("range_go", 0.0, "real"),
        ("STBG", np.zeros(3), "vec"), ("closing_speed", 0.0, "real"),
        ("targ_com_slot", 0, "int"),
    ):
        if name not in v.store.names():
            v.store.define(Field(name, value, ftype, "data", "test"))
    ctx = _ctx()
    mod.execute(v, ctx)
    assert v.health == 0
    assert ctx.combus[0].status == 0
    assert v.store.get("miss") == pytest.approx(5.0)

def test_mguidance_30_wp_flag_minus1_does_not_kill():
    v = _vehicle()
    mod = Cruise5Intercept()
    mod.define(v)
    for name, value, ftype in (
        ("alt", 7000.0, "real"), ("time", 1.0, "real"), ("dvbe", 200.0, "real"),
        ("psivgx", 90.0, "real"), ("thtvgx", 0.0, "real"),
        ("sbeg", np.zeros(3), "vec"), ("mguidance", 30, "int"),
        ("wp_lonx", 14.9, "real"), ("wp_latx", 35.4, "real"), ("wp_alt", 0.0, "real"),
        ("SWBG", np.array([10.0, 0.0, 0.0]), "vec"), ("wp_flag", -1, "int"),
        ("mseeker", 0, "int"), ("range_go", 0.0, "real"),
        ("STBG", np.zeros(3), "vec"), ("closing_speed", 0.0, "real"),
        ("targ_com_slot", 0, "int"),
    ):
        if name not in v.store.names():
            v.store.define(Field(name, value, ftype, "data", "test"))
    ctx = _ctx()
    mod.execute(v, ctx)
    assert v.health == 1
    assert ctx.combus[0].status == 1
