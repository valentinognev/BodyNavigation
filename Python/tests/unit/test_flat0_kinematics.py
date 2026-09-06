import numpy as np
from cadac.eom.flat0 import Flat0Kinematics
from cadac.kernel.state import StateStore
from cadac.kernel.executive import SimContext


class V:
    def __init__(self):
        self.store = StateStore()


def test_name_is_kinematics():
    assert Flat0Kinematics().name == "kinematics"


def test_flat0_kinematics_defines_cpp_fields():
    v = V()
    k = Flat0Kinematics()
    k.define(v)
    time = v.store.field("time")
    assert time.type == "real"
    assert time.role == "out"
    assert time.module == "kinematics"
    assert time.outputs == ("com",)
    delay = v.store.field("launch_delay")
    assert delay.type == "real"
    assert delay.role == "data"
    assert delay.module == "kinematics"
    epoch = v.store.field("launch_epoch")
    assert epoch.type == "real"
    assert epoch.role == "init"
    assert epoch.module == "kinematics"
    elapsed = v.store.field("launch_time")
    assert elapsed.type == "real"
    assert elapsed.role == "out"
    assert elapsed.module == "kinematics"


def test_flat0_launch_clock():
    v = V()
    k = Flat0Kinematics()
    k.define(v)
    v.store.set("launch_delay", 12.0)
    ctx = SimContext(sim_time=0.0, int_step=0.01, event_time=0.0, out_fact=0.0, combus=[], vehicle_slot=0)
    k.initialize(v, ctx)
    ctx.sim_time = 15.0
    k.execute(v, ctx)
    assert v.store.get("time") == 15.0
    np.testing.assert_allclose(v.store.get("launch_time"), 3.0, rtol=1e-12, atol=1e-14)
