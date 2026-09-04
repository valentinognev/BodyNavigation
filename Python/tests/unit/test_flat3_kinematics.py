from types import SimpleNamespace

from cadac.eom.flat3 import Flat3Kinematics
from cadac.kernel.state import StateStore


def test_name_is_kinematics():
    assert Flat3Kinematics().name == "kinematics"


def test_time_copy():
    s = StateStore()
    k = Flat3Kinematics()
    k.define(SimpleNamespace(store=s))
    ctx = SimpleNamespace(sim_time=1.5, event_time=0.2, int_step=0.05)
    k.execute(SimpleNamespace(store=s), ctx)
    assert s.get("time") == 1.5
    assert s.get("event_time") == 0.2
