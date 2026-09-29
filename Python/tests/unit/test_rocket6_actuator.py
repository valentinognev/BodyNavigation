from types import SimpleNamespace

import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round6.rocket6.actuator import Rocket6Actuator

DT = 0.001


def _ctx(int_step=DT):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _rocket6(
    *,
    delacx=0.0,
    delecx=0.0,
    delrcx=0.0,
):
    vehicle = SimpleNamespace(store=StateStore(), modules=[])
    act = Rocket6Actuator()
    act.define(vehicle)
    act.initialize(vehicle, _ctx())
    store = vehicle.store
    store.define(Field("delacx", delacx, "real", "out", "control"))
    store.define(Field("delecx", delecx, "real", "out", "control"))
    store.define(Field("delrcx", delrcx, "real", "out", "control"))
    vehicle.modules = [act]
    return vehicle


def _exec(vehicle, name):
    module = next(m for m in vehicle.modules if m.name == name)
    module.execute(vehicle, _ctx())


def test_rocket6_mact_02_four_fin_mix_position_limit():
    veh = _rocket6()
    veh.store.set("mact", 2)  # morder=0, mvehicle=2
    veh.store.set("dlimx", 20.0)
    veh.store.set("delacx", 5.0)
    veh.store.set("delecx", 2.0)
    veh.store.set("delrcx", 1.0)
    _exec(veh, "actuator")
    # Unlimited 4-fin mix recovers control deflections exactly:
    # delcx1=-4, delcx2=-2, delcx3=6, delcx4=8 → delax=5, delex=2, delrx=1
    assert abs(veh.store.get("delax") - 5.0) < 1e-9
    assert abs(veh.store.get("delex") - 2.0) < 1e-9
    assert abs(veh.store.get("delrx") - 1.0) < 1e-9


def test_rocket6_mact_22_second_order_rate_limit_path():
    veh = _rocket6()
    veh.store.set("mact", 22)
    veh.store.set("dlimx", 20.0)
    veh.store.set("ddlimx", 100.0)
    veh.store.set("wnact", 50.0)
    veh.store.set("zetact", 0.7)
    veh.store.set("delacx", 1.0)
    veh.store.set("delecx", 0.0)
    veh.store.set("delrcx", 0.0)
    _exec(veh, "actuator")
    assert "delax" in veh.store
