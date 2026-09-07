from math import isfinite

import numpy as np
import pytest

from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.sraam6.intercept import Sraam6Intercept

INT_STEP = 0.5
TIME_CLOSE = 1.0
TIME_OPEN = 1.5

# Missile flies +x past a target 50 m ahead and 3 m east. LOS range-rate
# UTBL·VTBEL is negative while closing and positive after CPA (C++ fire).
SBEL_CLOSE = np.array([0.0, 0.0, -1000.0])
SBEL_OPEN = np.array([80.0, 0.0, -1000.0])
SBEL_GROUND = np.array([0.0, 0.0, 1.0])
STEL = np.array([50.0, 3.0, -1000.0])
VBEL = np.array([100.0, 0.0, 0.0])
VTEL = np.array([0.0, 0.0, 0.0])

DEFINED = (
    "mterm",
    "write",
    "miss",
    "hit_time",
    "MISS",
    "time_m",
    "SBTLM",
    "STMEL",
    "SBMEL",
    "mode",
    "psiptx",
    "thtptx",
    "critmax",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()
        self.health = 1


def _range_rate(stel, sbel, vtel, vbel):
    stbl = stel - sbel
    dbt = float(np.linalg.norm(stbl))
    utbl = stbl / dbt
    return float(utbl @ (vtel - vbel))


def _ctx(combus=None, int_step=INT_STEP, vehicle_slot=0):
    if combus is None:
        combus = [
            Packet(name="m1", type="MISSILE6", status=1, vars={}),
            Packet(name="t1", type="TARGET3", status=1, vars={}),
        ]
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=vehicle_slot,
    )


def _plant(
    store,
    *,
    time=TIME_CLOSE,
    halt=0,
    stop=0,
    lconv=0,
    sbel=None,
    vbel=None,
    stel=None,
    vtel=None,
    tgt_com_slot=1,
    mprop=0,
    trcond=0,
    mseek=0,
    mguid=6,
    maut=3,
):
    if sbel is None:
        sbel = SBEL_CLOSE
    if vbel is None:
        vbel = VBEL
    if stel is None:
        stel = STEL
    if vtel is None:
        vtel = VTEL
    store.define(Field("time", time, "real", "exec", "kinematics"))
    store.define(Field("halt", halt, "int", "data", "kinematics"))
    store.define(Field("stop", stop, "int", "data", "kinematics"))
    store.define(Field("lconv", lconv, "int", "diag", "kinematics"))
    store.define(Field("SBEL", sbel, "vec", "state", "newton"))
    store.define(Field("VBEL", vbel, "vec", "out", "newton"))
    store.define(Field("STEL", stel, "vec", "out", "combus"))
    store.define(Field("VTEL", vtel, "vec", "out", "combus"))
    store.define(Field("tgt_com_slot", tgt_com_slot, "int", "out", "combus"))
    store.define(Field("mprop", mprop, "int", "data", "propulsion"))
    store.define(Field("trcond", trcond, "int", "diag", "kinematics"))
    store.define(Field("mseek", mseek, "int", "data/diag", "seeker"))
    store.define(Field("mguid", mguid, "int", "data", "guidance"))
    store.define(Field("maut", maut, "int", "data", "control"))


def _ready(
    *,
    mterm=1,
    halt=0,
    stop=0,
    trcond=0,
    mseek=0,
    time=TIME_CLOSE,
    sbel=None,
    stel=None,
    vbel=None,
    vtel=None,
    combus=None,
    int_step=INT_STEP,
):
    vehicle = _Vehicle()
    intercept = Sraam6Intercept()
    intercept.define(vehicle)
    vehicle.store.set("mterm", mterm)
    _plant(
        vehicle.store,
        time=time,
        halt=halt,
        stop=stop,
        sbel=sbel,
        vbel=vbel,
        stel=stel,
        vtel=vtel,
        trcond=trcond,
        mseek=mseek,
    )
    return vehicle, intercept, _ctx(combus=combus, int_step=int_step)


def test_name_is_intercept():
    assert Sraam6Intercept().name == "intercept"


def test_write_default_is_one():
    vehicle = _Vehicle()
    Sraam6Intercept().define(vehicle)
    assert vehicle.store.get("write") == 1
    assert type(vehicle.store.get("write")) is int
    assert list(vehicle.store.names()) == list(DEFINED)
    assert vehicle.store.get("mterm") == 0
    assert vehicle.store.get("critmax") == 200.0


def test_halt_1_sets_health_and_packet_status_0():
    vehicle, intercept, ctx = _ready(halt=1)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert ctx.combus[1].status == 1
    assert vehicle.store.get("write") == 1


def test_sbel_down_positive_kills_missile():
    vehicle, intercept, ctx = _ready(sbel=SBEL_GROUND)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert ctx.combus[1].status == 1
    assert vehicle.store.get("write") == 0


def test_mterm_3_raises():
    vehicle, intercept, ctx = _ready(mterm=3)
    with pytest.raises(ValueError, match="mterm"):
        intercept.execute(vehicle, ctx)


def test_trcond_and_stop_kills_missile_only():
    vehicle, intercept, ctx = _ready(trcond=4, stop=1)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert ctx.combus[1].status == 1


def test_two_step_cpa_fires_on_opening_range_rate():
    assert _range_rate(STEL, SBEL_CLOSE, VTEL, VBEL) <= 0.0
    assert _range_rate(STEL, SBEL_OPEN, VTEL, VBEL) > 0.0
    assert float(np.linalg.norm(STEL - SBEL_CLOSE)) < 100.0
    assert float(np.linalg.norm(STEL - SBEL_OPEN)) < 100.0

    vehicle, intercept, ctx = _ready(mterm=1, mseek=4, time=TIME_CLOSE, sbel=SBEL_CLOSE)
    intercept.execute(vehicle, ctx)
    assert vehicle.store.get("write") == 1
    assert vehicle.health == 1
    assert ctx.combus[0].status == 1
    assert ctx.combus[1].status == 1

    vehicle.store.set("time", TIME_OPEN)
    vehicle.store.set("SBEL", SBEL_OPEN)
    intercept.execute(vehicle, ctx)
    assert ctx.combus[0].status == 0
    assert ctx.combus[1].status == 0
    assert vehicle.health == 0
    assert vehicle.store.get("write") == 0
    miss = vehicle.store.get("miss")
    assert isfinite(miss)
    assert miss >= 0.0
