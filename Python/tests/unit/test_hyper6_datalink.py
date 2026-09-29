"""HYPER6 datalink (mnav / STCII), source-faithful to Hyper::datalink."""

from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import EPS
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round6.hyper6.vehicle import Hyper6

RTOL = 1e-12
ATOL = 1e-14
ZEROS3 = (0.0, 0.0, 0.0)

# C++ Hyper::def_datalink order (sat_num module=combus; rest datalink).
FIELDS = {
    "sat_num": ("int", "data", 0, (), "combus"),
    "mnav": ("int", "out", 0, (), "datalink"),
    "STCII": ("vec", "out", ZEROS3, (), "datalink"),
    "VTCII": ("vec", "out", ZEROS3, (), "datalink"),
    "tgt_pos": ("real", "save", 0.0, (), "datalink"),
}
DEFINED = tuple(FIELDS)

STCII1 = np.array([7005000.0, 98000.0, 201000.0], dtype=float)
VTCII1 = np.array([10.0, 7500.0, -5.0], dtype=float)
STCII1_MOVED = np.array([7005100.0, 98000.0, 201000.0], dtype=float)
STCII2 = np.array([6.8e6, 1.0e5, 2.0e5], dtype=float)
VTCII2 = np.array([1.0, 2.0, 3.0], dtype=float)
TGT_POS1 = float(np.linalg.norm(STCII1))
TGT_POS_MOVED = float(np.linalg.norm(STCII1_MOVED))


def _ctx(combus, vehicle_slot=0):
    return SimContext(
        sim_time=0.0,
        int_step=0.001,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=vehicle_slot,
    )


def _packet(name, ptype, **vars_):
    return Packet(name=name, type=ptype, status=1, vars=dict(vars_))


def _radar_vars(**overrides):
    vars_ = {
        "sbii": np.zeros(3),
        "vbii": np.zeros(3),
        "dbi": 0.0,
        "stcii1": STCII1.copy(),
        "vtcii1": VTCII1.copy(),
        "stcii2": STCII2.copy(),
        "vtcii2": VTCII2.copy(),
        "stcii3": np.zeros(3),
        "vtcii3": np.zeros(3),
        "stcii4": np.zeros(3),
        "vtcii4": np.zeros(3),
        "stcii5": np.zeros(3),
        "vtcii5": np.zeros(3),
    }
    vars_.update(overrides)
    return vars_


def _combus(radar_vars=None):
    radar = _packet("r1", "RADAR0", **(radar_vars if radar_vars is not None else _radar_vars()))
    return [
        _packet("h1", "HYPER6"),
        _packet("t1", "SAT3", sbii=STCII1.copy(), vbii=VTCII1.copy()),
        radar,
    ]


def _datalink_module():
    from cadac.vehicles.round6.hyper6.datalink import Hyper6Datalink

    return Hyper6Datalink()


def _ready(*, sat_num=1, combus=None):
    vehicle = SimpleNamespace(store=StateStore())
    link = _datalink_module()
    link.define(vehicle)
    vehicle.store.set("sat_num", sat_num)
    link.initialize(vehicle, _ctx(combus or [], vehicle_slot=0))
    return vehicle, link


def test_hyper6_has_datalink_module():
    # Break: datalink missing / wrong Cape order (after ins, before seeker).
    vehicle = Hyper6("Hypersonic", None, None)
    names = [module.name for module in vehicle.modules]
    assert "datalink" in names
    assert names.index("datalink") > names.index("ins")
    assert names.index("datalink") < names.index("seeker")
    assert names.index("datalink") < names.index("guidance")
    link = next(m for m in vehicle.modules if m.name == "datalink")
    assert type(link).__name__ == "Hyper6Datalink"


def test_define_registers_cpp_def_datalink_fields():
    vehicle = SimpleNamespace(store=StateStore())
    _datalink_module().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    zeros3 = np.zeros(3)
    for name, (ftype, role, default, outputs, module) in FIELDS.items():
        field = store.field(name)
        assert field.module == module
        assert field.type == ftype
        assert field.role == role
        assert field.outputs == outputs
        if ftype == "int":
            assert store.get(name) == default
            assert type(store.get(name)) is int
        elif ftype == "real":
            assert store.get(name) == default
        else:
            np.testing.assert_array_equal(store.get(name), zeros3)
            assert store.get(name).shape == (3,)


def test_hyper6_datalink_sets_mnav_on_track_change():
    # Break: execute does not write STCII / mnav=3 on radar track change.
    combus = _combus()
    vehicle, link = _ready(sat_num=1, combus=combus)
    store = vehicle.store
    ctx = _ctx(combus)

    link.execute(vehicle, ctx)
    np.testing.assert_allclose(store.get("STCII"), STCII1, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VTCII"), VTCII1, rtol=RTOL, atol=ATOL)
    assert store.get("mnav") == 3
    assert type(store.get("mnav")) is int
    np.testing.assert_allclose(store.get("tgt_pos"), TGT_POS1, rtol=RTOL, atol=ATOL)
    assert abs(TGT_POS1 - 0.0) > EPS

    link.execute(vehicle, ctx)
    np.testing.assert_allclose(store.get("STCII"), STCII1, rtol=RTOL, atol=ATOL)
    assert store.get("mnav") == 0
    np.testing.assert_allclose(store.get("tgt_pos"), TGT_POS1, rtol=RTOL, atol=ATOL)

    combus[2].vars["stcii1"] = STCII1_MOVED.copy()
    link.execute(vehicle, ctx)
    np.testing.assert_allclose(store.get("STCII"), STCII1_MOVED, rtol=RTOL, atol=ATOL)
    assert store.get("mnav") == 3
    np.testing.assert_allclose(store.get("tgt_pos"), TGT_POS_MOVED, rtol=RTOL, atol=ATOL)
    assert abs(TGT_POS_MOVED - TGT_POS1) > EPS


def test_no_r1_leaves_stcii_zeros_and_mnav_0():
    combus = [
        _packet("h1", "HYPER6"),
        _packet("t1", "SAT3", sbii=STCII1.copy()),
        _packet("r2", "RADAR0", **_radar_vars()),
    ]
    vehicle, link = _ready(sat_num=1, combus=combus)
    store = vehicle.store
    link.execute(vehicle, _ctx(combus))
    np.testing.assert_array_equal(store.get("STCII"), np.zeros(3))
    np.testing.assert_array_equal(store.get("VTCII"), np.zeros(3))
    assert store.get("mnav") == 0
    assert store.get("tgt_pos") == 0.0


def test_sat_num_selects_named_track_on_radar_packet():
    combus = _combus()
    vehicle, link = _ready(sat_num=2, combus=combus)
    store = vehicle.store
    link.execute(vehicle, _ctx(combus))
    np.testing.assert_allclose(store.get("STCII"), STCII2, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VTCII"), VTCII2, rtol=RTOL, atol=ATOL)
    assert store.get("mnav") == 3


def test_seeker_no_longer_defines_sat_num():
    # Break: sat_num still owned by seeker after datalink handoff.
    from cadac.vehicles.round6.hyper6.seeker import Hyper6Seeker

    vehicle = SimpleNamespace(store=StateStore())
    Hyper6Seeker().define(vehicle)
    assert "sat_num" not in vehicle.store.names()
    assert "STII" in vehicle.store.names()
    assert "VTII" in vehicle.store.names()
