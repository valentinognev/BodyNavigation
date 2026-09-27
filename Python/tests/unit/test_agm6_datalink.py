from types import SimpleNamespace

import numpy as np

from cadac.constants import EPS
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.agm6.datalink import Agm6Datalink

RTOL = 1e-12
ATOL = 1e-14
ZEROS3 = (0.0, 0.0, 0.0)

# C++ Missile::def_datalink order (empty last-arg → no plot/scrn).
FIELDS = {
    "mnav": ("int", "out", 0, ()),
    "STCEL": ("vec", "out", ZEROS3, ()),
    "VTCEL": ("vec", "out", ZEROS3, ()),
    "tgt_pos": ("real", "save", 0.0, ()),
    "SAEL": ("vec", "out", ZEROS3, ()),
    "VAEL": ("vec", "out", ZEROS3, ()),
}
DEFINED = tuple(FIELDS)
EXTERNALS = ("tgt_num", "time")

STCEL1 = np.array([33000.0, 10000.0, -100.0], dtype=float)
VTCEL1 = np.array([0.0, -5.0, 0.0], dtype=float)
SAEL = np.array([0.0, 0.0, -7000.0], dtype=float)
VAEL = np.array([250.0, 0.0, 0.0], dtype=float)
STCEL1_MOVED = np.array([33100.0, 10000.0, -100.0], dtype=float)
STCEL2 = np.array([1000.0, 2000.0, -50.0], dtype=float)
VTCEL2 = np.array([1.0, 2.0, 3.0], dtype=float)
DECOY_STCEL = np.array([1.0, 1.0, 1.0], dtype=float)
TGT_POS1 = float(np.linalg.norm(STCEL1))
TGT_POS_MOVED = float(np.linalg.norm(STCEL1_MOVED))


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


def _combus(aircraft_vars=None):
    if aircraft_vars is None:
        aircraft_vars = {
            "STCEL1": STCEL1.copy(),
            "VTCEL1": VTCEL1.copy(),
            "STCEL2": STCEL2.copy(),
            "VTCEL2": VTCEL2.copy(),
            "SAEL": SAEL.copy(),
            "VAEL": VAEL.copy(),
        }
    missile = _packet("a1", "MISSILE6")
    target = _packet("t1", "TARGET3", STCEL1=DECOY_STCEL.copy(), SAEL=STCEL1.copy())
    aircraft = _packet("blue", "AIRCRAFT3", **aircraft_vars)
    return [missile, target, aircraft]


def _ready(*, tgt_num=1, combus=None):
    vehicle = SimpleNamespace(store=StateStore())
    link = Agm6Datalink()
    link.define(vehicle)
    vehicle.store.define(Field("tgt_num", tgt_num, "int", "data", "sensor"))
    link.initialize(vehicle, _ctx(combus or [], vehicle_slot=0))
    return vehicle, link


def test_name_is_datalink():
    assert Agm6Datalink.name == "datalink"
    assert Agm6Datalink().name == "datalink"


def test_define_registers_cpp_def_datalink_fields():
    vehicle = SimpleNamespace(store=StateStore())
    Agm6Datalink().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    zeros3 = np.zeros(3)
    for name, (ftype, role, default, outputs) in FIELDS.items():
        field = store.field(name)
        assert field.module == "datalink"
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
    for name in EXTERNALS:
        assert name not in store.names()


def test_named_aircraft_track_mnav_on_position_change():
    combus = _combus()
    vehicle, link = _ready(tgt_num=1, combus=combus)
    store = vehicle.store
    ctx = _ctx(combus)

    link.execute(vehicle, ctx)
    np.testing.assert_allclose(store.get("STCEL"), STCEL1, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VTCEL"), VTCEL1, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SAEL"), SAEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VAEL"), VAEL, rtol=RTOL, atol=ATOL)
    assert store.get("mnav") == 3
    assert type(store.get("mnav")) is int
    np.testing.assert_allclose(store.get("tgt_pos"), TGT_POS1, rtol=RTOL, atol=ATOL)
    assert abs(TGT_POS1 - 0.0) > EPS

    link.execute(vehicle, ctx)
    np.testing.assert_allclose(store.get("STCEL"), STCEL1, rtol=RTOL, atol=ATOL)
    assert store.get("mnav") == 0
    np.testing.assert_allclose(store.get("tgt_pos"), TGT_POS1, rtol=RTOL, atol=ATOL)

    combus[2].vars["STCEL1"] = STCEL1_MOVED.copy()
    link.execute(vehicle, ctx)
    np.testing.assert_allclose(store.get("STCEL"), STCEL1_MOVED, rtol=RTOL, atol=ATOL)
    assert store.get("mnav") == 3
    np.testing.assert_allclose(store.get("tgt_pos"), TGT_POS_MOVED, rtol=RTOL, atol=ATOL)
    assert abs(TGT_POS_MOVED - TGT_POS1) > EPS


def test_no_aircraft3_leaves_stcel_zeros_and_mnav_0():
    combus = [
        _packet("a1", "MISSILE6"),
        _packet("t1", "TARGET3", STCEL1=DECOY_STCEL.copy()),
    ]
    vehicle, link = _ready(tgt_num=1, combus=combus)
    store = vehicle.store
    link.execute(vehicle, _ctx(combus))
    np.testing.assert_array_equal(store.get("STCEL"), np.zeros(3))
    np.testing.assert_array_equal(store.get("VTCEL"), np.zeros(3))
    np.testing.assert_array_equal(store.get("SAEL"), np.zeros(3))
    np.testing.assert_array_equal(store.get("VAEL"), np.zeros(3))
    assert store.get("mnav") == 0
    assert store.get("tgt_pos") == 0.0


def test_tgt_num_selects_named_track_on_aircraft_packet():
    combus = _combus()
    vehicle, link = _ready(tgt_num=2, combus=combus)
    store = vehicle.store
    link.execute(vehicle, _ctx(combus))
    np.testing.assert_allclose(store.get("STCEL"), STCEL2, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VTCEL"), VTCEL2, rtol=RTOL, atol=ATOL)
    assert store.get("mnav") == 3


def test_first_aircraft3_packet_wins():
    first = _packet(
        "lead",
        "AIRCRAFT3",
        STCEL1=STCEL1.copy(),
        VTCEL1=VTCEL1.copy(),
        SAEL=SAEL.copy(),
        VAEL=VAEL.copy(),
    )
    second = _packet(
        "trail",
        "AIRCRAFT3",
        STCEL1=STCEL1_MOVED.copy(),
        VTCEL1=VTCEL2.copy(),
        SAEL=np.array([9.0, 9.0, 9.0], dtype=float),
        VAEL=np.array([9.0, 9.0, 9.0], dtype=float),
    )
    combus = [_packet("m", "MISSILE6"), first, second]
    vehicle, link = _ready(tgt_num=1, combus=combus)
    store = vehicle.store
    link.execute(vehicle, _ctx(combus))
    np.testing.assert_allclose(store.get("STCEL"), STCEL1, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VTCEL"), VTCEL1, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SAEL"), SAEL, rtol=RTOL, atol=ATOL)
