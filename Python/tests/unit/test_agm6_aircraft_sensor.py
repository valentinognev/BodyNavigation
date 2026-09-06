from math import cos, sin
from types import SimpleNamespace

import numpy as np

from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import polar_from_cart
from cadac.vehicles.agm6.aircraft import Agm6Aircraft, Agm6AircraftSensor

RTOL = 1e-12
ATOL = 1e-14
ZEROS3 = (0.0, 0.0, 0.0)
COM = ("com",)
INT_STEP = 0.001
TRACK_STEP = 1.0

SAEL_ACFT = np.array([0.0, 0.0, -7000.0], dtype=float)
SAEL_TGT = np.array([33000.0, 10000.0, -100.0], dtype=float)
VTEL = np.array([0.0, -5.0, 0.0], dtype=float)
SAEL_TGT2 = np.array([1000.0, 2000.0, -50.0], dtype=float)
VTEL2 = np.array([1.0, 2.0, 3.0], dtype=float)
DECOY = np.array([1.0, 1.0, 1.0], dtype=float)
SBEL_TGT = np.array([33000.0, 10000.0, -100.0], dtype=float)
VBEL_TGT = np.array([0.0, -5.0, 0.0], dtype=float)

SENSOR_FIELDS = {
    "init_flag": ("int", "data", 1, ()),
    "track_epoch": ("real", "save", 0.0, ()),
    "track_step": ("real", "data", 0.0, ()),
    "target_num": ("int", "save", 0, ()),
    "STCEL1": ("vec", "out", ZEROS3, COM),
    "VTCEL1": ("vec", "out", ZEROS3, COM),
    "STCEL2": ("vec", "out", ZEROS3, COM),
    "VTCEL2": ("vec", "out", ZEROS3, COM),
    "STCEL3": ("vec", "out", ZEROS3, COM),
    "VTCEL3": ("vec", "out", ZEROS3, COM),
    "STCEL4": ("vec", "out", ZEROS3, ()),
    "VTCEL4": ("vec", "out", ZEROS3, ()),
    "STCEL5": ("vec", "out", ZEROS3, ()),
    "VTCEL5": ("vec", "out", ZEROS3, ()),
    "dat_sigma": ("real", "data", 0.0, ()),
    "azat_sigma": ("real", "data", 0.0, ()),
    "elat_sigma": ("real", "data", 0.0, ()),
    "vel_sigma": ("real", "data", 0.0, ()),
}


def _cart_from_pol(magnitude, azimuth, elevation):
    return np.array(
        [
            magnitude * (cos(elevation) * cos(azimuth)),
            magnitude * (cos(elevation) * sin(azimuth)),
            magnitude * (sin(elevation) * (-1.0)),
        ],
        dtype=float,
    )


def _expected(
    sael_acft,
    stel,
    vtel,
    dat_sigma=0.0,
    azat_sigma=0.0,
    elat_sigma=0.0,
    vel_sigma=0.0,
):
    satl = sael_acft - stel
    polar = polar_from_cart(satl)
    satcl = _cart_from_pol(
        float(polar[0]) + dat_sigma,
        float(polar[1]) + azat_sigma,
        float(polar[2]) + elat_sigma,
    )
    stcel = sael_acft - satcl
    vtcel = np.array(
        [vtel[0] + vel_sigma, vtel[1] + vel_sigma, vtel[2] + vel_sigma],
        dtype=float,
    )
    return stcel, vtcel


def _ctx(sim_time=0.0, combus=None):
    return SimContext(
        sim_time=sim_time,
        int_step=INT_STEP,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=0,
    )


def _packet(name, ptype, **vars_):
    return Packet(name=name, type=ptype, status=1, vars=dict(vars_))


def _ready(
    *,
    track_step=TRACK_STEP,
    sael=None,
    combus=None,
    dat_sigma=0.0,
    azat_sigma=0.0,
    elat_sigma=0.0,
    vel_sigma=0.0,
    sim_time=0.0,
):
    vehicle = SimpleNamespace(store=StateStore())
    sensor = Agm6AircraftSensor()
    sensor.define(vehicle)
    vehicle.store.define(Field("SAEL", ZEROS3, "vec", "state", "newton", ("com",)))
    vehicle.store.set("track_step", track_step)
    vehicle.store.set("SAEL", SAEL_ACFT if sael is None else sael)
    vehicle.store.set("dat_sigma", dat_sigma)
    vehicle.store.set("azat_sigma", azat_sigma)
    vehicle.store.set("elat_sigma", elat_sigma)
    vehicle.store.set("vel_sigma", vel_sigma)
    sensor.initialize(vehicle, _ctx(sim_time=sim_time, combus=combus or []))
    return vehicle, sensor, _ctx(sim_time=sim_time, combus=combus or [])


def test_name_is_sensor():
    assert Agm6AircraftSensor.name == "sensor"
    assert Agm6AircraftSensor().name == "sensor"


def test_define_com_on_track_files_1_to_3_only():
    vehicle = SimpleNamespace(store=StateStore())
    Agm6AircraftSensor().define(vehicle)
    store = vehicle.store
    assert "track_on" not in store.names()
    for name, (ftype, role, default, outputs) in SENSOR_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "sensor"
        assert field.outputs == outputs
        if ftype == "vec":
            np.testing.assert_array_equal(store.get(name), np.zeros(3))
        else:
            assert store.get(name) == default


def test_track_epoch_stcel1_equals_target_sael_vtcel1_equals_vtel():
    combus = [
        _packet("red", "TARGET3", SAEL=SAEL_TGT.copy(), VAEL=VTEL.copy()),
    ]
    vehicle, sensor, ctx = _ready(combus=combus)
    sensor.execute(vehicle, ctx)
    np.testing.assert_allclose(
        vehicle.store.get("STCEL1"), SAEL_TGT, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("VTCEL1"), VTEL, rtol=RTOL, atol=ATOL
    )


def test_target3_order_not_id_t1():
    combus = [
        _packet("t1", "MISSILE6", SAEL=DECOY.copy(), VAEL=DECOY.copy()),
        _packet("red", "TARGET3", SAEL=SAEL_TGT.copy(), VAEL=VTEL.copy()),
        _packet("t1", "TARGET3", SAEL=SAEL_TGT2.copy(), VAEL=VTEL2.copy()),
    ]
    vehicle, sensor, ctx = _ready(combus=combus)
    sensor.execute(vehicle, ctx)
    np.testing.assert_allclose(
        vehicle.store.get("STCEL1"), SAEL_TGT, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("VTCEL1"), VTEL, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("STCEL2"), SAEL_TGT2, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("VTCEL2"), VTEL2, rtol=RTOL, atol=ATOL
    )


def test_sbel_vbel_fallback():
    combus = [
        _packet("red", "TARGET3", SBEL=SBEL_TGT.copy(), VBEL=VBEL_TGT.copy()),
    ]
    vehicle, sensor, ctx = _ready(combus=combus)
    sensor.execute(vehicle, ctx)
    np.testing.assert_allclose(
        vehicle.store.get("STCEL1"), SBEL_TGT, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("VTCEL1"), VBEL_TGT, rtol=RTOL, atol=ATOL
    )


def test_max_five_target3_packets_in_appearance_order():
    targets = []
    for i in range(6):
        sael = np.array([float(i + 1) * 1000.0, 0.0, -100.0], dtype=float)
        vael = np.array([0.0, float(i + 1), 0.0], dtype=float)
        targets.append((sael, vael))
    combus = [_packet("m1", "MISSILE6", SAEL=DECOY.copy())]
    combus.extend(
        _packet(f"tgt{i}", "TARGET3", SAEL=sael.copy(), VAEL=vael.copy())
        for i, (sael, vael) in enumerate(targets)
    )
    vehicle, sensor, ctx = _ready(combus=combus)
    sensor.execute(vehicle, ctx)
    for i in range(5):
        want_stcel, want_vtcel = _expected(SAEL_ACFT, targets[i][0], targets[i][1])
        np.testing.assert_allclose(
            vehicle.store.get(f"STCEL{i + 1}"), want_stcel, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            vehicle.store.get(f"VTCEL{i + 1}"), want_vtcel, rtol=RTOL, atol=ATOL
        )


def test_between_track_epochs_does_not_update_files():
    combus = [
        _packet("red", "TARGET3", SAEL=SAEL_TGT.copy(), VAEL=VTEL.copy()),
    ]
    vehicle, sensor, ctx = _ready(combus=combus)
    sensor.execute(vehicle, ctx)
    moved = np.array([34000.0, 11000.0, -200.0], dtype=float)
    combus[0].vars["SAEL"] = moved
    later = _ctx(sim_time=0.5, combus=combus)
    sensor.execute(vehicle, later)
    np.testing.assert_allclose(
        vehicle.store.get("STCEL1"), SAEL_TGT, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("track_epoch"), TRACK_STEP, rtol=RTOL, atol=ATOL
    )


def test_stored_sigmas_added_to_polar_and_velocity():
    dat_sigma = 10.0
    azat_sigma = 0.01
    elat_sigma = -0.02
    vel_sigma = 1.5
    combus = [
        _packet("red", "TARGET3", SAEL=SAEL_TGT.copy(), VAEL=VTEL.copy()),
    ]
    vehicle, sensor, ctx = _ready(
        combus=combus,
        dat_sigma=dat_sigma,
        azat_sigma=azat_sigma,
        elat_sigma=elat_sigma,
        vel_sigma=vel_sigma,
    )
    sensor.execute(vehicle, ctx)
    want_stcel, want_vtcel = _expected(
        SAEL_ACFT,
        SAEL_TGT,
        VTEL,
        dat_sigma=dat_sigma,
        azat_sigma=azat_sigma,
        elat_sigma=elat_sigma,
        vel_sigma=vel_sigma,
    )
    np.testing.assert_allclose(
        vehicle.store.get("STCEL1"), want_stcel, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("VTCEL1"), want_vtcel, rtol=RTOL, atol=ATOL
    )


def test_vehicle_uses_same_sensor_class():
    vehicle = Agm6Aircraft("Blue")
    assert any(isinstance(module, Agm6AircraftSensor) for module in vehicle.modules)
    assert type(vehicle.modules[-1]) is Agm6AircraftSensor
    vehicle.define()
    vehicle.store.set("SAEL", SAEL_ACFT.copy())
    vehicle.store.set("track_step", TRACK_STEP)
    combus = [
        _packet("red", "TARGET3", SAEL=SAEL_TGT.copy(), VAEL=VTEL.copy()),
    ]
    ctx = _ctx(combus=combus)
    sensor = vehicle.modules[-1]
    sensor.execute(vehicle, ctx)
    np.testing.assert_allclose(
        vehicle.store.get("STCEL1"), SAEL_TGT, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("VTCEL1"), VTEL, rtol=RTOL, atol=ATOL
    )
    assert "com" in vehicle.store.field("STCEL1").outputs
    assert "com" in vehicle.store.field("STCEL3").outputs
    assert "com" not in vehicle.store.field("STCEL4").outputs
