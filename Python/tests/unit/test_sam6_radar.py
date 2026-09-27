import numpy as np
import pytest

from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck, Table
from cadac.vehicles.flat6.sam6.radar import Sam6Radar, Sam6RadarSensor

RTOL = 1e-12
ATOL = 1e-14
SMALL = 1e-7
ZEROS3 = (0.0, 0.0, 0.0)
LETHAL_RNG = 20e3
FAR_SAEL = np.array([0.0, -3e4, -1e4], dtype=float)
CLOSE_SAEL = np.array([0.0, -1e4, 0.0], dtype=float)

SENSOR_DEFINED = (
    "mtrack",
    "alt_engage",
    "init_flag",
    "track_epoch",
    "track_step",
    "rocket_num",
    "launch_delay",
    "SIEL",
    "ip_alt_bias",
    "lnch_dly_bias1",
    "lnch_dly_bias2",
    "lnch_dly_bias3",
    "dat_sigma",
    "azat_sigma",
    "elat_sigma",
    "vel_sigma",
    "apo_flag",
    "apo_epoch",
    "lnch_delay_m1",
    "lnch_delay_m2",
    "lnch_delay_m3",
    "SIEL1",
    "SIEL2",
    "SIEL3",
    "aircraft_num",
    "lethal_rng",
    "lethal_flag1",
    "lethal_flag2",
    "lethal_flag3",
    "launch_delay1",
    "launch_delay2",
    "launch_delay3",
)
SENSOR_ROLES = {
    "mtrack": "data",
    "alt_engage": "data",
    "init_flag": "init",
    "track_epoch": "save",
    "track_step": "data",
    "rocket_num": "save",
    "launch_delay": "save",
    "SIEL": "save",
    "ip_alt_bias": "data",
    "lnch_dly_bias1": "data",
    "lnch_dly_bias2": "data",
    "lnch_dly_bias3": "data",
    "dat_sigma": "data",
    "azat_sigma": "data",
    "elat_sigma": "data",
    "vel_sigma": "data",
    "apo_flag": "save",
    "apo_epoch": "save",
    "lnch_delay_m1": "out",
    "lnch_delay_m2": "out",
    "lnch_delay_m3": "out",
    "SIEL1": "out",
    "SIEL2": "out",
    "SIEL3": "out",
    "aircraft_num": "save",
    "lethal_rng": "data",
    "lethal_flag1": "save",
    "lethal_flag2": "save",
    "lethal_flag3": "save",
    "launch_delay1": "save",
    "launch_delay2": "save",
    "launch_delay3": "save",
}
SENSOR_OUTPUTS = {
    "mtrack": (),
    "alt_engage": (),
    "init_flag": (),
    "track_epoch": (),
    "track_step": (),
    "rocket_num": (),
    "launch_delay": (),
    "SIEL": (),
    "ip_alt_bias": (),
    "lnch_dly_bias1": (),
    "lnch_dly_bias2": (),
    "lnch_dly_bias3": (),
    "dat_sigma": (),
    "azat_sigma": (),
    "elat_sigma": (),
    "vel_sigma": (),
    "apo_flag": (),
    "apo_epoch": (),
    "lnch_delay_m1": ("com",),
    "lnch_delay_m2": ("com",),
    "lnch_delay_m3": ("com",),
    "SIEL1": ("com",),
    "SIEL2": ("com",),
    "SIEL3": ("com",),
    "aircraft_num": (),
    "lethal_rng": (),
    "lethal_flag1": (),
    "lethal_flag2": (),
    "lethal_flag3": (),
    "launch_delay1": (),
    "launch_delay2": (),
    "launch_delay3": (),
}
SENSOR_INT = (
    "mtrack",
    "init_flag",
    "rocket_num",
    "apo_flag",
    "aircraft_num",
    "lethal_flag1",
    "lethal_flag2",
    "lethal_flag3",
)
SENSOR_VEC = ("SIEL", "SIEL1", "SIEL2", "SIEL3")
NOT_DEFINED_BY_SENSOR = (
    "time",
    "launch_epoch",
    "launch_time",
    "srel1",
    "srel2",
    "srel3",
    "SREL",
)


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ctx(sim_time=0.0, combus=None, vehicle_slot=0, int_step=0.01):
    return SimContext(
        sim_time=sim_time,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=combus or [],
        vehicle_slot=vehicle_slot,
    )


def _packet(name, type_name, sael=None, vael=None, sbel=None):
    vars_ = {}
    if sael is not None:
        vars_["SAEL"] = np.asarray(sael, dtype=float)
    if vael is not None:
        vars_["VAEL"] = np.asarray(vael, dtype=float)
    if sbel is not None:
        vars_["SBEL"] = np.asarray(sbel, dtype=float)
    return Packet(name=name, type=type_name, status=1, vars=vars_)


def _sensor(vehicle):
    return next(module for module in vehicle.modules if module.name == "sensor")


def _table(name, x1, values):
    return Table(
        name=name,
        dim=1,
        x1=np.asarray(x1, dtype=float),
        x2=None,
        x3=None,
        values=np.asarray(values, dtype=float),
    )


def _traj_decks(apo_time=10.0, missile_time=4.0):
    srmb = Datadeck.from_tables(
        [
            _table("apotime_vs_descent_altitude", [0.0, 20000.0], [apo_time, apo_time]),
            _table("x_vs_launch_time", [0.0, 100.0], [1000.0, 1000.0]),
            _table("y_vs_launch_time", [0.0, 100.0], [2000.0, 2000.0]),
            _table("z_vs_launch_time", [0.0, 100.0], [-5000.0, -5000.0]),
        ]
    )
    sam = Datadeck.from_tables(
        [
            _table("time_vs_ascent_altitude", [0.0, 20000.0], [missile_time, missile_time]),
            _table("alt_vs_launch_time", [0.0, 100.0], [0.0, 5000.0]),
        ]
    )
    return sam, srmb


def _defined_radar(**kwargs):
    radar = Sam6Radar("f1", **kwargs)
    radar.define()
    return radar


def test_type_health_and_modules():
    radar = Sam6Radar("f1")
    assert radar.type == "RADAR0"
    assert radar.health == 1
    names = [module.name for module in radar.modules]
    assert "kinematics" in names
    assert "newton" in names
    assert "sensor" in names
    assert names.count("sensor") == 1


def test_define_launch_delay_defaults_are_sensor_not_kinematics():
    radar = _defined_radar()
    assert _approx(radar.store.get("lnch_delay_m1"), 0.0)
    assert _approx(radar.store.get("lnch_delay_m2"), 0.0)
    assert _approx(radar.store.get("lnch_delay_m3"), 0.0)
    assert _approx(radar.store.get("launch_delay"), 9999.0)
    assert _approx(radar.store.get("launch_delay1"), 9999.0)
    assert _approx(radar.store.get("launch_delay2"), 9999.0)
    assert _approx(radar.store.get("launch_delay3"), 9999.0)
    assert radar.store.field("launch_delay").role == "save"
    assert radar.store.field("launch_delay").module == "sensor"


def test_define_skips_existing_field():
    radar = Sam6Radar("f1")
    radar.store.define(Field("lnch_delay_m1", 42.0, "real", "out", "pre", ("com",)))
    radar.define()
    assert _approx(radar.store.get("lnch_delay_m1"), 42.0)
    assert radar.store.field("lnch_delay_m1").module == "pre"


def test_com_names_include_radar_com_fields():
    radar = _defined_radar()
    for name in (
        "time",
        "lnch_delay_m1",
        "lnch_delay_m2",
        "lnch_delay_m3",
        "SIEL1",
        "SIEL2",
        "SIEL3",
    ):
        assert name in radar.com_names


def test_sensor_define_registers_cpp_fields():
    class _Vehicle:
        def __init__(self):
            self.store = StateStore()

    vehicle = _Vehicle()
    Sam6RadarSensor().define(vehicle)
    names = vehicle.store.names()
    for name in SENSOR_DEFINED:
        assert name in names
        field = vehicle.store.field(name)
        assert field.role == SENSOR_ROLES[name], name
        assert field.module == "sensor", name
        assert field.outputs == SENSOR_OUTPUTS[name], name
        if name in SENSOR_INT:
            assert field.type == "int", name
        elif name in SENSOR_VEC:
            assert field.type == "vec", name
        else:
            assert field.type == "real", name
    for name in NOT_DEFINED_BY_SENSOR:
        assert name not in names
    assert vehicle.store.get("init_flag") == 1
    assert _approx(vehicle.store.get("launch_delay"), 9999.0)
    assert _approx(vehicle.store.get("launch_delay1"), 9999.0)


def test_mtrack_0_no_raise_delays_unchanged():
    radar = _defined_radar()
    radar.store.set("mtrack", 0)
    _sensor(radar).execute(radar, _ctx(sim_time=1.0, combus=[]))
    assert _approx(radar.store.get("lnch_delay_m1"), 0.0)
    assert _approx(radar.store.get("launch_delay1"), 9999.0)
    assert _approx(radar.store.get("launch_delay"), 9999.0)


def test_mtrack_2_far_aircraft_keeps_delay_defaults():
    radar = _defined_radar()
    radar.store.set("mtrack", 2)
    radar.store.set("lethal_rng", LETHAL_RNG)
    combus = [
        _packet("a1", "AIRCRAFT3", sael=FAR_SAEL, vael=[0.0, 200.0, 0.0]),
    ]
    _sensor(radar).execute(radar, _ctx(sim_time=1.0, combus=combus))
    assert _approx(radar.store.get("lnch_delay_m1"), 0.0)
    assert _approx(radar.store.get("launch_delay1"), 9999.0)


def test_mtrack_2_lethal_aircraft_sets_launch_delay1_and_com():
    radar = _defined_radar()
    bias = 0.25
    sim_time = 2.0
    radar.store.set("mtrack", 2)
    radar.store.set("lethal_rng", LETHAL_RNG)
    radar.store.set("lnch_dly_bias1", bias)
    combus = [
        _packet("a1", "AIRCRAFT3", sael=CLOSE_SAEL, vael=[0.0, 200.0, 0.0]),
    ]
    _sensor(radar).execute(radar, _ctx(sim_time=sim_time, combus=combus))
    assert _approx(radar.store.get("launch_delay1"), sim_time)
    assert _approx(
        radar.store.get("lnch_delay_m1"),
        radar.store.get("launch_delay1") + bias,
    )


def test_mtrack_2_lethal_flag_latches_launch_delay1():
    radar = _defined_radar()
    radar.store.set("mtrack", 2)
    radar.store.set("lethal_rng", LETHAL_RNG)
    radar.store.set("track_step", 0.0)
    close = [_packet("a1", "AIRCRAFT3", sael=CLOSE_SAEL, vael=[0.0, 0.0, 0.0])]
    _sensor(radar).execute(radar, _ctx(sim_time=1.5, combus=close))
    far = [_packet("a1", "AIRCRAFT3", sael=FAR_SAEL, vael=[0.0, 0.0, 0.0])]
    _sensor(radar).execute(radar, _ctx(sim_time=3.0, combus=far))
    assert _approx(radar.store.get("launch_delay1"), 1.5)
    assert radar.store.get("lethal_flag1") == 1


def test_mtrack_1_after_vtcel_z_positive_uses_traj_tables():
    apo_time = 10.0
    missile_time = 4.0
    bias = 0.2
    sam, srmb = _traj_decks(apo_time=apo_time, missile_time=missile_time)
    radar = _defined_radar(sam_deck=sam, srmb_deck=srmb)
    radar.store.set("mtrack", 1)
    radar.store.set("lnch_dly_bias1", bias)
    radar.store.set("track_step", 0.0)
    sael = np.array([0.0, 0.0, -8000.0], dtype=float)
    climb = [
        _packet("r1", "ROCKET5", sael=sael, vael=[100.0, 0.0, -50.0]),
    ]
    _sensor(radar).execute(radar, _ctx(sim_time=1.0, combus=climb))
    assert _approx(radar.store.get("launch_delay"), 9999.0)
    assert radar.store.get("apo_flag") == 0
    descent = [
        _packet("r1", "ROCKET5", sael=sael, vael=[100.0, 0.0, 40.0]),
    ]
    sim_time = 3.0
    _sensor(radar).execute(radar, _ctx(sim_time=sim_time, combus=descent))
    expected_launch = sim_time + apo_time - missile_time
    assert _approx(radar.store.get("launch_delay"), expected_launch)
    assert _approx(radar.store.get("lnch_delay_m1"), expected_launch + bias)
    assert radar.store.get("lnch_delay_m1") not in (0.0, 9999.0)
    assert radar.store.get("apo_flag") == 1


def test_mtrack_4_raises():
    radar = _defined_radar()
    radar.store.set("mtrack", 4)
    with pytest.raises(ValueError, match="mtrack"):
        _sensor(radar).execute(radar, _ctx(sim_time=0.0, combus=[]))


def test_sensor_name_and_small():
    assert Sam6RadarSensor.name == "sensor"
    from cadac.vehicles.flat6.sam6 import radar as radar_mod

    assert radar_mod.SMALL == SMALL
