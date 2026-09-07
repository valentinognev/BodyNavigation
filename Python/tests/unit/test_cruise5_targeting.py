import pathlib

import pytest

from cadac.constants import RAD
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.earth import cadine
from cadac.vehicles.cruise5.targeting import Cruise5Targeting

UAV_LONX, UAV_LATX, UAV_ALT = 14.7, 35.4, 7000.0
TGT_LONX, TGT_LATX, TGT_ALT = 15.4, 35.3, 100.0
SAT_LONX, SAT_LATX, SAT_ALT = 10.0, 30.0, 500000.0
FAR_TGT_LONX, FAR_TGT_LATX, FAR_TGT_ALT = 20.0, 40.0, 100.0
TIME = 0.0
WP_LONX_PRESET = 14.9

TARGETING_PY = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "cadac"
    / "vehicles"
    / "cruise5"
    / "targeting.py"
)

TARGETING_FIELDS = {
    "mtargeting": ("int", "data", ("scrn", "plot"), 0),
    "del_radius": ("real", "data", (), 0.0),
    "clost_tgt_slot": ("int", "out", (), 0),
    "tgtng_sat_slot": ("int", "out", (), 0),
}


def _sbii(lonx, latx, alt, time=TIME):
    return cadine(lonx * RAD, latx * RAD, alt, time)


def _uav_packet(name="UAV"):
    return Packet(name=name, type="CRUISE3", status=1, vars={})


def _target_packet(lonx, latx, alt, name="Tank"):
    return Packet(
        name=name,
        type="TARGET3",
        status=1,
        vars={
            "lonx": lonx,
            "latx": latx,
            "alt": alt,
            "sbii": _sbii(lonx, latx, alt),
        },
    )


def _sat_packet(lonx=SAT_LONX, latx=SAT_LATX, alt=SAT_ALT, name="Sat_s1"):
    return Packet(
        name=name,
        type="SATELLITE3",
        status=1,
        vars={"sbii": _sbii(lonx, latx, alt)},
    )


def _combus_one_each():
    return [
        _uav_packet(),
        _target_packet(TGT_LONX, TGT_LATX, TGT_ALT),
        _sat_packet(),
    ]


def _ctx(combus):
    return SimContext(
        sim_time=TIME,
        int_step=0.05,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=0,
    )


def _ready(mtargeting=1, combus=None, wp_lonx=WP_LONX_PRESET):
    vehicle = type("V", (), {"store": StateStore()})()
    targeting = Cruise5Targeting()
    targeting.define(vehicle)
    store = vehicle.store
    store.set("mtargeting", mtargeting)
    store.set("del_radius", 5000.0)
    store.define(Field("lonx", UAV_LONX, "real", "init/diag", "newton"))
    store.define(Field("latx", UAV_LATX, "real", "init/diag", "newton"))
    store.define(
        Field(
            "sbii",
            _sbii(UAV_LONX, UAV_LATX, UAV_ALT),
            "vec",
            "state",
            "newton",
        )
    )
    for name, value in (
        ("wp_lonx", wp_lonx),
        ("wp_latx", 35.4),
        ("wp_alt", 0.0),
    ):
        store.define(Field(name, value, "real", "data", "guidance"))
    if combus is None:
        combus = _combus_one_each()
    targeting.initialize(vehicle, _ctx(combus))
    return vehicle, targeting, _ctx(combus)


def test_mtargeting_0_does_not_write_waypoints():
    vehicle, targeting, ctx = _ready(mtargeting=0)
    targeting.execute(vehicle, ctx)
    store = vehicle.store
    assert store.get("wp_lonx") == WP_LONX_PRESET
    assert store.get("wp_latx") == 35.4
    assert store.get("wp_alt") == 0.0
    assert store.get("clost_tgt_slot") == 0
    assert store.get("tgtng_sat_slot") == 0


def test_mtargeting_2_raises():
    vehicle, targeting, ctx = _ready(mtargeting=2)
    with pytest.raises(ValueError, match="mtargeting"):
        targeting.execute(vehicle, ctx)
    store = vehicle.store
    assert store.get("wp_lonx") == WP_LONX_PRESET
    assert store.get("wp_latx") == 35.4
    assert store.get("wp_alt") == 0.0


def test_mtargeting_1_writes_tank_waypoint():
    vehicle, targeting, ctx = _ready(mtargeting=1)
    targeting.execute(vehicle, ctx)
    store = vehicle.store
    assert store.get("wp_lonx") == 15.4
    assert store.get("wp_latx") == 35.3
    assert store.get("wp_alt") == 100
    assert store.get("clost_tgt_slot") == 1
    assert store.get("tgtng_sat_slot") == 2


def test_closest_target_by_ground_range():
    combus = [
        _uav_packet(),
        _target_packet(FAR_TGT_LONX, FAR_TGT_LATX, FAR_TGT_ALT, name="Far"),
        _target_packet(TGT_LONX, TGT_LATX, TGT_ALT, name="Tank"),
        _sat_packet(),
    ]
    vehicle, targeting, ctx = _ready(mtargeting=1, combus=combus)
    targeting.execute(vehicle, ctx)
    store = vehicle.store
    assert store.get("wp_lonx") == 15.4
    assert store.get("wp_latx") == 35.3
    assert store.get("wp_alt") == 100
    assert store.get("clost_tgt_slot") == 2
    assert store.get("tgtng_sat_slot") == 3


def test_identifies_by_type_not_name_prefix():
    combus = [
        Packet(
            name="s1",
            type="CRUISE3",
            status=1,
            vars={"sbii": _sbii(SAT_LONX, SAT_LATX, SAT_ALT)},
        ),
        Packet(
            name="t1",
            type="CRUISE3",
            status=1,
            vars={
                "lonx": FAR_TGT_LONX,
                "latx": FAR_TGT_LATX,
                "alt": FAR_TGT_ALT,
                "sbii": _sbii(FAR_TGT_LONX, FAR_TGT_LATX, FAR_TGT_ALT),
            },
        ),
        Packet(
            name="alpha",
            type="TARGET3",
            status=1,
            vars={
                "lonx": TGT_LONX,
                "latx": TGT_LATX,
                "alt": TGT_ALT,
                "sbii": _sbii(TGT_LONX, TGT_LATX, TGT_ALT),
            },
        ),
        Packet(
            name="omega",
            type="SATELLITE3",
            status=1,
            vars={"sbii": _sbii(SAT_LONX, SAT_LATX, SAT_ALT)},
        ),
    ]
    vehicle, targeting, ctx = _ready(mtargeting=1, combus=combus)
    targeting.execute(vehicle, ctx)
    store = vehicle.store
    assert store.get("wp_lonx") == 15.4
    assert store.get("wp_latx") == 35.3
    assert store.get("wp_alt") == 100
    assert store.get("clost_tgt_slot") == 2
    assert store.get("tgtng_sat_slot") == 3


def test_name_is_targeting():
    assert Cruise5Targeting().name == "targeting"


def test_define_registers_def_targeting_fields():
    vehicle = type("V", (), {"store": StateStore()})()
    Cruise5Targeting().define(vehicle)
    store = vehicle.store
    for name, (ftype, role, outputs, default) in TARGETING_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "targeting"
        assert field.outputs == outputs
        assert store.get(name) == default
    assert type(store.get("mtargeting")) is int
    assert type(store.get("clost_tgt_slot")) is int
    assert type(store.get("tgtng_sat_slot")) is int
    assert "wp_lonx" not in store.names()
    assert "wp_latx" not in store.names()
    assert "wp_alt" not in store.names()


def test_big_is_module_level_not_in_constants():
    from cadac.vehicles.cruise5 import targeting as targeting_mod
    import cadac.constants as constants

    assert targeting_mod.BIG == 1e10
    assert not hasattr(constants, "BIG")


def test_no_hyper5_import():
    text = TARGETING_PY.read_text(encoding="utf-8")
    assert "hyper5" not in text
    assert "Hyper5" not in text
