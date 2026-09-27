import pathlib

import pytest

from cadac.constants import RAD
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.earth import cadine
from cadac.vehicles.round3.hyper5.targeting import Hyper5Targeting

RTOL = 1e-12
ATOL = 1e-14

# Demo 4.7 Hyper / Target; satellite overhead of Hyper so LOS is clear.
HYPER_LONX = -106.28
HYPER_LATX = 33.35
HYPER_ALT = 2400.0
TGT_LONX = -106.28
TGT_LATX = 33.4
TGT_ALT = 1200.0
FAR_TGT_LONX = -106.28
FAR_TGT_LATX = 34.0
FAR_TGT_ALT = 1200.0
SAT_LONX = -106.28
SAT_LATX = 33.35
SAT_ALT = 500000.0
TIME = 0.0

TARGETING_PY = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "cadac"
    / "vehicles"
    / "round3"
    / "hyper5"
    / "targeting.py"
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _sbii(lonx, latx, alt, time=TIME):
    return cadine(lonx * RAD, latx * RAD, alt, time)


def _hyper_packet():
    return Packet(name="RR3X", type="HYPER5", status=1, vars={})


def _target_packet(lonx, latx, alt, name="Truck"):
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


def _sat_packet(lonx=SAT_LONX, latx=SAT_LATX, alt=SAT_ALT, name="Satellite_s1"):
    return Packet(
        name=name,
        type="SATELLITE3",
        status=1,
        vars={"sbii": _sbii(lonx, latx, alt)},
    )


def _combus_one_each():
    return [
        _hyper_packet(),
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


def _ready(mtargeting=1, combus=None):
    vehicle = _Vehicle()
    targeting = Hyper5Targeting()
    targeting.define(vehicle)
    store = vehicle.store
    store.set("mtargeting", mtargeting)
    store.set("del_radius", 0.0)
    store.define(Field("lonx", HYPER_LONX, "real", "init/diag", "newton"))
    store.define(Field("latx", HYPER_LATX, "real", "init/diag", "newton"))
    store.define(
        Field("sbii", _sbii(HYPER_LONX, HYPER_LATX, HYPER_ALT), "vec", "state", "newton")
    )
    for name, value in (
        ("wp_lonx", 7.0),
        ("wp_latx", 7.0),
        ("wp_alt", 7.0),
    ):
        store.define(Field(name, value, "real", "data", "guidance"))
    if combus is None:
        combus = _combus_one_each()
    targeting.initialize(vehicle, _ctx(combus))
    return vehicle, targeting, _ctx(combus)


def test_mtargeting_1_sets_waypoint_to_target_lon_lat_alt():
    vehicle, targeting, ctx = _ready(mtargeting=1)
    targeting.execute(vehicle, ctx)
    store = vehicle.store
    assert store.get("wp_lonx") == pytest.approx(TGT_LONX, rel=RTOL, abs=ATOL)
    assert store.get("wp_latx") == pytest.approx(TGT_LATX, rel=RTOL, abs=ATOL)
    assert store.get("wp_alt") == pytest.approx(TGT_ALT, rel=RTOL, abs=ATOL)
    assert store.get("clost_tgt_slot") == 1
    assert store.get("tgtng_sat_slot") == 2


def test_mtargeting_2_raises():
    vehicle, targeting, ctx = _ready(mtargeting=2)
    with pytest.raises(ValueError, match="unknown mtargeting"):
        targeting.execute(vehicle, ctx)
    store = vehicle.store
    assert store.get("wp_lonx") == 7.0
    assert store.get("wp_latx") == 7.0
    assert store.get("wp_alt") == 7.0
    assert store.get("clost_tgt_slot") == 0
    assert store.get("tgtng_sat_slot") == 0


def test_identifies_satellite_and_target_by_type_not_id():
    combus = [
        Packet(
            name="s1",
            type="HYPER5",
            status=1,
            vars={"sbii": _sbii(SAT_LONX, SAT_LATX, SAT_ALT)},
        ),
        Packet(
            name="t1",
            type="HYPER5",
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
    assert store.get("wp_lonx") == pytest.approx(TGT_LONX, rel=RTOL, abs=ATOL)
    assert store.get("wp_latx") == pytest.approx(TGT_LATX, rel=RTOL, abs=ATOL)
    assert store.get("wp_alt") == pytest.approx(TGT_ALT, rel=RTOL, abs=ATOL)
    assert store.get("clost_tgt_slot") == 2
    assert store.get("tgtng_sat_slot") == 3


def test_closest_target_by_ground_range():
    combus = [
        _hyper_packet(),
        _target_packet(FAR_TGT_LONX, FAR_TGT_LATX, FAR_TGT_ALT, name="Far"),
        _target_packet(TGT_LONX, TGT_LATX, TGT_ALT, name="Truck"),
        _sat_packet(),
    ]
    vehicle, targeting, ctx = _ready(mtargeting=1, combus=combus)
    targeting.execute(vehicle, ctx)
    store = vehicle.store
    assert store.get("wp_lonx") == pytest.approx(TGT_LONX, rel=RTOL, abs=ATOL)
    assert store.get("wp_latx") == pytest.approx(TGT_LATX, rel=RTOL, abs=ATOL)
    assert store.get("wp_alt") == pytest.approx(TGT_ALT, rel=RTOL, abs=ATOL)
    assert store.get("clost_tgt_slot") == 2
    assert store.get("tgtng_sat_slot") == 3


def test_large_is_module_level_not_in_constants():
    from cadac.vehicles.round3.hyper5 import targeting as targeting_mod
    import cadac.constants as constants

    assert targeting_mod.LARGE == 1e10
    assert not hasattr(constants, "LARGE")
    assert not hasattr(constants, "BIG")


def test_no_plane5_or_flat6_imports():
    text = TARGETING_PY.read_text(encoding="utf-8")
    assert "plane5" not in text
    assert "Plane5" not in text
    assert "flat6" not in text
    assert "Flat6" not in text
