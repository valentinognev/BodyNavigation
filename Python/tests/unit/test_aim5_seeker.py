import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr, polar_from_cart
from cadac.vehicles.aim5.seeker import Aim5Seeker

RTOL = 1e-12
ATOL = 1e-14

DVAE = 269.0
MISSILE_SBEL = np.array([0.0, -9000.0, -10000.0])
AIRCRAFT_SBEL = np.array([0.0, 0.0, -10000.0])
AIRCRAFT2_SBEL = np.array([1000.0, 0.0, -10000.0])
MISSILE_PSIVLX = 45.0
MISSILE_THTVLX = 0.0
AIRCRAFT_PSIVLX = -90.0
AIRCRAFT_THTVLX = 0.0

SEEKER_FIELDS = {
    "acft_com_slot": ("int", "out", "combus", ()),
    "VTEL": ("vec", "out", "combus", ()),
    "psivlx_acft": ("real", "out", "combus", ()),
    "thtvlx_acft": ("real", "out", "combus", ()),
    "tgt_num": ("int", "data", "combus", ()),
    "mseek": ("int", "data", "seeker", ()),
    "dta": ("real", "out", "seeker", ("scrn", "plot")),
    "dvta": ("real", "out", "seeker", ()),
    "tgo_aim": ("real", "diag", "seeker", ()),
    "los_azx": ("real", "diag", "seeker", ()),
    "los_elx": ("real", "diag", "seeker", ()),
    "sigdy": ("real", "diag", "seeker", ()),
    "sigdz": ("real", "diag", "seeker", ()),
    "UTAA": ("vec", "out", "seeker", ()),
    "WOEA": ("vec", "out", "seeker", ()),
    "STAL": ("vec", "out", "seeker", ()),
}

EXTERNALS = ("TBL", "SBEL", "VBEL")


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _skew(vec):
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def _vbel(dvae, psivlx, thtvlx):
    psivl = psivlx * RAD
    thtvl = thtvlx * RAD
    return np.array(
        [
            dvae * np.cos(thtvl) * np.cos(psivl),
            dvae * np.cos(thtvl) * np.sin(psivl),
            dvae * (-np.sin(thtvl)),
        ]
    )


def _expected_los(sbel, vbel, tbl, stel, vtel):
    stal = stel - sbel
    dta = float(np.linalg.norm(stal))
    utal = stal / dta
    utaa = tbl @ utal
    polar = polar_from_cart(utaa)
    vtael = vtel - vbel
    dvta = float(utal @ vtael)
    tgo_aim = dta / abs(dvta)
    woea = tbl @ (_skew(utal) @ vtael) * (1.0 / dta)
    return {
        "STAL": stal,
        "dta": dta,
        "UTAA": utaa,
        "los_azx": polar[1] * DEG,
        "los_elx": polar[2] * DEG,
        "dvta": dvta,
        "tgo_aim": tgo_aim,
        "WOEA": woea,
        "sigdy": woea[1],
        "sigdz": woea[2],
    }


def _aircraft_packet(sbel, vbel, psivlx, thtvlx, name="Target"):
    return Packet(
        name=name,
        type="AIRCRAFT3",
        status=1,
        vars={
            "SBEL": np.array(sbel, dtype=float),
            "VBEL": np.array(vbel, dtype=float),
            "psivlx": psivlx,
            "thtvlx": thtvlx,
        },
    )


def _missile_packet():
    return Packet(
        name="Missile",
        type="AIM5",
        status=1,
        vars={
            "SBEL": MISSILE_SBEL.copy(),
            "VBEL": _vbel(DVAE, MISSILE_PSIVLX, MISSILE_THTVLX),
            "psivlx": MISSILE_PSIVLX,
            "thtvlx": MISSILE_THTVLX,
        },
    )


def _ctx(combus, vehicle_slot=0):
    return SimContext(
        sim_time=0.0,
        int_step=0.002,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=vehicle_slot,
    )


def _ready(*, mseek=1, tgt_num=None, extra_packets=()):
    vehicle = _Vehicle()
    seeker = Aim5Seeker()
    seeker.define(vehicle)
    store = vehicle.store
    vbel = _vbel(DVAE, MISSILE_PSIVLX, MISSILE_THTVLX)
    tbl = mat2tr(MISSILE_PSIVLX * RAD, MISSILE_THTVLX * RAD)
    store.define(Field("SBEL", MISSILE_SBEL, "vec", "state", "newton"))
    store.define(Field("VBEL", vbel, "vec", "state", "newton"))
    store.define(Field("TBL", tbl, "mat", "out", "newton"))
    store.set("mseek", mseek)
    if tgt_num is not None:
        store.set("tgt_num", tgt_num)
    acft_vbel = _vbel(DVAE, AIRCRAFT_PSIVLX, AIRCRAFT_THTVLX)
    combus = [
        _missile_packet(),
        _aircraft_packet(AIRCRAFT_SBEL, acft_vbel, AIRCRAFT_PSIVLX, AIRCRAFT_THTVLX),
        *extra_packets,
    ]
    seeker.initialize(vehicle, _ctx(combus))
    return vehicle, seeker, combus


def test_name_is_seeker():
    assert Aim5Seeker().name == "seeker"


def test_define_tgt_num_defaults_to_1():
    vehicle = _Vehicle()
    Aim5Seeker().define(vehicle)
    store = vehicle.store
    assert store.get("tgt_num") == 1
    plot = ("scrn", "plot")
    zeros3 = np.zeros(3)
    for name, (ftype, role, module, outputs) in SEEKER_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == module
        assert field.outputs == outputs
        if name == "tgt_num":
            assert field.value == 1
        elif ftype == "int":
            assert field.value == 0
        elif ftype == "vec":
            np.testing.assert_allclose(field.value, zeros3, rtol=RTOL, atol=ATOL)
        else:
            assert field.value == pytest.approx(0.0, abs=ATOL)
    assert store.field("dta").outputs == plot
    for name in EXTERNALS:
        assert name not in store.names()


def test_mseek_1_acquires_first_aircraft3_dta_9000():
    vehicle, seeker, combus = _ready(mseek=1)
    acft = combus[1]
    want = _expected_los(
        vehicle.store.get("SBEL"),
        vehicle.store.get("VBEL"),
        vehicle.store.get("TBL"),
        acft.vars["SBEL"],
        acft.vars["VBEL"],
    )

    seeker.execute(vehicle, _ctx(combus))

    store = vehicle.store
    assert store.get("acft_com_slot") == 1
    np.testing.assert_allclose(store.get("dta"), 9000.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VTEL"), acft.vars["VBEL"], rtol=RTOL, atol=ATOL)
    assert store.get("psivlx_acft") == pytest.approx(AIRCRAFT_PSIVLX, rel=RTOL, abs=ATOL)
    assert store.get("thtvlx_acft") == pytest.approx(AIRCRAFT_THTVLX, rel=RTOL, abs=ATOL)
    for name in ("STAL", "UTAA", "WOEA"):
        np.testing.assert_allclose(store.get(name), want[name], rtol=RTOL, atol=ATOL)
    for name in ("dta", "dvta", "tgo_aim", "los_azx", "los_elx", "sigdy", "sigdz"):
        np.testing.assert_allclose(store.get(name), want[name], rtol=RTOL, atol=ATOL)


def test_mseek_0_writes_slot_but_not_los():
    vehicle, seeker, combus = _ready(mseek=0)
    acft = combus[1]

    seeker.execute(vehicle, _ctx(combus))

    store = vehicle.store
    assert store.get("acft_com_slot") == 1
    np.testing.assert_allclose(store.get("VTEL"), acft.vars["VBEL"], rtol=RTOL, atol=ATOL)
    assert store.get("psivlx_acft") == pytest.approx(AIRCRAFT_PSIVLX, rel=RTOL, abs=ATOL)
    assert store.get("thtvlx_acft") == pytest.approx(AIRCRAFT_THTVLX, rel=RTOL, abs=ATOL)
    assert store.get("dta") == pytest.approx(0.0, abs=ATOL)
    np.testing.assert_allclose(store.get("STAL"), np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("UTAA"), np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("WOEA"), np.zeros(3), rtol=RTOL, atol=ATOL)
    assert store.get("dvta") == pytest.approx(0.0, abs=ATOL)
    assert store.get("tgo_aim") == pytest.approx(0.0, abs=ATOL)


def test_mseek_99_raises():
    vehicle, seeker, combus = _ready(mseek=99)
    with pytest.raises(ValueError):
        seeker.execute(vehicle, _ctx(combus))


def test_tgt_num_selects_among_aircraft3():
    acft2_vbel = _vbel(DVAE, 0.0, 0.0)
    extra = (
        _aircraft_packet(
            AIRCRAFT2_SBEL, acft2_vbel, 0.0, 0.0, name="Decoy"
        ),
    )
    vehicle, seeker, combus = _ready(mseek=1, tgt_num=2, extra_packets=extra)

    seeker.execute(vehicle, _ctx(combus))

    store = vehicle.store
    assert store.get("acft_com_slot") == 2
    np.testing.assert_allclose(store.get("VTEL"), acft2_vbel, rtol=RTOL, atol=ATOL)
    want = _expected_los(
        store.get("SBEL"),
        store.get("VBEL"),
        store.get("TBL"),
        AIRCRAFT2_SBEL,
        acft2_vbel,
    )
    np.testing.assert_allclose(store.get("dta"), want["dta"], rtol=RTOL, atol=ATOL)

    vehicle_one, seeker_one, combus_one = _ready(mseek=1, tgt_num=1, extra_packets=extra)
    seeker_one.execute(vehicle_one, _ctx(combus_one))
    assert vehicle_one.store.get("acft_com_slot") == 1
    np.testing.assert_allclose(
        vehicle_one.store.get("dta"), 9000.0, rtol=RTOL, atol=ATOL
    )


def test_mseek_1_missing_aircraft3_raises():
    vehicle, seeker, combus = _ready(mseek=1, tgt_num=2)
    with pytest.raises(ValueError):
        seeker.execute(vehicle, _ctx(combus))

    vehicle_empty = _Vehicle()
    seeker_empty = Aim5Seeker()
    seeker_empty.define(vehicle_empty)
    store = vehicle_empty.store
    store.define(Field("SBEL", MISSILE_SBEL, "vec", "state", "newton"))
    store.define(
        Field("VBEL", _vbel(DVAE, MISSILE_PSIVLX, MISSILE_THTVLX), "vec", "state", "newton")
    )
    store.define(
        Field(
            "TBL",
            mat2tr(MISSILE_PSIVLX * RAD, MISSILE_THTVLX * RAD),
            "mat",
            "out",
            "newton",
        )
    )
    store.set("mseek", 1)
    with pytest.raises(ValueError):
        seeker_empty.execute(vehicle_empty, _ctx([_missile_packet()]))


def test_initialize_and_terminate_are_pass():
    vehicle, seeker, combus = _ready(mseek=1)
    seeker.terminate(vehicle, _ctx(combus))
    assert vehicle.store.get("dta") == pytest.approx(0.0, abs=ATOL)
    assert vehicle.store.get("tgt_num") == 1
    assert vehicle.store.get("acft_com_slot") == 0
