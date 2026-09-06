import numpy as np
import pytest

from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.aim5.aircraft import (
    Aim5AircraftControl,
    Aim5AircraftForces,
    Aim5AircraftGuidance,
)

RTOL = 1e-12
ATOL = 1e-14

GRAV = 9.8
GTURN = 2.0
GUID_GAIN = 3.0
AIRCRAFT_SBEL = np.array([0.0, 0.0, -10000.0])
AIRCRAFT_VBEL = np.array([0.0, -269.0, 0.0])
MISSILE_SBEL = np.array([0.0, -9000.0, -10000.0])
MISSILE_VBEL = np.array([269.0 / np.sqrt(2.0), 269.0 / np.sqrt(2.0), 0.0])

GUIDANCE_FIELDS = {
    "acft_option": ("int", "data", 0),
    "guid_gain": ("real", "data", 0.0),
    "gturn": ("real", "data", 0.0),
}

EXTERNALS = ("grav", "TVL", "SBEL", "VBEL")


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


def _expected_escape(sbel, vbel, stel, vtel, guid_gain, grav):
    satl = sbel - stel
    dab = float(np.linalg.norm(satl))
    vael = vbel
    gain = guid_gain * float(np.linalg.norm(_skew(vael) @ vtel)) / dab
    uvtel = vtel / np.linalg.norm(vtel)
    uvael = vael / np.linalg.norm(vael)
    epsl = _skew(uvael) @ uvtel
    return _skew(epsl) @ uvael * gain + np.array([0.0, 0.0, -grav])


def _missile_packet():
    return Packet(
        name="m1",
        type="AIM5",
        status=1,
        vars={
            "SBEL": MISSILE_SBEL.copy(),
            "VBEL": MISSILE_VBEL.copy(),
        },
    )


def _decoy_t1_packet():
    return Packet(
        name="t1",
        type="AIRCRAFT3",
        status=1,
        vars={
            "SBEL": np.array([999.0, 999.0, 999.0]),
            "VBEL": np.array([1.0, 0.0, 0.0]),
        },
    )


def _ctx(combus=None, vehicle_slot=0):
    return SimContext(
        sim_time=0.0,
        int_step=0.002,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=vehicle_slot,
    )


def _ready_guidance(
    *,
    acft_option=0,
    guid_gain=GUID_GAIN,
    gturn=GTURN,
    grav=GRAV,
    tvl=None,
    sbel=None,
    vbel=None,
    combus=None,
):
    vehicle = _Vehicle()
    guidance = Aim5AircraftGuidance()
    guidance.define(vehicle)
    store = vehicle.store
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(
        Field("TVL", np.eye(3) if tvl is None else tvl, "mat", "out", "kinematics")
    )
    store.define(
        Field(
            "SBEL",
            AIRCRAFT_SBEL if sbel is None else sbel,
            "vec",
            "state",
            "newton",
        )
    )
    store.define(
        Field(
            "VBEL",
            AIRCRAFT_VBEL if vbel is None else vbel,
            "vec",
            "state",
            "newton",
        )
    )
    store.set("acft_option", acft_option)
    store.set("guid_gain", guid_gain)
    store.set("gturn", gturn)
    guidance.initialize(vehicle, _ctx(combus))
    return vehicle, guidance, combus


def test_name_is_guidance():
    assert Aim5AircraftGuidance().name == "guidance"


def test_forces_and_control_still_present():
    assert Aim5AircraftForces().name == "forces"
    assert Aim5AircraftControl().name == "control"


def test_define_registers_guidance_fields():
    vehicle = _Vehicle()
    Aim5AircraftGuidance().define(vehicle)
    store = vehicle.store
    names = store.names()
    for name, (ftype, role, default) in GUIDANCE_FIELDS.items():
        assert name in names
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "guidance"
        assert field.outputs == ()
        assert field.value == pytest.approx(default, abs=ATOL)

    assert "ACOML" in names
    acoml = store.field("ACOML")
    np.testing.assert_array_equal(acoml.value, np.zeros(3))
    assert acoml.type == "vec"
    assert acoml.role == "out"
    assert acoml.module == "guidance"
    assert acoml.outputs == ()
    for name in EXTERNALS:
        assert name not in names


def test_initialize_is_pass():
    vehicle, _guidance, _combus = _ready_guidance()
    np.testing.assert_array_equal(vehicle.store.get("ACOML"), np.zeros(3))
    assert vehicle.store.get("acft_option") == 0
    assert vehicle.store.get("guid_gain") == pytest.approx(GUID_GAIN, abs=ATOL)


def test_terminate_is_pass():
    vehicle, guidance, _combus = _ready_guidance()
    guidance.terminate(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("ACOML"), np.zeros(3))
    assert vehicle.store.get("acft_option") == 0


def test_option_0_horizontal_gravity_bias():
    vehicle, guidance, _combus = _ready_guidance(acft_option=0, grav=GRAV)
    guidance.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        vehicle.store.get("ACOML"),
        np.array([0.0, 0.0, -9.8]),
        rtol=RTOL,
        atol=ATOL,
    )


def test_option_1_horizontal_gturn_identity_tvl():
    vehicle, guidance, _combus = _ready_guidance(
        acft_option=1, gturn=2, tvl=np.eye(3), grav=GRAV
    )
    guidance.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        vehicle.store.get("ACOML"),
        np.array([0.0, 19.6, -9.8]),
        rtol=RTOL,
        atol=ATOL,
    )


def test_option_2_escape_matches_replica():
    combus = [_decoy_t1_packet(), _missile_packet()]
    vehicle, guidance, combus = _ready_guidance(acft_option=2, combus=combus)
    guidance.execute(vehicle, _ctx(combus))
    acoml = vehicle.store.get("ACOML")
    assert np.isfinite(acoml).all()
    expected = _expected_escape(
        AIRCRAFT_SBEL,
        AIRCRAFT_VBEL,
        MISSILE_SBEL,
        MISSILE_VBEL,
        GUID_GAIN,
        GRAV,
    )
    np.testing.assert_allclose(acoml, expected, rtol=RTOL, atol=ATOL)


def test_option_3_raises():
    vehicle, guidance, _combus = _ready_guidance(acft_option=3)
    with pytest.raises(ValueError):
        guidance.execute(vehicle, _ctx())
