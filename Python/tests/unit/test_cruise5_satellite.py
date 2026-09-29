import inspect
import pathlib

import numpy as np

from cadac.eom.round3 import Round3Newton
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round3.cruise5.environment import Cruise5Environment
from cadac.vehicles.round3.cruise5.satellite import (
    Cruise5Satellite,
    Cruise5SatelliteForces,
)

RTOL = 1e-12
ATOL = 1e-14
COM_AT_LEAST = (
    "time",
    "mach",
    "lonx",
    "latx",
    "alt",
    "dvbe",
    "psivgx",
    "thtvgx",
    "sbeg",
    "vbeg",
    "sbii",
)
SATELLITE_PY = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "cadac"
    / "vehicles"
    / "round3"
    / "cruise5"
    / "satellite.py"
)


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.05,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def test_constructor_type_health_no_aero_deck():
    sig = inspect.signature(Cruise5Satellite.__init__)
    assert list(sig.parameters) == ["self", "name", "events"]
    assert sig.parameters["events"].default is None
    vehicle = Cruise5Satellite("Sat_s1")
    assert vehicle.type == "SATELLITE3"
    assert vehicle.name == "Sat_s1"
    assert vehicle.health == 1
    assert not hasattr(vehicle, "aero_deck")
    assert [type(m) for m in vehicle.modules] == [
        Cruise5Environment,
        Cruise5SatelliteForces,
        Round3Newton,
    ]


def test_com_names_from_com_outputs():
    vehicle = Cruise5Satellite("Sat_s1")
    vehicle.define()
    assert {"lonx", "latx", "alt", "sbii"} <= set(vehicle.com_names)
    for name in COM_AT_LEAST:
        assert name in vehicle.com_names
        assert "com" in vehicle.store.field(name).outputs
    assert vehicle.com_names == [
        name
        for name in vehicle.store.names()
        if "com" in vehicle.store.field(name).outputs
    ]


def test_define_skips_existing_fspv():
    vehicle = type("V", (), {"store": StateStore()})()
    sentinel = np.array([1.0, 2.0, 3.0])
    vehicle.store.define(Field("FSPV", sentinel, "vec", "out", "newton"))
    Cruise5SatelliteForces().define(vehicle)
    np.testing.assert_array_equal(vehicle.store.get("FSPV"), sentinel)
    assert vehicle.store.field("FSPV").module == "newton"
    assert vehicle.store.get("sat_thrust") == 0.0
    assert vehicle.store.get("sat_mass") == 100.0


def test_default_thrust_zero_fspv0():
    vehicle = Cruise5Satellite("Sat_s1")
    vehicle.define()
    assert vehicle.store.get("sat_thrust") == 0.0
    assert vehicle.store.get("sat_mass") == 100.0
    named = {m.name: m for m in vehicle.modules}
    named["forces"].execute(vehicle, _ctx())
    fspv = vehicle.store.get("FSPV")
    np.testing.assert_allclose(fspv[0], 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(fspv[1:], 0.0, rtol=RTOL, atol=ATOL)


def test_thrust_100_fspv0_is_one():
    vehicle = Cruise5Satellite("Sat_s1")
    vehicle.define()
    vehicle.store.set("sat_thrust", 100.0)
    named = {m.name: m for m in vehicle.modules}
    named["forces"].execute(vehicle, _ctx())
    fspv = vehicle.store.get("FSPV")
    np.testing.assert_allclose(fspv[0], 1.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(fspv[1:], 0.0, rtol=RTOL, atol=ATOL)


def test_no_hyper5_satellite_import():
    text = SATELLITE_PY.read_text()
    assert "cadac.vehicles.round3.hyper5.satellite" not in text
    assert "Satellite3" not in text
    assert "Cruise3" not in text
    assert "Hyper5" not in text
    assert "Target3" not in text
