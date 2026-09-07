import numpy as np
import pytest
from cadac.constants import DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import cadtbv
from cadac.vehicles.cruise5.control import Cruise5Control

RTOL = 1e-12
INT_STEP = 0.05


def _ctx():
    return SimContext(0.0, INT_STEP, 0.0, 0.0, None, 0)


def test_mcontrol_0_zeros_phimvx_alphax():
    vehicle = type("V", (), {"store": StateStore()})()
    control = Cruise5Control()
    control.define(vehicle)
    store = vehicle.store
    store.define(Field(
        "tgv", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "mat", "init", "newton",
    ))
    store.set("phimvx", 12.0)
    store.set("alphax", 5.0)
    store.set("mcontrol", 0)
    control.execute(vehicle, _ctx())
    assert store.get("phimvx") == 0.0
    assert store.get("alphax") == 0.0


def test_mcontrol_46_writes_finite_phimvx_alphax():
    vehicle = type("V", (), {"store": StateStore()})()
    control = Cruise5Control()
    control.define(vehicle)
    store = vehicle.store
    store.define(Field(
        "tgv", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "mat", "init", "newton",
    ))
    for name, value, ftype in (
        ("FSPV", np.array([2.0, 1.0, -12.0]), "vec"),
        ("grav", 9.81, "real"),
        ("mass", 1000.0, "real"),
        ("dvbe", 200.0, "real"),
        ("pdynmc", 5000.0, "real"),
        ("thrust", 1500.0, "real"),
        ("area", 0.929, "real"),
        ("cla", 0.11, "real"),
        ("vbeg", np.array([200.0, 0.0, 0.0]), "vec"),
        ("alt", 7000.0, "real"),
    ):
        if name not in store.names():
            store.define(Field(name, value, ftype, "out", "test"))
        else:
            store.set(name, value)
    store.set("mcontrol", 46)
    store.set("alcomx", 0.2)
    store.set("gcp", 2.0)
    store.set("allimx", 1.0)
    store.set("philimx", 70.0)
    store.set("tphi", 0.5)
    store.set("gacp", 10.0)
    store.set("ta", 0.8)
    store.set("anposlimx", 3.0)
    store.set("anneglimx", -1.0)
    store.set("alpposlimx", 15.0)
    store.set("alpneglimx", -10.0)
    store.set("altcom", 7000.0)
    store.set("gh", 0.3)
    store.set("gv", 1.0)
    store.set("altdlim", 50.0)
    control.execute(vehicle, _ctx())
    assert np.isfinite(store.get("phimvx"))
    assert np.isfinite(store.get("alphax"))
    assert store.get("TBV").shape == (3, 3)
    assert store.get("TBG").shape == (3, 3)


def test_mcontrol_44_uses_ancomx_not_altitude():
    vehicle = type("V", (), {"store": StateStore()})()
    control = Cruise5Control()
    control.define(vehicle)
    store = vehicle.store
    store.define(Field(
        "tgv", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "mat", "init", "newton",
    ))
    for name, value, ftype in (
        ("FSPV", np.array([2.0, 1.0, -12.0]), "vec"),
        ("grav", 9.81, "real"),
        ("mass", 1000.0, "real"),
        ("dvbe", 200.0, "real"),
        ("pdynmc", 5000.0, "real"),
        ("thrust", 1500.0, "real"),
        ("area", 0.929, "real"),
        ("cla", 0.11, "real"),
    ):
        if name not in store.names():
            store.define(Field(name, value, ftype, "out", "test"))
    store.set("mcontrol", 44)
    store.set("alcomx", 0.2)
    store.set("ancomx", 1.5)
    store.set("gcp", 2.0)
    store.set("allimx", 1.0)
    store.set("philimx", 70.0)
    store.set("tphi", 0.5)
    store.set("gacp", 10.0)
    store.set("ta", 0.8)
    store.set("anposlimx", 3.0)
    store.set("anneglimx", -1.0)
    store.set("alpposlimx", 15.0)
    store.set("alpneglimx", -10.0)
    control.execute(vehicle, _ctx())
    assert np.isfinite(store.get("alphax"))
    assert np.isfinite(store.get("phimvx"))


def test_mcontrol_unknown_raises():
    vehicle = type("V", (), {"store": StateStore()})()
    control = Cruise5Control()
    control.define(vehicle)
    vehicle.store.define(Field(
        "tgv", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "mat", "init", "newton",
    ))
    vehicle.store.set("mcontrol", 16)
    with pytest.raises(ValueError, match="mcontrol"):
        control.execute(vehicle, _ctx())


def _expected_lateral(alcomx, allimx, gcp, fspv, grav, phimvx, alphax):
    tbv = cadtbv(phimvx * RAD, alphax * RAD)
    fspb = tbv @ fspv
    anx = -fspb[2] / grav
    if alcomx > allimx:
        alcomx = allimx
    if alcomx < -allimx:
        alcomx = -allimx
    sign = 1 if anx >= 0 else -1
    phicx = DEG * gcp * sign / (abs(anx) + 0.001) * alcomx
    alx = fspv[1] / grav
    return phicx, alx


def test_control_lateral_matches_cpp_replica():
    vehicle = type("V", (), {"store": StateStore()})()
    control = Cruise5Control()
    control.define(vehicle)
    store = vehicle.store
    fspv = np.array([2.0, 1.0, -12.0])
    store.define(Field("FSPV", fspv, "vec", "out", "test"))
    store.define(Field("grav", 9.81, "real", "out", "test"))
    store.set("gcp", 2.0)
    store.set("allimx", 1.0)
    store.set("alcomx", 0.5)
    store.set("alphax", 0.0)
    store.set("phimvx", 0.0)
    expected_phicx, expected_alx = _expected_lateral(
        0.5, 1.0, 2.0, fspv, 9.81, 0.0, 0.0,
    )
    got = control.control_lateral(vehicle, 0.5)
    assert got == pytest.approx(expected_phicx, rel=RTOL, abs=1e-14)
    assert store.get("alx") == pytest.approx(expected_alx, rel=RTOL, abs=1e-14)
    assert store.get("phicx") == 0.0
