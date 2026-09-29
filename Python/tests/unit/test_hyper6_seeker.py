"""HYPER6 vehicle RF seeker (mseek 2–5), source-faithful to Hyper::seeker."""

import math

import numpy as np
import pytest

from cadac.constants import DEG
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import cart_from_pol, polar_from_cart
from cadac.vehicles.round6.hyper6.vehicle import Hyper6

RTOL = 1e-12
ATOL = 1e-14

INT_STEP = 0.01
TIME = 10.0
RACQ = 50000.0
DTIMAC = 2.0

SBII = np.array([6.5e6, 1.0e5, 2.0e5], dtype=float)
VBII = np.array([100.0, 7000.0, 50.0], dtype=float)
STII = np.array([6.52e6, 1.05e5, 2.1e5], dtype=float)
VTII = np.array([80.0, 7100.0, 40.0], dtype=float)
TBI = np.eye(3)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(int_step=INT_STEP, sim_time=TIME, combus=None):
    return SimContext(
        sim_time=sim_time,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=0,
    )


def _sat_packet(stii=STII, vtii=VTII):
    return Packet(
        name="Satellite",
        type="SAT3",
        status=1,
        vars={"sbii": np.array(stii, dtype=float), "vbii": np.array(vtii, dtype=float)},
    )


def _plant_kinematics(store):
    for name, value, ftype, role, module in (
        ("time", TIME, "real", "exec", "kinematics"),
        ("SBII", SBII, "vec", "state", "newton"),
        ("VBII", VBII, "vec", "state", "newton"),
        ("TBI", TBI, "mat", "out", "kinematics"),
        ("mguide", 0, "int", "data", "guidance"),
        ("trcode", 0.0, "real", "init", "aerodynamics"),
    ):
        if name not in store.names():
            store.define(Field(name, value, ftype, role, module))
        store.set(name, value)


def _seeker_module():
    from cadac.vehicles.round6.hyper6.seeker import Hyper6Seeker

    return Hyper6Seeker()


def _ready(mseek, skr_dyn=0, racq=RACQ, dtimac=DTIMAC, isets1=1, combus=None):
    vehicle = _Vehicle()
    seeker = _seeker_module()
    seeker.define(vehicle)
    _plant_kinematics(vehicle.store)
    store = vehicle.store
    if "sat_num" not in store.names():
        store.define(Field("sat_num", 0, "int", "data", "combus"))
    store.set("sat_num", 1)
    store.set("mseek", mseek)
    store.set("skr_dyn", skr_dyn)
    store.set("racq", racq)
    store.set("dtimac", dtimac)
    store.set("isets1", isets1)
    store.set("fovlimx", 90.0)
    packets = combus if combus is not None else [_sat_packet()]
    ctx = _ctx(combus=packets)
    seeker.initialize(vehicle, ctx)
    return seeker, vehicle, ctx


def _expected_kin():
    sbti = SBII - STII
    stbi = -sbti
    dab = float(np.linalg.norm(stbi))
    utbi = stbi / dab
    vtbi = VTII - VBII
    ddab = float(utbi @ vtbi)
    stbb = TBI @ stbi
    polar = polar_from_cart(stbb)
    azab = float(polar[1])
    elab = float(polar[2])
    stbbk = cart_from_pol(dab, azab, elab)
    stbik = TBI.T @ stbbk
    vtbik = VTII - VBII
    return stbik, vtbik, dab, ddab, azab * DEG, elab * DEG


def test_hyper6_has_seeker_module():
    # Break: seeker missing / wrong Cape order (after ins, before guidance).
    vehicle = Hyper6("Hypersonic", None, None)
    names = [module.name for module in vehicle.modules]
    assert "seeker" in names
    assert names.index("seeker") > names.index("ins")
    assert names.index("seeker") < names.index("guidance")
    seeker = next(m for m in vehicle.modules if m.name == "seeker")
    assert type(seeker).__name__ == "Hyper6Seeker"


def test_hyper6_mseek_3_tracking_fields():
    # Break: mseek KeyError / missing STBIK, dab, azabx from RF seeker.
    seeker, vehicle, ctx = _ready(mseek=3, skr_dyn=0, isets1=1)
    seeker.execute(vehicle, ctx)
    store = vehicle.store
    stbik, vtbik, dab, ddab, azabx, elabx = _expected_kin()
    assert store.get("mseek") == 3
    np.testing.assert_allclose(store.get("STBIK"), stbik, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VTBIK"), vtbik, rtol=RTOL, atol=ATOL)
    assert store.get("dab") == pytest.approx(dab, rel=RTOL, abs=ATOL)
    assert store.get("ddab") == pytest.approx(ddab, rel=RTOL, abs=ATOL)
    assert store.get("azabx") == pytest.approx(azabx, rel=RTOL, abs=ATOL)
    assert store.get("elabx") == pytest.approx(elabx, rel=RTOL, abs=ATOL)
    assert store.get("dbtk") == pytest.approx(float(np.linalg.norm(SBII - STII)), rel=RTOL, abs=ATOL)


def test_mseek_2_acquires_within_racq():
    seeker, vehicle, ctx = _ready(mseek=2, racq=RACQ)
    seeker.execute(vehicle, ctx)
    assert vehicle.store.get("mseek") == 3


@pytest.mark.parametrize("mseek", [2, 3, 4, 5])
def test_mseek_2_to_5_do_not_raise(mseek):
    seeker, vehicle, ctx = _ready(mseek=mseek, skr_dyn=0, isets1=0)
    seeker.execute(vehicle, ctx)
    assert vehicle.store.get("mseek") in (2, 3, 4, 5)
    assert "STBIK" in vehicle.store.names()
    assert math.isfinite(float(vehicle.store.get("dab")))
