"""HYPER6 mguide=8 glideslope: datalink STCII/VTCII satisfy guidance_glideslope."""

from types import SimpleNamespace

import numpy as np

from cadac.constants import REARTH
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.math.frames import mat2tr
from cadac.tables.lookup import Datadeck
from cadac.vehicles.round6.hyper6.vehicle import Hyper6

INT_STEP = 0.01
TBIC = mat2tr(0.3, -0.2)
SBIIC = np.array([REARTH + 50e3, 1.0e5, 2.0e5])
VBIIC = np.array([400.0, 2500.0, 200.0])
STCII = np.array([7005000.0, 98000.0, 201000.0])
VTCII = np.array([80.0, 7480.0, 40.0])
SATL = np.array([100.0, -50.0, 20.0])


def _hyper6():
    deck = Datadeck.from_tables([])
    veh = Hyper6("hyper6", aero_deck=deck, prop_deck=deck, events=[])
    veh.define()
    return veh


def _plant_glideslope_kinematics(store):
    store.set("mguide", 8)
    store.set("time", 10.0)
    store.set("SBIIC", SBIIC)
    store.set("VBIIC", VBIIC)
    store.set("TBIC", TBIC)
    store.set("STCII", STCII)
    store.set("VTCII", VTCII)
    store.set("mprop", 0)
    store.set("time_gs", 200.0)
    store.set("num_burns", 4)
    store.set("closing_rate", -30.0)
    store.set("orbital_rate", 0.0011)
    store.set("satl1", float(SATL[0]))
    store.set("satl2", float(SATL[1]))
    store.set("satl3", float(SATL[2]))
    store.set("alcomx", 7.0)
    store.set("ancomx", 7.0)
    store.set("phicomx", 7.0)


def _execute(veh, name, ctx=None):
    if ctx is None:
        ctx = SimpleNamespace(int_step=INT_STEP)
    for module in veh.modules:
        if module.name == name:
            module.execute(veh, ctx)
            return
    raise AssertionError(f"{name} module missing")


def test_hyper6_has_datalink_stcii_for_mguide8():
    veh = _hyper6()
    names = [m.name for m in veh.modules]
    assert "datalink" in names
    assert "STCII" in veh.store
    assert "VTCII" in veh.store
    assert veh.store.field("STCII").module == "datalink"


def test_hyper6_mguide8_datalink_stcii_no_keyerror():
    """mguide=8 reads STCII/VTCII — must not KeyError after datalink define."""
    veh = _hyper6()
    store = veh.store
    _plant_glideslope_kinematics(store)
    store.set("mseek", 4)

    _execute(veh, "guidance")

    utbc = np.asarray(store.get("UTBC"), dtype=float)
    assert np.isfinite(utbc).all()
    assert np.linalg.norm(utbc) > 0.0
    assert store.get("gs_flag") == 0
    assert store.get("aycomx") == 0.0
    assert store.get("azcomx") == 0.0


def test_hyper6_mguide8_mseek3_points_utbc_no_keyerror():
    """mguide=8 with mseek==3 re-points UTBC along LOS; still needs STCII."""
    veh = _hyper6()
    store = veh.store
    _plant_glideslope_kinematics(store)
    store.set("mseek", 3)

    _execute(veh, "guidance")

    utbc = np.asarray(store.get("UTBC"), dtype=float)
    assert np.isfinite(utbc).all()
    assert np.linalg.norm(utbc) > 0.0


def test_hyper6_mguide8_one_step_datalink_then_guidance():
    """Datalink writes STCII from radar combus, then guidance mguide=8."""
    veh = _hyper6()
    store = veh.store
    store.set("mguide", 8)
    store.set("sat_num", 1)
    store.set("time", 10.0)
    store.set("SBIIC", SBIIC)
    store.set("VBIIC", VBIIC)
    store.set("TBIC", TBIC)
    store.set("mprop", 0)
    store.set("mseek", 4)
    store.set("time_gs", 200.0)
    store.set("num_burns", 4)
    store.set("closing_rate", -30.0)
    store.set("orbital_rate", 0.0011)
    store.set("satl1", float(SATL[0]))
    store.set("satl2", float(SATL[1]))
    store.set("satl3", float(SATL[2]))
    store.set("alcomx", 0.0)
    store.set("ancomx", 0.0)
    store.set("phicomx", 0.0)

    radar = Packet(
        name="r1",
        type="RADAR0",
        status=1,
        vars={
            "stcii1": STCII.copy(),
            "vtcii1": VTCII.copy(),
            "stcii2": np.zeros(3),
            "vtcii2": np.zeros(3),
            "stcii3": np.zeros(3),
            "vtcii3": np.zeros(3),
            "stcii4": np.zeros(3),
            "vtcii4": np.zeros(3),
            "stcii5": np.zeros(3),
            "vtcii5": np.zeros(3),
        },
    )
    ctx = SimContext(
        sim_time=10.0,
        int_step=INT_STEP,
        event_time=0.0,
        out_fact=0.0,
        combus=[radar],
        vehicle_slot=0,
    )

    _execute(veh, "datalink", ctx)
    np.testing.assert_allclose(store.get("STCII"), STCII, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(store.get("VTCII"), VTCII, rtol=0.0, atol=0.0)
    assert store.get("mnav") == 3

    _execute(veh, "guidance", ctx)

    utbc = np.asarray(store.get("UTBC"), dtype=float)
    assert np.isfinite(utbc).all()
    assert np.linalg.norm(utbc) > 0.0
