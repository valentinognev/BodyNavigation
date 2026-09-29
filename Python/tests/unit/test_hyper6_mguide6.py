"""HYPER6 mguide=6 pro-nav: seeker/datalink fields satisfy guidance_pronav."""
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
STBIK = np.array([12000.0, -3500.0, 800.0])
VTBIK = np.array([-180.0, 40.0, 25.0])
STCII = np.array([7005000.0, 98000.0, 201000.0])
VTCII = np.array([80.0, 7480.0, 40.0])
STII = np.array([7.1e6, 0.0, 0.0])
VTII = np.array([0.0, 7500.0, 0.0])
SBII = np.array([6.5e6, 1.0e5, 2.0e5], dtype=float)
VBII = np.array([100.0, 7000.0, 50.0], dtype=float)


def _hyper6():
    deck = Datadeck.from_tables([])
    veh = Hyper6("hyper6", aero_deck=deck, prop_deck=deck, events=[])
    veh.define()
    return veh


def _plant_pronav_kinematics(store):
    store.set("mguide", 6)
    store.set("gnav", 3.0)
    store.set("SBIIC", SBIIC)
    store.set("VBIIC", VBIIC)
    store.set("TBIC", TBIC)
    store.set("STCII", STCII)
    store.set("VTCII", VTCII)
    store.set("STII", STII)
    store.set("VTII", VTII)
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


def test_hyper6_has_seeker_for_mguide6():
    veh = _hyper6()
    names = [m.name for m in veh.modules]
    assert "seeker" in names
    assert "datalink" in names
    assert "mseek" in veh.store


def test_hyper6_mguide6_seeker_tracking_no_keyerror():
    """mguide=6 with mseek>3 uses STBIK/VTBIK — must not KeyError mseek."""
    veh = _hyper6()
    store = veh.store
    _plant_pronav_kinematics(store)
    store.set("mseek", 4)
    store.set("STBIK", STBIK)
    store.set("VTBIK", VTBIK)

    _execute(veh, "guidance")

    assert np.isfinite(store.get("aycomx"))
    assert np.isfinite(store.get("azcomx"))
    assert store.get("aycomx") != 0.0 or store.get("azcomx") != 0.0
    assert np.isfinite(np.linalg.norm(np.asarray(store.get("UTBC"), dtype=float)))


def test_hyper6_mguide6_datalink_fields_no_keyerror():
    """mguide=6 with mseek<=3 uses STCII/VTCII relative kinematics."""
    veh = _hyper6()
    store = veh.store
    _plant_pronav_kinematics(store)
    store.set("mseek", 3)

    _execute(veh, "guidance")

    assert np.isfinite(store.get("aycomx"))
    assert np.isfinite(store.get("azcomx"))
    assert store.get("aycomx") != 0.0 or store.get("azcomx") != 0.0


def test_hyper6_mguide6_one_step_seeker_then_guidance():
    """Seeker tracking fields then guidance execute — full one-step wiring."""
    veh = _hyper6()
    store = veh.store
    store.set("mguide", 6)
    store.set("gnav", 3.0)
    store.set("mseek", 3)
    store.set("skr_dyn", 0)
    store.set("racq", 50000.0)
    store.set("dtimac", 2.0)
    store.set("isets1", 1)
    store.set("fovlimx", 90.0)
    store.set("sat_num", 1)
    store.set("time", 10.0)
    store.set("SBII", SBII)
    store.set("VBII", VBII)
    store.set("TBI", np.eye(3))
    store.set("SBIIC", SBIIC)
    store.set("VBIIC", VBIIC)
    store.set("TBIC", TBIC)
    store.set("STCII", STCII)
    store.set("VTCII", VTCII)
    store.set("alcomx", 0.0)
    store.set("ancomx", 0.0)
    store.set("phicomx", 0.0)
    store.set("trcode", 0.0)

    packets = [
        Packet(
            name="Satellite",
            type="SAT3",
            status=1,
            vars={"sbii": STII.copy(), "vbii": VTII.copy()},
        )
    ]
    ctx = SimContext(
        sim_time=10.0,
        int_step=INT_STEP,
        event_time=0.0,
        out_fact=0.0,
        combus=packets,
        vehicle_slot=0,
    )
    for module in veh.modules:
        if module.name == "seeker":
            module.initialize(veh, ctx)
            break

    _execute(veh, "seeker", ctx)
    assert store.get("mseek") == 3
    assert "STBIK" in store
    assert "STII" in store

    _execute(veh, "guidance", ctx)

    assert np.isfinite(store.get("aycomx"))
    assert np.isfinite(store.get("azcomx"))
    assert store.get("aycomx") != 0.0 or store.get("azcomx") != 0.0
