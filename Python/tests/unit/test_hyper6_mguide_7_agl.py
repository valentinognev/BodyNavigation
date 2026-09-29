"""HYPER6 mguide=7 AGL: STBIK supplied from kinematics when seeker absent."""
from types import SimpleNamespace

import numpy as np

from cadac.constants import REARTH
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr
from cadac.tables.lookup import Datadeck
from cadac.vehicles.round6.hyper6.vehicle import Hyper6

INT_STEP = 0.01
TBIC = mat2tr(0.3, -0.2)
SBIIC = np.array([REARTH + 50e3, 1.0e5, 2.0e5])
VBIIC = np.array([400.0, 2500.0, 200.0])
STCII = np.array([7005000.0, 98000.0, 201000.0])
VTCII = np.array([80.0, 7480.0, 40.0])


def _execute_guidance(veh):
    ctx = SimpleNamespace(int_step=INT_STEP)
    for module in veh.modules:
        if module.name == "guidance":
            module.execute(veh, ctx)
            return
    raise AssertionError("guidance module missing")


def _ensure(store, name, value, ftype):
    if name not in store:
        store.define(Field(name, value, ftype, "data", "test"))
    else:
        store.set(name, value)


def test_hyper6_mguide_7_agl():
    """mguide=7 must not KeyError STBIK; AGL writes finite body accel commands."""
    deck = Datadeck.from_tables([])
    veh = Hyper6("hyper6", aero_deck=deck, prop_deck=deck, events=[])
    veh.define()
    store = veh.store
    store.set("mguide", 7)
    store.set("gnavpn", 4.0)
    store.set("gnavps", 1.5)
    store.set("SBIIC", SBIIC)
    store.set("VBIIC", VBIIC)
    store.set("TBIC", TBIC)
    store.set("alcomx", 7.0)
    store.set("ancomx", 7.0)
    store.set("phicomx", 7.0)
    # Truth kinematics only — do not plant STBIK/VTBIK (seeker not on vehicle).
    _ensure(store, "STCII", STCII, "vec")
    _ensure(store, "VTCII", VTCII, "vec")
    assert "STBIK" not in store
    assert "VTBIK" not in store

    _execute_guidance(veh)

    assert "STBIK" in store
    stbik = np.asarray(store.get("STBIK"), dtype=float)
    np.testing.assert_allclose(stbik, STCII - SBIIC, rtol=0.0, atol=0.0)
    assert np.isfinite(store.get("aycomx"))
    assert np.isfinite(store.get("azcomx"))
    assert store.get("aycomx") != 0.0 or store.get("azcomx") != 0.0
