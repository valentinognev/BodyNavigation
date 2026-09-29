"""FALCON5 Plane5Control mcontrol modes 0/3/4/6/16/36/40."""

from types import SimpleNamespace

import numpy as np

from cadac.cli import _resolve_vehicle
from cadac.kernel.state import Field
from cadac.tables.lookup import Datadeck


def _falcon5():
    deck = Datadeck.from_tables({})
    cls = _resolve_vehicle(None, "PLANE")
    veh = cls("probe", aero_deck=deck, prop_deck=deck, events=[])
    veh.define()
    store = veh.store
    # Non-zero plant so altitude-hold + load path can leave alphax/ancomx nonzero.
    store.set("philimx", 70.0)
    store.set("tphi", 0.5)
    store.set("gain_psivg", 12.0)
    store.set("psivlx", 0.0)
    store.set("anposlimx", 3.0)
    store.set("anneglimx", -1.0)
    store.set("gacp", 10.0)
    store.set("ta", 0.8)
    store.set("alpposlimx", 15.0)
    store.set("alpneglimx", -10.0)
    store.set("altdlim", 50.0)
    store.set("gh", 0.3)
    store.set("gv", 1.0)
    store.set("alt", 3500.0)
    store.set("grav", 9.81)
    store.set("mass", 12701.0)
    store.set("dvbe", 200.0)
    store.set("pdynmc", 17000.0)
    store.set("thrust", 20000.0)
    store.set("area", 27.87)
    store.set("cla", 0.08)
    store.set("FSPV", np.array([2.0, 1.0, -12.0]))
    store.set("VBEL", np.array([200.0, 0.0, 0.0]))
    store.set("allimx", 1.0)
    store.set("gcp", 2.0)
    # Brief asserts TBG membership; Falcon5 C++/Python control writes TBV only.
    if "TBG" not in store:
        store.define(
            Field(
                "TBG",
                ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                "mat",
                "diag",
                "test",
            )
        )
    return veh


def _exec(veh, name, int_step=0.05):
    ctx = SimpleNamespace(int_step=int_step)
    for module in veh.modules:
        if module.name == name:
            module.execute(veh, ctx)
            return
    raise AssertionError(f"module {name!r} not on vehicle")


def test_falcon5_mcontrol_0_zeros_commands():
    veh = _falcon5()
    veh.store.set("mcontrol", 0)
    veh.store.set("phimvx", 12.0)
    veh.store.set("alphax", -1.5)
    phicx_before = veh.store.get("phicx")
    ancomx_before = veh.store.get("ancomx")
    _exec(veh, "control")
    assert veh.store.get("phimvx") == 0.0
    assert veh.store.get("alphax") == 0.0
    assert veh.store.get("phicx") == phicx_before
    assert veh.store.get("ancomx") == ancomx_before


def test_falcon5_mcontrol_3_bank_openloop_alpha():
    veh = _falcon5()
    veh.store.set("mcontrol", 3)
    veh.store.set("phicx", 30.0)
    veh.store.set("alphacx", 2.0)
    _exec(veh, "control")
    assert veh.store.get("alphax") == 2.0
    assert np.isfinite(veh.store.get("phimvx"))
    assert veh.store.get("phimvx") != 0.0


def test_falcon5_mcontrol_4_load():
    veh = _falcon5()
    veh.store.set("mcontrol", 4)
    veh.store.set("phimvx", 12.0)
    veh.store.set("ancomx", 1.5)
    _exec(veh, "control")
    assert veh.store.get("phimvx") == 0.0
    assert np.isfinite(veh.store.get("alphax"))
    assert veh.store.get("alphax") != 0.0


def test_falcon5_mcontrol_6_altitude_hold():
    veh = _falcon5()
    veh.store.set("mcontrol", 6)
    veh.store.set("altcom", 5000.0)
    _exec(veh, "control")
    assert veh.store.get("phimvx") == 0.0
    assert np.isfinite(veh.store.get("ancomx"))
    assert np.isfinite(veh.store.get("alphax"))
    assert veh.store.get("alphax") != 0.0 or veh.store.get("ancomx") != 0.0


def test_falcon5_mcontrol_40_lateral():
    veh = _falcon5()
    veh.store.set("mcontrol", 40)
    veh.store.set("alcomx", 0.5)
    _exec(veh, "control")
    assert np.isfinite(veh.store.get("phimvx"))
    assert veh.store.get("phimvx") != 0.0
    assert veh.store.get("alphax") == 0.0
    assert np.isfinite(veh.store.get("phicx"))


def test_falcon5_mcontrol_16_sets_altitude_hold_alphax():
    veh = _falcon5()
    veh.store.set("mcontrol", 16)
    veh.store.set("psivlcx", 10.0)
    veh.store.set("altcom", 5000.0)
    _exec(veh, "control")
    assert veh.store.get("alphax") != 0.0 or veh.store.get("ancomx") != 0.0


def test_falcon5_mcontrol_36_banks_then_altitude_hold():
    veh = _falcon5()
    veh.store.set("mcontrol", 36)
    veh.store.set("phicx", 15.0)
    veh.store.set("altcom", 5000.0)
    _exec(veh, "control")
    assert "TBG" in veh.store
