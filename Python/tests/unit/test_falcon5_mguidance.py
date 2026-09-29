import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat3.falcon5.guidance import Plane5Guidance

RTOL = 1e-12
ATOL = 1e-14

SWEL1 = 5000.0
SWEL2 = 2000.0
SWEL3 = 0.0
LINE_GAIN = 1.5
NL_GAIN_FACT = 0.4
DECREMENT = 800.0
PSIFLX = 180.0
THTFLX = -30.0
POINT_GAIN = 1.0
PHILIMX = 70.0
GRAV = 9.81
THTVLX = 0.0
SBEL = np.array([0.0, 0.0, -3500.0])
VBEL = np.array([200.0, 0.0, 0.0])
ANPOSLIMX = 3.0
ANNEGLIMX = -1.0
ALLIMX = 1.0


def _clip(alcomx, ancomx, anposlimx, anneglimx, allimx):
    if ancomx > anposlimx:
        ancomx = anposlimx
    if ancomx < anneglimx:
        ancomx = anneglimx
    if alcomx > allimx:
        alcomx = allimx
    if alcomx < -allimx:
        alcomx = -allimx
    return alcomx, ancomx


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.05,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _ready(mguidance):
    vehicle = type("V", (), {"store": StateStore()})()
    guidance = Plane5Guidance()
    guidance.define(vehicle)
    store = vehicle.store
    store.define(Field("SBEL", SBEL, "vec", "state", "newton"))
    store.define(Field("VBEL", VBEL, "vec", "state", "newton"))
    store.define(Field("grav", GRAV, "real", "out", "environment"))
    store.define(Field("thtvlx", THTVLX, "real", "out", "newton"))
    store.define(Field("philimx", PHILIMX, "real", "data", "control"))
    store.define(Field("ancomx", 0.0, "real", "data", "control", ("plot",)))
    store.define(Field("alcomx", 0.0, "real", "data", "control", ("plot",)))
    store.define(Field("anposlimx", ANPOSLIMX, "real", "data", "control"))
    store.define(Field("anneglimx", ANNEGLIMX, "real", "data", "control"))
    store.define(Field("allimx", ALLIMX, "real", "data", "control"))
    store.define(Field("phicx", 12.0, "real", "data", "control", ("scrn", "plot")))
    store.set("mguidance", mguidance)
    store.set("swel1", SWEL1)
    store.set("swel2", SWEL2)
    store.set("swel3", SWEL3)
    store.set("point_gain", POINT_GAIN)
    store.set("line_gain", LINE_GAIN)
    store.set("nl_gain_fact", NL_GAIN_FACT)
    store.set("decrement", DECREMENT)
    store.set("psiflx", PSIFLX)
    store.set("thtflx", THTFLX)
    store.set("write", 0)
    return vehicle, guidance


def test_mguidance_43_point_lateral_line_pitch():
    vehicle, guidance = _ready(mguidance=43)
    store = vehicle.store
    phicx_before = store.get("phicx")
    algv = guidance.guidance_line(vehicle)
    apgv = guidance.guidance_point(vehicle)
    grav = store.get("grav")
    want_al, want_an = _clip(
        float(apgv[1] / grav),
        float(-algv[2] / grav),
        store.get("anposlimx"),
        store.get("anneglimx"),
        store.get("allimx"),
    )
    guidance.execute(vehicle, _ctx())
    assert store.get("alcomx") == pytest.approx(want_al, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(want_an, rel=RTOL, abs=ATOL)
    assert store.get("phicx") == phicx_before
