import numpy as np
import pytest

from cadac.constants import RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.earth import cadine, cadtei, cadtge
from cadac.math.frames import mat2tr
from cadac.vehicles.round3.cruise5.guidance import Cruise5Guidance

RTOL = 1e-12
ATOL = 1e-14


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


def _round3_state(lonx, latx, alt, psivgx, thtvgx, dvbe, time):
    tei = cadtei(time)
    tge = cadtge(lonx * RAD, latx * RAD)
    tig = tei.T @ tge.T
    sbii = cadine(lonx * RAD, latx * RAD, alt, time)
    vbeg = mat2tr(psivgx * RAD, thtvgx * RAD).T @ np.array([dvbe, 0.0, 0.0])
    return tig, sbii, vbeg


def _ready(mguidance):
    vehicle = type("V", (), {"store": StateStore()})()
    guidance = Cruise5Guidance()
    guidance.define(vehicle)
    store = vehicle.store
    tig, sbii, vbeg = _round3_state(14.7, 35.4, 7000.0, 90.0, 0.0, 200.0, 0.0)
    store.define(Field("time", 0.0, "real", "exec", "environment"))
    store.define(Field("grav", 9.81, "real", "out", "environment"))
    store.define(Field("tig", tig, "mat", "init/out", "newton"))
    store.define(Field("thtvgx", 0.0, "real", "init/out", "newton"))
    store.define(Field("vbeg", vbeg, "vec", "state", "newton"))
    store.define(Field("sbii", sbii, "vec", "state", "newton"))
    store.define(Field("philimx", 70.0, "real", "data", "control"))
    store.define(Field("anposlimx", 3.0, "real", "data", "control"))
    store.define(Field("anneglimx", -1.0, "real", "data", "control"))
    store.define(Field("allimx", 1.0, "real", "data", "control"))
    store.define(Field("phicx", 0.0, "real", "data", "control", ("scrn", "plot")))
    store.define(Field("alcomx", 0.0, "real", "data", "control", ("plot",)))
    store.define(Field("ancomx", 0.0, "real", "data", "control", ("plot",)))
    store.set("mguidance", mguidance)
    store.set("wp_lonx", 14.9)
    store.set("wp_latx", 35.4)
    store.set("wp_alt", 0.0)
    store.set("psifgx", 90.0)
    store.set("thtfgx", 0.0)
    store.set("line_gain", 1.0)
    store.set("nl_gain_fact", 0.6)
    store.set("decrement", 1000.0)
    store.set("point_gain", 1.0)
    return vehicle, guidance


def test_mguidance_30_alcomx_from_line_ancomx_stays_0():
    vehicle, guidance = _ready(mguidance=30)
    store = vehicle.store
    store.set("alcomx", 7.0)
    store.set("ancomx", 7.0)
    algv = guidance.guidance_line(vehicle)
    grav = store.get("grav")
    want_al, want_an = _clip(
        float(algv[1] / grav),
        0.0,
        store.get("anposlimx"),
        store.get("anneglimx"),
        store.get("allimx"),
    )
    guidance.execute(vehicle, SimContext(0.0, 0.05, 0.0, 0.0, None, 0))
    assert store.get("alcomx") == pytest.approx(want_al, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(want_an, rel=RTOL, abs=ATOL)
    assert want_an == 0.0


def test_mguidance_43_alcomx_from_point_ancomx_from_line_pitch():
    vehicle, guidance = _ready(mguidance=43)
    store = vehicle.store
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
    guidance.execute(vehicle, SimContext(0.0, 0.05, 0.0, 0.0, None, 0))
    assert store.get("alcomx") == pytest.approx(want_al, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(want_an, rel=RTOL, abs=ATOL)


def test_mguidance_0_does_not_write_commands():
    vehicle, guidance = _ready(mguidance=0)
    store = vehicle.store
    store.set("alcomx", 7.0)
    store.set("ancomx", 7.0)
    guidance.execute(vehicle, SimContext(0.0, 0.05, 0.0, 0.0, None, 0))
    assert store.get("alcomx") == 7.0
    assert store.get("ancomx") == 7.0


def test_mguidance_40_alcomx_from_point_ancomx_stays_0():
    vehicle, guidance = _ready(mguidance=40)
    store = vehicle.store
    store.set("alcomx", 7.0)
    store.set("ancomx", 7.0)
    apgv = guidance.guidance_point(vehicle)
    grav = store.get("grav")
    want_al, want_an = _clip(
        float(apgv[1] / grav),
        0.0,
        store.get("anposlimx"),
        store.get("anneglimx"),
        store.get("allimx"),
    )
    guidance.execute(vehicle, SimContext(0.0, 0.05, 0.0, 0.0, None, 0))
    assert store.get("alcomx") == pytest.approx(want_al, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(want_an, rel=RTOL, abs=ATOL)
    assert want_an == 0.0


def test_mguidance_33_alcomx_and_ancomx_from_line():
    vehicle, guidance = _ready(mguidance=33)
    store = vehicle.store
    algv = guidance.guidance_line(vehicle)
    grav = store.get("grav")
    want_al, want_an = _clip(
        float(algv[1] / grav),
        float(-algv[2] / grav),
        store.get("anposlimx"),
        store.get("anneglimx"),
        store.get("allimx"),
    )
    guidance.execute(vehicle, SimContext(0.0, 0.05, 0.0, 0.0, None, 0))
    assert store.get("alcomx") == pytest.approx(want_al, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(want_an, rel=RTOL, abs=ATOL)


def test_mguidance_66_alcomx_and_ancomx_from_pronav():
    vehicle, guidance = _ready(mguidance=66)
    store = vehicle.store
    store.define(Field("TBG", np.eye(3), "mat", "out", "control"))
    store.define(Field("WOEB", np.array([0.0, 0.01, 0.02]), "vec", "out", "seeker"))
    store.define(Field("closing_speed", 250.0, "real", "out", "seeker"))
    store.define(Field("UTBB", np.array([1.0, 0.0, 0.0]), "vec", "out", "seeker"))
    store.set("pronav_gain", 3.0)
    apnb = guidance.guidance_pronav(vehicle)
    grav = store.get("grav")
    want_al, want_an = _clip(
        float(apnb[1] / grav),
        float(-apnb[2] / grav),
        store.get("anposlimx"),
        store.get("anneglimx"),
        store.get("allimx"),
    )
    guidance.execute(vehicle, SimContext(0.0, 0.05, 0.0, 0.0, None, 0))
    assert store.get("alcomx") == pytest.approx(want_al, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(want_an, rel=RTOL, abs=ATOL)
    assert want_al != 0.0 or want_an != 0.0


def test_mguidance_70_arc_lateral_bank():
    vehicle, guidance = _ready(mguidance=70)
    store = vehicle.store
    store.define(Field("FSPV", np.array([2.0, 1.0, -12.0]), "vec", "out", "forces"))
    store.define(Field("dvbe", 200.0, "real", "init/out", "newton"))
    store.define(Field("alphax", 0.0, "real", "out", "aerodynamics"))
    store.define(Field("phimvx", 0.0, "real", "out", "control"))
    store.set("alcomx", 7.0)
    store.set("ancomx", 7.0)
    store.set("phicx", 12.0)
    want_phicx = guidance.guidance_arc(vehicle)
    guidance.execute(vehicle, SimContext(0.0, 0.05, 0.0, 0.0, None, 0))
    assert store.get("phicx") == pytest.approx(want_phicx, rel=RTOL, abs=ATOL)
    assert store.get("alcomx") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert want_phicx != 12.0


def test_unknown_mguidance_raises():
    vehicle, guidance = _ready(mguidance=-1)
    with pytest.raises(ValueError, match="unknown mguidance"):
        guidance.execute(vehicle, SimContext(0.0, 0.05, 0.0, 0.0, None, 0))
