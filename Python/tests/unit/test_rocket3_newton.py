"""ROCKET3 Newton D1 — Fortran MODULE.FOR D1/D1I + open-loop ALPHAX schedule."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cadac.constants import DEG, RAD, REARTH, WEII3
from cadac.io.scenario import load_scenario
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.earth import cadsph, cadtei, cadtge
from cadac.math.frames import mat2tr, polar_from_cart
from cadac.vehicles.round3.rocket3.newton import Rocket3Newton
from cadac.vehicles.round3.rocket3.vehicle import Rocket3

CASE = Path(__file__).resolve().parents[2] / "cases" / "rocket3" / "inlaunch.jsonc"

# INLAUNCH launch ICs (Fortran D1I inputs).
BLON = 4.8770
BLAT = 0.49620
BALT = 1.0000
DVBE = 1.0000
PSIVGX = 90.000
THTVGX = 90.000
GRAV = 9.80675445
DT = 0.01


def _matcar(dvbe: float, psivg: float, thtvg: float) -> np.ndarray:
    """Fortran MATCAR(VBEG, DVBE, PSIVG, THTVG)."""
    cel = np.cos(thtvg)
    return np.array(
        [
            dvbe * cel * np.cos(psivg),
            dvbe * cel * np.sin(psivg),
            -dvbe * np.sin(thtvg),
        ],
        dtype=float,
    )


def _ready(dt=DT):
    store = StateStore()
    vehicle = SimpleNamespace(store=store)
    newton = Rocket3Newton()
    newton.define(vehicle)
    store.define(Field("FSPV", (0.0, 0.0, 0.0), "vec", "out", "forces"))
    store.define(Field("grav", GRAV, "real", "out", "environment"))
    store.set("blon", BLON)
    store.set("blat", BLAT)
    store.set("balt", BALT)
    store.set("dvbe", DVBE)
    store.set("psivgx", PSIVGX)
    store.set("thtvgx", THTVGX)
    store.set("FSPV", np.array([2.0, 0.1, -0.5]))
    ctx = SimpleNamespace(int_step=dt, sim_time=0.0)
    newton.initialize(vehicle, ctx)
    return vehicle, newton, ctx


def test_name_is_newton():
    assert Rocket3Newton().name == "newton"


def test_open_loop_alphax_event_minus_5_5_stage():
    """INLAUNCH IF TIME > 10 → ALPHAX = -5.5° (open-loop pitch schedule)."""
    cfg = load_scenario(CASE)
    spec = cfg.vehicles[0]
    vehicle = Rocket3("rocket3", spec.events)
    vehicle.define()
    store = vehicle.store
    for name, value in spec.params.items():
        if name in store:
            store.set(name, value)
    if "time" not in store:
        store.define(Field("time", 0.0, "real", "exec", "environment"))
    assert store.get("alphax") == 0 or store.get("alphax") == 0.0

    store.set("time", 10.0)
    assert vehicle.events.evaluate(store) is False
    assert store.get("alphax") == 0 or store.get("alphax") == 0.0

    store.set("time", 10.01)
    assert vehicle.events.evaluate(store) is True
    assert store.get("alphax") == -5.5


def test_d1_one_step_integrates_trajectory():
    """Fortran D1: AI→VBIID; SBIID=VBII; one trapezoidal SBII/VBII step + geo update."""
    vehicle, newton, ctx = _ready()
    store = vehicle.store
    dt = ctx.int_step
    sbii0 = np.asarray(store.get("SBII"), dtype=float).copy()
    vbii0 = np.asarray(store.get("VBII"), dtype=float).copy()
    abii0 = np.asarray(store.get("abii"), dtype=float).copy()
    tgv0 = np.asarray(store.get("TGV"), dtype=float).copy()
    tig0 = np.asarray(store.get("TIG"), dtype=float).copy()
    weii = np.asarray(store.get("WEII"), dtype=float).copy()
    fspv = np.asarray(store.get("FSPV"), dtype=float).copy()
    grav = float(store.get("grav"))

    newton.execute(vehicle, ctx)

    grav_vec = np.array([0.0, 0.0, grav])
    abii_new = tig0 @ ((tgv0 @ fspv) + grav_vec)
    vbii_expected = integrate(abii_new, abii0, vbii0, dt)
    sbii_expected = integrate(vbii_expected, vbii0, sbii0, dt)

    np.testing.assert_allclose(store.get("abii"), abii_new, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("VBII"), vbii_expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("SBII"), sbii_expected, rtol=1e-12, atol=1e-14)

    tei = cadtei(ctx.sim_time)
    lon, lat, alt = cadsph(tei @ sbii_expected)
    tge = cadtge(lon, lat)
    tgi = tge @ tei
    vbeg = tgi @ (vbii_expected - weii @ sbii_expected)
    polar = polar_from_cart(vbeg)
    tig_expected = tgi.T
    tgv_expected = mat2tr(polar[1], polar[2]).T

    np.testing.assert_allclose(store.get("blon"), lon, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("blat"), lat, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("balt"), alt, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("VBEG"), vbeg, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("dvbe"), polar[0], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("psivgx"), polar[1] * DEG, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("thtvgx"), polar[2] * DEG, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("TIG"), tig_expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("TGV"), tgv_expected, rtol=1e-12, atol=1e-14)


def test_d1i_initializes_sbii_from_geographic():
    """Fortran D1I: SBIE from BLON/BLAT/BALT → SBII; VBEG via MATCAR."""
    vehicle, newton, ctx = _ready()
    store = vehicle.store
    sbie_expected = np.array(
        [
            (BALT + REARTH) * np.cos(BLAT) * np.cos(BLON),
            (BALT + REARTH) * np.cos(BLAT) * np.sin(BLON),
            (BALT + REARTH) * np.sin(BLAT),
        ]
    )
    np.testing.assert_allclose(store.get("SBII"), sbie_expected, rtol=1e-12, atol=1e-9)
    vbeg = _matcar(DVBE, PSIVGX * RAD, THTVGX * RAD)
    np.testing.assert_allclose(store.get("VBEG"), vbeg, rtol=1e-12, atol=1e-14)
    weii = np.zeros((3, 3))
    weii[0, 1] = -WEII3
    weii[1, 0] = WEII3
    np.testing.assert_allclose(store.get("WEII"), weii, rtol=1e-12, atol=1e-14)
    assert store.get("balt0") == BALT
