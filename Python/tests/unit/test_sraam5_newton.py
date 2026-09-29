"""SRAAM5 Newton D1 — Fortran MODULE.FOR kinematics slice."""

from types import SimpleNamespace

import numpy as np

from cadac.constants import AGRAV, DEG
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr
from cadac.vehicles.flat5.sraam5.newton import Sraam5Newton


def _matcar(dvbe: float, psivl: float, thtvl: float) -> np.ndarray:
    """Fortran MATCAR(VBEL, DVBE, PSIVL, THTVL)."""
    cel = np.cos(thtvl)
    return np.array(
        [
            dvbe * cel * np.cos(psivl),
            dvbe * cel * np.sin(psivl),
            -dvbe * np.sin(thtvl),
        ],
        dtype=float,
    )


def _ready(dt=0.0123):
    store = StateStore()
    vehicle = SimpleNamespace(store=store)
    newton = Sraam5Newton()
    newton.define(vehicle)
    store.define(Field("FSPV", (0.0, 0.0, 0.0), "vec", "out", "forces"))
    # Plant polar / position like G1I missile init (level flight).
    store.set("dvbe", 240.0)
    store.set("psivl", -np.pi)
    store.set("thtvl", 0.0)
    store.set("SBELS", np.array([1000.0, 2000.0, -5000.0]))
    store.set("SBEL", np.array([1000.0, 2000.0, -5000.0]))
    store.set("dvbed", 0.0)
    store.set("psivld", 0.0)
    store.set("thtvld", 0.0)
    store.set("SBELSD", np.zeros(3))
    store.set("gndtck", 0.0)
    store.set("icoor", 0)
    store.set("FSPV", np.array([1.5, 0.2, -0.5]))
    ctx = SimpleNamespace(int_step=dt)
    return vehicle, newton, ctx


def test_name_is_newton():
    assert Sraam5Newton().name == "newton"


def test_d1_one_step_sbel_vbel_kinematics_slice():
    """Fortran D1: MATCAR→VBEL; SBELSD=VBEL; SBEL←SBELS; one trapezoidal SBELS step."""
    vehicle, newton, ctx = _ready()
    store = vehicle.store
    dt = ctx.int_step
    dvbe0 = float(store.get("dvbe"))
    psivl0 = float(store.get("psivl"))
    thtvl0 = float(store.get("thtvl"))
    sbels0 = np.asarray(store.get("SBELS"), dtype=float).copy()
    sbelsd0 = np.asarray(store.get("SBELSD"), dtype=float).copy()
    dvbed0 = float(store.get("dvbed"))
    psivld0 = float(store.get("psivld"))
    thtvld0 = float(store.get("thtvld"))
    fspv = np.asarray(store.get("FSPV"), dtype=float)

    newton.execute(vehicle, ctx)

    # Derivatives at planted polar (Fortran D1 EOMs).
    dvbed_new = fspv[0] - np.sin(thtvl0) * AGRAV
    psivld_new = fspv[1] / (dvbe0 * np.cos(thtvl0))
    thtvld_new = -(fspv[2] + np.cos(thtvl0) * AGRAV) / dvbe0

    vbel_slice = _matcar(dvbe0, psivl0, thtvl0)
    sbels_expected = integrate(vbel_slice, sbelsd0, sbels0, dt)
    dvbe_expected = integrate(dvbed_new, dvbed0, dvbe0, dt)
    psivl_expected = integrate(psivld_new, psivld0, psivl0, dt)
    thtvl_expected = integrate(thtvld_new, thtvld0, thtvl0, dt)

    # Kinematics slice outputs use pre-integration polar (Fortran D1 order).
    np.testing.assert_allclose(store.get("VBEL"), vbel_slice, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("SBELSD"), vbel_slice, rtol=1e-12, atol=1e-14)
    # After folded executive integrate: SBEL mirrors advanced SBELS.
    np.testing.assert_allclose(store.get("SBELS"), sbels_expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("SBEL"), sbels_expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("dvbe"), dvbe_expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("psivl"), psivl_expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("thtvl"), thtvl_expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("hbe"), -sbels_expected[2], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("psivlx"), psivl0 * DEG, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("thtvlx"), thtvl0 * DEG, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        store.get("TVL"), mat2tr(psivl0, thtvl0), rtol=1e-12, atol=1e-14
    )
    wvev_expected = np.array(
        [
            -np.sin(thtvl0) * psivld_new,
            thtvld_new,
            np.cos(thtvl0) * psivld_new,
        ]
    )
    np.testing.assert_allclose(store.get("WVEV"), wvev_expected, rtol=1e-12, atol=1e-14)
