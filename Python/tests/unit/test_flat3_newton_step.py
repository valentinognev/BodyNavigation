from types import SimpleNamespace

import numpy as np

from cadac.constants import DEG
from cadac.env.gravity import gravity
from cadac.eom.flat3 import Flat3Newton
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr, polar_from_cart


def _vehicle_after_init():
    s = StateStore()
    vehicle = SimpleNamespace(store=s)
    newton = Flat3Newton()
    newton.define(vehicle)
    s.set("sbel1", 0.0)
    s.set("sbel2", 0.0)
    s.set("sbel3", -3500.0)
    s.set("dvbe", 200.0)
    s.set("psivlx", 0.0)
    s.set("thtvlx", 0.0)
    s.define(Field("FSPV", (0.0, 0.0, 0.0), "vec", "out", "forces"))
    s.define(Field("grav", gravity(3500.0), "real", "out", "environment"))
    newton.initialize(vehicle, None)
    return vehicle, newton


def test_execute_zero_fspv_falls_and_abel_matches_next_acc():
    vehicle, newton = _vehicle_after_init()
    s = vehicle.store
    tbl = s.get("TBL").copy()
    fspv = s.get("FSPV").copy()
    grav = s.get("grav")
    sbel_old = s.get("SBEL").copy()
    vbel_old = s.get("VBEL").copy()
    abel_old = s.get("ABEL").copy()
    dt = 0.05
    ctx = SimpleNamespace(int_step=dt)

    newton.execute(vehicle, ctx)

    assert s.get("SBEL")[2] > sbel_old[2]
    assert s.get("SBEL")[2] < 0.0

    next_acc = tbl.T @ fspv + np.array([0.0, 0.0, grav])
    np.testing.assert_allclose(s.get("ABEL"), next_acc, rtol=1e-12, atol=1e-14)

    next_vel = integrate(next_acc, abel_old, vbel_old, dt)
    sbel_expected = integrate(next_vel, vbel_old, sbel_old, dt)
    np.testing.assert_allclose(s.get("VBEL"), next_vel, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(s.get("SBEL"), sbel_expected, rtol=1e-12, atol=1e-14)

    polar = polar_from_cart(next_vel)
    tvl = mat2tr(polar[1], polar[2])
    tbv = np.eye(3)
    tbl_expected = tbv @ tvl
    np.testing.assert_allclose(s.get("TVL"), tvl, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(s.get("TBV"), tbv, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(s.get("TBL"), tbl_expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(s.get("dvbe"), polar[0], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(s.get("psivl"), polar[1], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(s.get("thtvl"), polar[2], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(s.get("psivlx"), polar[1] * DEG, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(s.get("thtvlx"), polar[2] * DEG, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(s.get("alt"), -sbel_expected[2], rtol=1e-12, atol=1e-14)
