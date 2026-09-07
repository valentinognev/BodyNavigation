import math
from types import SimpleNamespace

import numpy as np

from cadac.constants import AGRAV, DEG, RAD
from cadac.env.us76 import atmosphere76
from cadac.eom.rotor import RPM, RHO_SL, RotorTrajectory
from cadac.kernel.state import StateStore
from cadac.math.frames import mat2tr


def test_name_is_trajectory():
    assert RotorTrajectory().name == "trajectory"


def test_rect_mr1_init_sbel_velocityx_gamma_omegax():
    s = StateStore()
    vehicle = SimpleNamespace(store=s)
    traj = RotorTrajectory()
    traj.define(vehicle)
    params = {
        "cd": 1.31,
        "cmdw": -0.45,
        "clw": 2.51,
        "cma": 0.508,
        "mass": 1.5,
        "ref_area": 0.0468,
        "ref_length": 0.0625,
        "dvbe": 16.6,
        "psivlx": 0.0,
        "thtvlx": -77.0,
        "hbe": 1000.0,
        "sbel1": 0.0,
        "sbel2": 0.0,
        "sbel3": 99.0,
        "omega_rpm": 850.0,
        "moi_spin": 0.004,
    }
    for name, value in params.items():
        s.set(name, value)
    traj.initialize(vehicle, None)

    gamma_ss = math.atan(1.31 * (-0.45) / (2.51 * 0.508))
    velocity_ss = math.sqrt(
        2 * AGRAV * 1.5 * abs(math.sin(gamma_ss)) / (RHO_SL * 0.0468 * 1.31)
    )
    omega_ss = -velocity_ss * 0.508 / (0.0625 * (-0.45))
    rho, _, _ = atmosphere76(1000.0)
    tau = 2 * 1.5 / (rho * 0.0468 * velocity_ss)
    np.testing.assert_allclose(s.get("gamma_ss"), gamma_ss, rtol=1e-12)
    np.testing.assert_allclose(s.get("velocity_ss"), velocity_ss, rtol=1e-12)
    np.testing.assert_allclose(s.get("omega_ss"), omega_ss, rtol=1e-12)
    np.testing.assert_allclose(s.get("SBEL"), np.array([0.0, 0.0, -1000.0]), rtol=1e-12)
    np.testing.assert_allclose(s.get("velocityx"), 16.6 / velocity_ss, rtol=1e-12)
    np.testing.assert_allclose(s.get("gamma"), -77.0 * RAD, rtol=1e-12)
    np.testing.assert_allclose(s.get("omegax"), (850.0 / RPM) * tau, rtol=1e-12)
    np.testing.assert_allclose(s.get("tau"), tau, rtol=1e-12)
    tvl = mat2tr(0.0, -77.0 * RAD)
    vbel = tvl.T @ np.array([16.6, 0.0, 0.0])
    np.testing.assert_allclose(s.get("VBEL"), vbel, rtol=1e-12)
    assert s.get("time") == 0.0
    assert "plot" in s.field("sim_time").outputs
    assert "plot" in s.field("time").outputs
