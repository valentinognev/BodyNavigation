from types import SimpleNamespace

import numpy as np

from cadac.constants import RAD
from cadac.eom.rotor import RotorAttitude, RotorTrajectory
from cadac.kernel.state import StateStore


def test_name_is_attitude():
    assert RotorAttitude().name == "attitude"


def test_rect_mr1_attitude_init_rates_in_dnu():
    s = StateStore()
    vehicle = SimpleNamespace(store=s)
    traj = RotorTrajectory()
    att = RotorAttitude()
    traj.define(vehicle)
    att.define(vehicle)
    for name, value in {
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
        "omega_rpm": 850.0,
        "moi_spin": 0.004,
        "betax": 0.0,
        "phix": 3.0,
        "ppx": 0.0,
        "psix": 0.0,
        "rrx": -40.0,
        "moi_trans": 0.0268,
        "nonlinear": 0,
    }.items():
        s.set(name, value)
    traj.initialize(vehicle, None)
    att.initialize(vehicle, None)
    tau = s.get("tau")
    np.testing.assert_allclose(s.get("beta"), 0.0, rtol=1e-12)
    np.testing.assert_allclose(s.get("phi"), 3.0 * RAD, rtol=1e-12)
    np.testing.assert_allclose(s.get("phid"), 0.0, rtol=1e-12)
    np.testing.assert_allclose(s.get("psi"), 0.0, rtol=1e-12)
    np.testing.assert_allclose(s.get("psid"), -40.0 * RAD * tau, rtol=1e-12)
