from types import SimpleNamespace

import numpy as np

from cadac.constants import DEG, RAD
from cadac.eom.rotor import RotorEnvironment, RotorTrajectory
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import StateStore
from cadac.math.frames import mat2tr


def _rect_ready(hbg=0.0):
    s = StateStore()
    vehicle = SimpleNamespace(store=s, health=1, name="RECT.MR1")
    env = RotorEnvironment()
    traj = RotorTrajectory()
    env.define(vehicle)
    traj.define(vehicle)
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
        "hbg": hbg,
        "sbel1": 0.0,
        "sbel2": 0.0,
        "omega_rpm": 850.0,
        "moi_spin": 0.004,
    }.items():
        s.set(name, value)
    traj.initialize(vehicle, None)
    return vehicle, env, traj


def test_one_dnt_step_matches_cpp_velocityxd():
    vehicle, env, traj = _rect_ready()
    dt = 0.0001
    ctx = SimContext(
        sim_time=0.0,
        int_step=dt,
        event_time=0.0,
        out_fact=0.0,
        combus=[Packet(name="RECT.MR1", type="ROTOR", status=1, vars={})],
        vehicle_slot=0,
    )
    env.execute(vehicle, ctx)
    s = vehicle.store
    velocityx = s.get("velocityx")
    gamma = s.get("gamma")
    omegax = s.get("omegax")
    cd, clw, cma, cmdw = 1.31, 2.51, 0.508, -0.45
    mass, ref_area, ref_length = 1.5, 0.0468, 0.0625
    velocity_ss = s.get("velocity_ss")
    rho = s.get("rho")
    grav = s.get("grav")
    tau = 2 * mass / (rho * ref_area * velocity_ss)
    mu = 2 * mass / (rho * ref_area * ref_length)
    moi_spinx = 0.004 / (ref_length**2 * mu**2 * mass)
    velocityxd_new = -cd * velocityx * velocityx - tau * grav * np.sin(gamma) / velocity_ss
    vx = integrate(velocityxd_new, 0.0, velocityx, dt)
    traj.execute(vehicle, ctx)
    np.testing.assert_allclose(s.get("velocityx"), vx, rtol=1e-12)
    np.testing.assert_allclose(s.get("moi_spinx"), moi_spinx, rtol=1e-12)
    np.testing.assert_allclose(s.get("time"), tau * 0.0, rtol=1e-12)
    assert s.get("hbe") > 990.0
    assert vehicle.health == 1


def test_ground_impact_sets_health_no_sys_exit():
    vehicle, env, traj = _rect_ready(hbg=10000.0)
    ctx = SimContext(
        sim_time=0.0,
        int_step=0.0001,
        event_time=0.0,
        out_fact=0.0,
        combus=[Packet(name="RECT.MR1", type="ROTOR", status=1, vars={})],
        vehicle_slot=0,
    )
    env.execute(vehicle, ctx)
    traj.execute(vehicle, ctx)
    assert vehicle.health == 0
    assert ctx.combus[0].status == 0


def test_sbel_dt_is_int_step_times_tau_and_time_at_nonzero_sim_time():
    vehicle, env, traj = _rect_ready()
    dt = 0.0001
    sim_time = 0.01
    ctx = SimContext(
        sim_time=sim_time,
        int_step=dt,
        event_time=0.0,
        out_fact=0.0,
        combus=[Packet(name="RECT.MR1", type="ROTOR", status=1, vars={})],
        vehicle_slot=0,
    )
    env.execute(vehicle, ctx)
    s = vehicle.store
    sbel0 = np.array(s.get("SBEL"), dtype=float)
    sbeld0 = np.array(s.get("SBELD"), dtype=float)
    velocityx = s.get("velocityx")
    velocityxd = s.get("velocityxd")
    gamma = s.get("gamma")
    gammaxd = s.get("gammaxd")
    omegax = s.get("omegax")
    omegaxd = s.get("omegaxd")
    cd, clw, cma, cmdw = 1.31, 2.51, 0.508, -0.45
    mass, ref_area, ref_length = 1.5, 0.0468, 0.0625
    velocity_ss = s.get("velocity_ss")
    rho, grav, psivlx = s.get("rho"), s.get("grav"), s.get("psivlx")
    tau = 2 * mass / (rho * ref_area * velocity_ss)
    mu = 2 * mass / (rho * ref_area * ref_length)
    moi_spinx = 0.004 / (ref_length**2 * mu**2 * mass)
    velocityxd_new = -cd * velocityx**2 - tau * grav * np.sin(gamma) / velocity_ss
    vx = integrate(velocityxd_new, velocityxd, velocityx, dt)
    gammaxd_new = clw * omegax / mu - tau * grav * np.cos(gamma) / (velocity_ss * vx)
    gamma_new = integrate(gammaxd_new, gammaxd, gamma, dt)
    omegaxd_new = (
        cma * vx**2 / (mu * moi_spinx)
        + cmdw * vx * omegax / (mu**2 * moi_spinx)
    )
    dvbe = vx * velocity_ss
    thtvlx = gamma_new * DEG
    tvl = mat2tr(psivlx * RAD, thtvlx * RAD)
    vbel = tvl.T @ np.array([dvbe, 0.0, 0.0])
    sbel_want = integrate(vbel, sbeld0, sbel0, dt * tau)
    sbel_wrong_dt = integrate(vbel, sbeld0, sbel0, dt)
    assert not np.allclose(sbel_want, sbel_wrong_dt, rtol=1e-9)
    traj.execute(vehicle, ctx)
    np.testing.assert_allclose(s.get("SBEL"), sbel_want, rtol=1e-12)
    np.testing.assert_allclose(s.get("time"), tau * sim_time, rtol=1e-12)
    np.testing.assert_allclose(s.get("sim_time"), sim_time, rtol=1e-12)
