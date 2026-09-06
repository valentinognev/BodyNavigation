from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import DEG
from cadac.eom.rotor import RotorAttitude, RotorEnvironment, RotorTrajectory
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import StateStore


def _attitude_ready(nonlinear=0):
    s = StateStore()
    vehicle = SimpleNamespace(store=s, health=1, name="RECT.MR1")
    env, traj, att = RotorEnvironment(), RotorTrajectory(), RotorAttitude()
    env.define(vehicle)
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
        "hbg": 0.0,
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
        "nonlinear": nonlinear,
        "cyb": -3.82,
        "clwb": -0.357,
        "clp": -5.82,
        "cnb": -0.737,
        "cnr": -13.8,
        "cyb3": 0.0,
        "clwb3": 0.0,
        "clp3": 0.0,
        "clwb2p": 0.0,
        "clwbp2": 0.0,
        "cnb3": 0.0,
        "cnr3": 0.0,
        "cnb2r": 0.0,
        "cnbr2": 0.0,
    }.items():
        s.set(name, value)
    traj.initialize(vehicle, None)
    att.initialize(vehicle, None)
    return vehicle, env, traj, att


def test_nonlinear0_one_step_beta_matches_cpp():
    vehicle, env, traj, att = _attitude_ready(0)
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
    traj.execute(vehicle, ctx)
    s = vehicle.store
    beta, betad = s.get("beta"), s.get("betad")
    phi, psi = s.get("phi"), s.get("psi")
    psid = s.get("psid")
    velocityx, velocityxd = s.get("velocityx"), s.get("velocityxd")
    gamma, tau = s.get("gamma"), s.get("tau")
    grav, velocity_ss = s.get("grav"), s.get("velocity_ss")
    cyb = -3.82
    betad_new = (
        (-velocityxd / velocityx + velocityx * cyb) * beta
        - psid
        + tau * grav * np.cos(gamma) / (velocityx * velocity_ss) * phi
        + tau * grav * np.sin(gamma) / (velocityx * velocity_ss) * psi
    )
    beta_want = integrate(betad_new, betad, beta, dt)
    att.execute(vehicle, ctx)
    np.testing.assert_allclose(s.get("beta"), beta_want, rtol=1e-12)
    assert np.isfinite(s.get("phix"))
    assert s.get("nonlinear") == 0


def test_nonlinear1_adds_cyb3_term():
    vehicle, env, traj, att = _attitude_ready(1)
    vehicle.store.set("cyb3", 1.0)
    vehicle.store.set("beta", 0.1)
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
    traj.execute(vehicle, ctx)
    s = vehicle.store
    beta, betad = s.get("beta"), s.get("betad")
    phi, psi, psid = s.get("phi"), s.get("psi"), s.get("psid")
    velocityx, velocityxd = s.get("velocityx"), s.get("velocityxd")
    gamma, tau = s.get("gamma"), s.get("tau")
    grav, velocity_ss = s.get("grav"), s.get("velocity_ss")
    betad_new = (
        (-velocityxd / velocityx + velocityx * (-3.82)) * beta
        - psid
        + tau * grav * np.cos(gamma) / (velocityx * velocity_ss) * phi
        + tau * grav * np.sin(gamma) / (velocityx * velocity_ss) * psi
        + 1 * (velocityx * 1.0 * beta**3 / 6.0)
    )
    beta_want = integrate(betad_new, betad, beta, dt)
    att.execute(vehicle, ctx)
    np.testing.assert_allclose(s.get("beta"), beta_want, rtol=1e-12)


def test_nonlinear_2_raises():
    vehicle, env, traj, att = _attitude_ready(2)
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
    with pytest.raises(ValueError, match="nonlinear"):
        att.execute(vehicle, ctx)


def test_phi_uses_new_beta_then_psidd_uses_new_phid():
    vehicle, env, traj, att = _attitude_ready(0)
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
    traj.execute(vehicle, ctx)
    s = vehicle.store
    beta, betad = s.get("beta"), s.get("betad")
    phi, phid, phidd = s.get("phi"), s.get("phid"), s.get("phidd")
    psi, psid, psidd = s.get("psi"), s.get("psid"), s.get("psidd")
    velocityx, velocityxd = s.get("velocityx"), s.get("velocityxd")
    gamma, tau = s.get("gamma"), s.get("tau")
    grav, velocity_ss = s.get("grav"), s.get("velocity_ss")
    omegax = s.get("omegax")
    mu, moi_spinx = s.get("mu"), s.get("moi_spinx")
    moi_transx = 0.0268 / (0.0625**2 * mu**2 * 1.5)
    betad_new = (
        (-velocityxd / velocityx + velocityx * (-3.82)) * beta
        - psid
        + tau * grav * np.cos(gamma) / (velocityx * velocity_ss) * phi
        + tau * grav * np.sin(gamma) / (velocityx * velocity_ss) * psi
    )
    beta_new = integrate(betad_new, betad, beta, dt)
    phidd_new = (
        velocityx * omegax * (-0.357) / (mu * mu * moi_transx) * beta_new
        + velocityx * (-5.82) / (mu * mu * moi_transx) * phid
        + moi_spinx * omegax / moi_transx * psid
    )
    phid_new = integrate(phidd_new, phidd, phid, dt)
    phi_want = integrate(phid_new, phid_new, phi, dt)
    psidd_new = (
        velocityx * velocityx * (-0.737) / (mu * moi_transx) * beta_new
        - moi_spinx * omegax / moi_transx * phid_new
        + velocityx * (-13.8) / (mu * mu * moi_transx) * psid
    )
    psid_new = integrate(psidd_new, psidd, psid, dt)
    psi_want = integrate(psid_new, psid_new, psi, dt)
    att.execute(vehicle, ctx)
    np.testing.assert_allclose(s.get("beta"), beta_new, rtol=1e-12)
    np.testing.assert_allclose(s.get("phid"), phid_new, rtol=1e-12)
    np.testing.assert_allclose(s.get("phi"), phi_want, rtol=1e-12)
    np.testing.assert_allclose(s.get("psid"), psid_new, rtol=1e-12)
    np.testing.assert_allclose(s.get("psi"), psi_want, rtol=1e-12)
    np.testing.assert_allclose(s.get("ppx"), phid_new * DEG / tau, rtol=1e-12)
