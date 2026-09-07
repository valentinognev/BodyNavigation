import math

import numpy as np

from cadac.constants import AGRAV, DEG, R, RAD
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr

RPM = 9.5493
RHO_SL = 1.225


class RotorEnvironment:
    name = "environment"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("mwind", 0, "int", "data", "environment"),
            Field("press", 0.0, "real", "out", "environment"),
            Field("rho", 0.0, "real", "out", "environment"),
            Field("vsound", 0.0, "real", "diag", "environment"),
            Field("grav", 0.0, "real", "out", "environment"),
            Field("vmach", 0.0, "real", "out", "environment", ("scrn", "plot", "com")),
            Field("pdynmc", 0.0, "real", "out", "environment", ("scrn", "plot")),
            Field("tempk", 0.0, "real", "out", "environment"),
            Field("dvae", 0.0, "real", "data", "environment"),
            Field("dvael", 0.0, "real", "data", "environment"),
            Field("waltl", 0.0, "real", "data", "environment"),
            Field("dvaeh", 0.0, "real", "data", "environment"),
            Field("walth", 0.0, "real", "data", "environment"),
            Field("vaed3", 0.0, "real", "data", "environment"),
            Field("psiwdx", 0.0, "real", "data", "environment"),
            Field("twind", 0.1, "real", "data", "environment"),
            Field("VAELS", zeros3, "vec", "state", "environment"),
            Field("VAELSD", zeros3, "vec", "state", "environment"),
            Field("VAEL", zeros3, "vec", "out", "environment"),
            Field("dvba", 0.0, "real", "out", "environment"),
            Field("VBAL", zeros3, "vec", "out", "environment"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mwind = store.get("mwind")
        if mwind != 0:
            raise ValueError(f"unknown mwind {mwind}")
        hbe = store.get("hbe")
        vbel = store.get("VBEL")
        rho, press, tempk = atmosphere76(hbe)
        vsound = math.sqrt(1.4 * R * tempk)
        vael = np.zeros(3)
        vbal = vbel - vael
        dvba = float(np.linalg.norm(vbal))
        vmach = abs(dvba / vsound)
        pdynmc = 0.5 * rho * dvba * dvba
        store.set("grav", gravity(hbe))
        store.set("rho", rho)
        store.set("press", press)
        store.set("tempk", tempk)
        store.set("vsound", vsound)
        store.set("VAEL", vael)
        store.set("VBAL", vbal)
        store.set("dvba", dvba)
        store.set("vmach", vmach)
        store.set("pdynmc", pdynmc)

    def terminate(self, vehicle, ctx):
        pass


class RotorTrajectory:
    name = "trajectory"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("time", 0.0, "real", "diag", "trajectory", ("scrn", "plot")),
            Field("sim_time", 0.0, "real", "exec", "trajectory", ("scrn", "plot")),
            Field("cd", 0.0, "real", "data", "trajectory"),
            Field("cmdw", 0.0, "real", "data", "trajectory"),
            Field("clw", 0.0, "real", "data", "trajectory"),
            Field("cma", 0.0, "real", "data", "trajectory"),
            Field("mass", 0.0, "real", "data", "trajectory"),
            Field("ref_area", 0.0, "real", "data", "trajectory"),
            Field("ref_length", 0.0, "real", "data", "trajectory"),
            Field("velocity_ss", 0.0, "real", "out", "trajectory"),
            Field("gamma_ss", 0.0, "real", "out", "trajectory"),
            Field("omega_ss", 0.0, "real", "out", "trajectory"),
            Field("dvbe", 0.0, "real", "init/diag", "trajectory", ("scrn", "plot")),
            Field("psivlx", 0.0, "real", "init/diag", "trajectory", ("scrn", "plot")),
            Field("thtvlx", 0.0, "real", "init/diag", "trajectory", ("scrn", "plot")),
            Field("hbg", 0.0, "real", "data", "trajectory"),
            Field("hbe", 0.0, "real", "init/diag", "trajectory", ("scrn", "plot")),
            Field("omega", 0.0, "real", "init/out", "trajectory"),
            Field("sbel1", 0.0, "real", "init", "trajectory"),
            Field("sbel2", 0.0, "real", "init", "trajectory"),
            Field("sbel3", 0.0, "real", "init", "trajectory"),
            Field("velocityx", 0.0, "real", "state", "trajectory", ("plot",)),
            Field("velocityxd", 0.0, "real", "state", "trajectory"),
            Field("gamma", 0.0, "real", "state", "trajectory", ("plot",)),
            Field("gammaxd", 0.0, "real", "state", "trajectory"),
            Field("omegax", 0.0, "real", "state", "trajectory", ("plot",)),
            Field("omegaxd", 0.0, "real", "state", "trajectory"),
            Field("moi_spin", 0.0, "real", "out", "trajectory"),
            Field("moi_spinx", 0.0, "real", "data", "trajectory"),
            Field("SBEL", zeros3, "vec", "state", "trajectory", ("plot",)),
            Field("SBELD", zeros3, "vec", "state", "trajectory"),
            Field("VBEL", zeros3, "vec", "diag", "trajectory", ("plot",)),
            Field("omega_rpm", 0.0, "real", "diag", "trajectory", ("scrn", "plot")),
            Field("tau", 0.0, "real", "out", "trajectory"),
            Field("mu", 0.0, "real", "out", "trajectory"),
            Field("tpsp_ratio", 0.0, "real", "diag", "trajectory", ("plot", "scrn")),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        cd = store.get("cd")
        cmdw = store.get("cmdw")
        clw = store.get("clw")
        cma = store.get("cma")
        mass = store.get("mass")
        ref_area = store.get("ref_area")
        ref_length = store.get("ref_length")
        dvbe = store.get("dvbe")
        psivlx = store.get("psivlx")
        thtvlx = store.get("thtvlx")
        hbe = store.get("hbe")
        sbel1 = store.get("sbel1")
        sbel2 = store.get("sbel2")
        omega_rpm = store.get("omega_rpm")

        gamma_ss = math.atan(cd * cmdw / (clw * cma))
        velocity_ss = math.sqrt(
            2 * AGRAV * mass * abs(math.sin(gamma_ss)) / (RHO_SL * ref_area * cd)
        )
        omega_ss = -velocity_ss * cma / (ref_length * cmdw)
        sbel = np.array([sbel1, sbel2, -hbe])
        tvl = mat2tr(psivlx * RAD, thtvlx * RAD)
        vbel = tvl.T @ np.array([dvbe, 0.0, 0.0])
        velocityx = dvbe / velocity_ss
        gamma = thtvlx * RAD
        rho, _, _ = atmosphere76(hbe)
        tau = 2 * mass / (rho * ref_area * velocity_ss)
        omegax = (omega_rpm / RPM) * tau

        store.set("velocity_ss", velocity_ss)
        store.set("gamma_ss", gamma_ss)
        store.set("omega_ss", omega_ss)
        store.set("tau", tau)
        store.set("velocityx", velocityx)
        store.set("gamma", gamma)
        store.set("omegax", omegax)
        store.set("SBEL", sbel)
        store.set("time", 0.0)
        store.set("hbe", hbe)
        store.set("VBEL", vbel)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        cd = store.get("cd")
        cmdw = store.get("cmdw")
        clw = store.get("clw")
        cma = store.get("cma")
        mass = store.get("mass")
        ref_area = store.get("ref_area")
        ref_length = store.get("ref_length")
        psivlx = store.get("psivlx")
        hbg = store.get("hbg")
        moi_spin = store.get("moi_spin")
        velocity_ss = store.get("velocity_ss")
        rho = store.get("rho")
        grav = store.get("grav")
        velocityx = store.get("velocityx")
        velocityxd = store.get("velocityxd")
        gamma = store.get("gamma")
        gammaxd = store.get("gammaxd")
        omegax = store.get("omegax")
        omegaxd = store.get("omegaxd")
        sbel = store.get("SBEL")
        sbeld = store.get("SBELD")
        int_step = ctx.int_step

        tau = 2 * mass / (rho * ref_area * velocity_ss)
        mu = 2 * mass / (rho * ref_area * ref_length)
        moi_spinx = moi_spin / (ref_length * ref_length * mu * mu * mass)

        velocityxd_new = (
            -cd * velocityx * velocityx - tau * grav * math.sin(gamma) / velocity_ss
        )
        velocityx = integrate(velocityxd_new, velocityxd, velocityx, int_step)
        velocityxd = velocityxd_new

        gammaxd_new = clw * omegax / mu - tau * grav * math.cos(gamma) / (
            velocity_ss * velocityx
        )
        gamma = integrate(gammaxd_new, gammaxd, gamma, int_step)
        gammaxd = gammaxd_new

        omegaxd_new = (
            cma * velocityx * velocityx / (mu * moi_spinx)
            + cmdw * velocityx * omegax / (mu * mu * moi_spinx)
        )
        omegax = integrate(omegaxd_new, omegaxd, omegax, int_step)
        omegaxd = omegaxd_new

        dvbe = velocityx * velocity_ss
        thtvlx = gamma * DEG
        omega = omegax / tau
        omega_rpm = omega * RPM

        tvl = mat2tr(psivlx * RAD, thtvlx * RAD)
        vbel = tvl.T @ np.array([dvbe, 0.0, 0.0])
        sbeld_new = vbel
        sbel = integrate(sbeld_new, sbeld, sbel, int_step * tau)
        sbeld = sbeld_new

        hbe = -float(sbel[2])
        tpsp_ratio = omega * ref_length / dvbe
        time = tau * ctx.sim_time

        if hbe < hbg:
            vehicle.health = 0
            ctx.combus[ctx.vehicle_slot].status = 0

        store.set("velocityx", velocityx)
        store.set("velocityxd", velocityxd)
        store.set("gamma", gamma)
        store.set("gammaxd", gammaxd)
        store.set("omegax", omegax)
        store.set("omegaxd", omegaxd)
        store.set("SBEL", sbel)
        store.set("SBELD", sbeld)
        store.set("moi_spinx", moi_spinx)
        store.set("tau", tau)
        store.set("mu", mu)
        store.set("time", time)
        store.set("sim_time", ctx.sim_time)
        store.set("dvbe", dvbe)
        store.set("thtvlx", thtvlx)
        store.set("hbe", hbe)
        store.set("omega", omega)
        store.set("VBEL", vbel)
        store.set("omega_rpm", omega_rpm)
        store.set("tpsp_ratio", tpsp_ratio)

    def terminate(self, vehicle, ctx):
        pass


class RotorAttitude:
    name = "attitude"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("nonlinear", 0, "int", "data", "attitude"),
            Field("moi_trans", 0.0, "real", "data", "attitude"),
            Field("cyb", 0.0, "real", "data", "attitude"),
            Field("cyb3", 0.0, "real", "data", "attitude"),
            Field("clwb", 0.0, "real", "data", "attitude"),
            Field("clp", 0.0, "real", "data", "attitude"),
            Field("clwb3", 0.0, "real", "data", "attitude"),
            Field("clp3", 0.0, "real", "data", "attitude"),
            Field("clwb2p", 0.0, "real", "data", "attitude"),
            Field("clwbp2", 0.0, "real", "data", "attitude"),
            Field("cnb", 0.0, "real", "data", "attitude"),
            Field("cnr", 0.0, "real", "data", "attitude"),
            Field("cnb3", 0.0, "real", "data", "attitude"),
            Field("cnr3", 0.0, "real", "data", "attitude"),
            Field("cnb2r", 0.0, "real", "data", "attitude"),
            Field("cnbr2", 0.0, "real", "data", "attitude"),
            Field("beta", 0.0, "real", "state", "attitude"),
            Field("betad", 0.0, "real", "state", "attitude"),
            Field("phi", 0.0, "real", "state", "attitude"),
            Field("phid", 0.0, "real", "state", "attitude"),
            Field("phidd", 0.0, "real", "state", "attitude"),
            Field("psi", 0.0, "real", "state", "attitude"),
            Field("psid", 0.0, "real", "state", "attitude"),
            Field("psidd", 0.0, "real", "state", "attitude"),
            Field("betax", 0.0, "real", "diag", "attitude", ("scrn", "plot")),
            Field("phix", 0.0, "real", "diag", "attitude", ("scrn", "plot")),
            Field("ppx", 0.0, "real", "diag", "attitude", ("scrn", "plot")),
            Field("psix", 0.0, "real", "diag", "attitude", ("scrn", "plot")),
            Field("rrx", 0.0, "real", "diag", "attitude", ("scrn", "plot")),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        betax = store.get("betax")
        phix = store.get("phix")
        ppx = store.get("ppx")
        psix = store.get("psix")
        rrx = store.get("rrx")
        tau = store.get("tau")
        store.set("beta", betax * RAD)
        store.set("phi", phix * RAD)
        store.set("phid", ppx * RAD * tau)
        store.set("psi", psix * RAD)
        store.set("psid", rrx * RAD * tau)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        nonlinear = store.get("nonlinear")
        if nonlinear not in (0, 1):
            raise ValueError(f"unknown nonlinear {nonlinear}")
        moi_trans = store.get("moi_trans")
        cyb = store.get("cyb")
        cyb3 = store.get("cyb3")
        clwb = store.get("clwb")
        clp = store.get("clp")
        clwb3 = store.get("clwb3")
        clp3 = store.get("clp3")
        clwb2p = store.get("clwb2p")
        clwbp2 = store.get("clwbp2")
        cnb = store.get("cnb")
        cnr = store.get("cnr")
        cnb3 = store.get("cnb3")
        cnr3 = store.get("cnr3")
        cnb2r = store.get("cnb2r")
        cnbr2 = store.get("cnbr2")
        grav = store.get("grav")
        mass = store.get("mass")
        ref_length = store.get("ref_length")
        velocity_ss = store.get("velocity_ss")
        velocityx = store.get("velocityx")
        velocityxd = store.get("velocityxd")
        gamma = store.get("gamma")
        omegax = store.get("omegax")
        tau = store.get("tau")
        moi_spinx = store.get("moi_spinx")
        mu = store.get("mu")
        beta = store.get("beta")
        betad = store.get("betad")
        phi = store.get("phi")
        phid = store.get("phid")
        phidd = store.get("phidd")
        psi = store.get("psi")
        psid = store.get("psid")
        psidd = store.get("psidd")
        int_step = ctx.int_step

        moi_transx = moi_trans / (ref_length * ref_length * mu * mu * mass)

        betad_new = (
            (-velocityxd / velocityx + velocityx * cyb) * beta
            - psid
            + tau * grav * math.cos(gamma) / (velocityx * velocity_ss) * phi
            + tau * grav * math.sin(gamma) / (velocityx * velocity_ss) * psi
            + nonlinear * (velocityx * cyb3 * beta**3 / 6)
        )
        beta = integrate(betad_new, betad, beta, int_step)
        betad = betad_new

        phidd_new = (
            velocityx * omegax * clwb / (mu * mu * moi_transx) * beta
            + velocityx * clp / (mu * mu * moi_transx) * phid
            + moi_spinx * omegax / moi_transx * psid
            + nonlinear
            * (
                velocityx * omegax * clwb3 / (6 * mu * mu * moi_transx) * beta**3
                + clp3 / (6 * mu**4 * moi_transx * velocityx) * phid**3
                + omegax * clwb2p / (6 * mu**3 * moi_transx) * beta * beta * phid
                + omegax * clwbp2 / (6 * mu**4 * moi_transx * velocityx) * beta * phid * phid
            )
        )
        phid_new = integrate(phidd_new, phidd, phid, int_step)
        phidd = phidd_new
        phi = integrate(phid_new, phid_new, phi, int_step)
        phid = phid_new

        psidd_new = (
            velocityx * velocityx * cnb / (mu * moi_transx) * beta
            - moi_spinx * omegax / moi_transx * phid
            + velocityx * cnr / (mu * mu * moi_transx) * psid
            + nonlinear
            * (
                velocityx * velocityx * cnb3 / (6 * mu * moi_transx) * beta**3
                + cnr3 / (6 * mu**4 * moi_transx * velocityx) * psid**3
                + velocityx * cnb2r / (6 * mu * mu * moi_transx) * beta * beta * psid
                + cnbr2 / (6 * mu**3 * moi_transx) * beta * psid * psid
            )
        )
        psid_new = integrate(psidd_new, psidd, psid, int_step)
        psidd = psidd_new
        psi = integrate(psid_new, psid_new, psi, int_step)
        psid = psid_new

        betax = beta * DEG
        phix = phi * DEG
        ppx = phid * DEG / tau
        psix = psi * DEG
        rrx = psid * DEG / tau

        store.set("beta", beta)
        store.set("betad", betad)
        store.set("phi", phi)
        store.set("phid", phid)
        store.set("phidd", phidd)
        store.set("psi", psi)
        store.set("psid", psid)
        store.set("psidd", psidd)
        store.set("betax", betax)
        store.set("phix", phix)
        store.set("ppx", ppx)
        store.set("psix", psix)
        store.set("rrx", rrx)

    def terminate(self, vehicle, ctx):
        pass

