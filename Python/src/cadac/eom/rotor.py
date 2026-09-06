import math

import numpy as np

from cadac.constants import AGRAV, R, RAD
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
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
        pass

    def terminate(self, vehicle, ctx):
        pass

