import math

import numpy as np

from cadac.constants import R, RAD
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr


class Flat3Environment:
    name = "environment"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("grav", 0.0, "real", "out", "environment"),
            Field("rho", 0.0, "real", "out", "environment"),
            Field("pdynmc", 0.0, "real", "out", "environment", ("scrn", "plot")),
            Field("mach", 0.0, "real", "out", "environment", ("scrn", "plot")),
            Field("vsound", 0.0, "real", "diag", "environment"),
            Field("press", 0.0, "real", "out", "environment"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        dvbe = store.get("dvbe")
        sbel = store.get("SBEL")
        alt = -sbel[2]
        rho, press, tempk = atmosphere76(alt)
        vsound = math.sqrt(1.4 * R * tempk)
        mach = abs(dvbe / vsound)
        pdynmc = 0.5 * rho * dvbe**2
        store.set("grav", gravity(alt))
        store.set("rho", rho)
        store.set("pdynmc", pdynmc)
        store.set("mach", mach)
        store.set("vsound", vsound)
        store.set("press", press)

    def terminate(self, vehicle, ctx):
        pass


class Flat3Kinematics:
    name = "kinematics"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("time", 0.0, "real", "exec", "kinematics", ("scrn", "plot")),
            Field("event_time", 0.0, "real", "exec", "kinematics"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        vehicle.store.set("time", ctx.sim_time)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        store.set("time", ctx.sim_time)
        store.set("event_time", ctx.event_time)

    def terminate(self, vehicle, ctx):
        pass


class Flat3Newton:
    name = "newton"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for field in (
            Field("TBL", zeros33, "mat", "out", "newton"),
            Field("TBV", zeros33, "mat", "diag", "newton"),
            Field("TVL", zeros33, "mat", "diag", "newton"),
            Field("dvbe", 0.0, "real", "init/out", "newton", ("scrn",)),
            Field("SBEL", zeros3, "vec", "state", "newton", ("plot",)),
            Field("VBEL", zeros3, "vec", "state", "newton"),
            Field("ABEL", zeros3, "vec", "state", "newton"),
            Field("psivlx", 0.0, "real", "init/diag", "newton", ("scrn", "plot")),
            Field("thtvlx", 0.0, "real", "init/diag", "newton", ("scrn", "plot")),
            Field("sbel1", 0.0, "real", "init", "newton"),
            Field("sbel2", 0.0, "real", "init", "newton"),
            Field("sbel3", 0.0, "real", "init", "newton"),
            Field("psivl", 0.0, "real", "out", "newton"),
            Field("thtvl", 0.0, "real", "out", "newton"),
            Field("alt", 0.0, "real", "out", "newton", ("scrn", "plot")),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        dvbe = store.get("dvbe")
        psivlx = store.get("psivlx")
        thtvlx = store.get("thtvlx")
        sbel1 = store.get("sbel1")
        sbel2 = store.get("sbel2")
        sbel3 = store.get("sbel3")
        phiavout = store.get("phiavout") if "phiavout" in store.names() else 0.0

        psivl = psivlx * RAD
        thtvl = thtvlx * RAD
        vbel = np.array(
            [
                dvbe * np.cos(thtvl) * np.cos(psivl),
                dvbe * np.cos(thtvl) * np.sin(psivl),
                dvbe * (-np.sin(thtvl)),
            ]
        )
        tvl = mat2tr(psivl, thtvl)
        tbv = np.eye(3)
        cphi = math.cos(phiavout)
        sphi = math.sin(phiavout)
        tbv[1, 1] = cphi
        tbv[2, 2] = cphi
        tbv[1, 2] = sphi
        tbv[2, 1] = -sphi
        tbl = tbv @ tvl
        sbel = np.array([sbel1, sbel2, sbel3])

        store.set("TBL", tbl)
        store.set("TBV", tbv)
        store.set("TVL", tvl)
        store.set("SBEL", sbel)
        store.set("VBEL", vbel)
        store.set("psivl", psivl)
        store.set("thtvl", thtvl)
        store.set("alt", -sbel[2])

    def execute(self, vehicle, ctx):
        pass

    def terminate(self, vehicle, ctx):
        pass
