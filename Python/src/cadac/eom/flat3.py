import math

import numpy as np

from cadac.constants import DEG, R, RAD
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr, polar_from_cart


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
            if field.name not in store.names():
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
        # C++ Flat3::init_newton does not load alt; first propulsion/control see 0.

    def execute(self, vehicle, ctx):
        store = vehicle.store
        fspv = store.get("FSPV")
        grav = store.get("grav")
        phiavout = store.get("phiavout") if "phiavout" in store.names() else 0.0
        tbl = store.get("TBL")
        sbel = store.get("SBEL")
        vbel = store.get("VBEL")
        abel = store.get("ABEL")
        int_step = ctx.int_step

        gravl = np.array([0.0, 0.0, grav])
        next_acc = tbl.T @ fspv + gravl
        next_vel = integrate(next_acc, abel, vbel, int_step)
        sbel = integrate(next_vel, vbel, sbel, int_step)
        abel = next_acc
        vbel = next_vel

        polar = polar_from_cart(vbel)
        dvbe = float(polar[0])
        psivl = float(polar[1])
        thtvl = float(polar[2])
        tvl = mat2tr(psivl, thtvl)

        tbv = np.eye(3)
        cphi = math.cos(phiavout)
        sphi = math.sin(phiavout)
        tbv[1, 1] = cphi
        tbv[2, 2] = cphi
        tbv[1, 2] = sphi
        tbv[2, 1] = -sphi
        tbl = tbv @ tvl

        store.set("SBEL", sbel)
        store.set("VBEL", vbel)
        store.set("ABEL", abel)
        store.set("TBL", tbl)
        store.set("TBV", tbv)
        store.set("TVL", tvl)
        store.set("dvbe", dvbe)
        store.set("psivl", psivl)
        store.set("thtvl", thtvl)
        store.set("psivlx", psivl * DEG)
        store.set("thtvlx", thtvl * DEG)
        store.set("alt", -sbel[2])

    def terminate(self, vehicle, ctx):
        pass


class Flat3AircraftEnvironment:
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
        dvae = store.get("dvae")
        sael = store.get("SAEL")
        alt = -sael[2]
        rho, press, tempk = atmosphere76(alt)
        vsound = math.sqrt(1.4 * R * tempk)
        mach = abs(dvae / vsound)
        pdynmc = 0.5 * rho * dvae**2
        store.set("grav", gravity(alt))
        store.set("rho", rho)
        store.set("pdynmc", pdynmc)
        store.set("mach", mach)
        store.set("vsound", vsound)
        store.set("press", press)

    def terminate(self, vehicle, ctx):
        pass


class Flat3AircraftNewton:
    name = "newton"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for field in (
            Field("TAL", zeros33, "mat", "out", "newton"),
            Field("TAV", zeros33, "mat", "diag", "newton"),
            Field("TVL", zeros33, "mat", "diag", "newton"),
            Field("dvae", 0.0, "real", "init/out", "newton", ("com",)),
            Field("SAEL", zeros3, "vec", "state", "newton", ("com",)),
            Field("VAEL", zeros3, "vec", "state", "newton", ("com",)),
            Field("AAEL", zeros3, "vec", "state", "newton"),
            Field("psialx", 0.0, "real", "init/diag", "newton"),
            Field("thtalx", 0.0, "real", "init/diag", "newton"),
            Field("sael1", 0.0, "real", "init", "newton"),
            Field("sael2", 0.0, "real", "init", "newton"),
            Field("sael3", 0.0, "real", "init", "newton"),
            Field("psial", 0.0, "real", "out", "newton", ("com",)),
            Field("thtal", 0.0, "real", "out", "newton", ("com",)),
            Field("alt", 0.0, "real", "out", "newton", ("scrn", "plot")),
        ):
            if field.name not in store.names():
                store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        dvae = store.get("dvae")
        psialx = store.get("psialx")
        thtalx = store.get("thtalx")
        sael1 = store.get("sael1")
        sael2 = store.get("sael2")
        sael3 = store.get("sael3")
        phiavout = store.get("phiavout") if "phiavout" in store.names() else 0.0

        psial = psialx * RAD
        thtal = thtalx * RAD
        vael = np.array(
            [
                dvae * np.cos(thtal) * np.cos(psial),
                dvae * np.cos(thtal) * np.sin(psial),
                dvae * (-np.sin(thtal)),
            ]
        )
        tvl = mat2tr(psial, thtal)
        tav = np.eye(3)
        cphi = math.cos(phiavout)
        sphi = math.sin(phiavout)
        tav[1, 1] = cphi
        tav[2, 2] = cphi
        tav[1, 2] = sphi
        tav[2, 1] = -sphi
        tal = tav @ tvl
        sael = np.array([sael1, sael2, sael3])

        store.set("TAL", tal)
        store.set("TAV", tav)
        store.set("TVL", tvl)
        store.set("SAEL", sael)
        store.set("VAEL", vael)
        store.set("psial", psial)
        store.set("thtal", thtal)
        store.set("alt", -sael[2])

    def execute(self, vehicle, ctx):
        store = vehicle.store
        fspa = store.get("FSPA")
        grav = store.get("grav")
        phiavout = store.get("phiavout") if "phiavout" in store.names() else 0.0
        tal = store.get("TAL")
        sael = store.get("SAEL")
        vael = store.get("VAEL")
        aael = store.get("AAEL")
        int_step = ctx.int_step

        gravl = np.array([0.0, 0.0, grav])
        next_acc = tal.T @ fspa + gravl
        next_vel = integrate(next_acc, aael, vael, int_step)
        sael = integrate(next_vel, vael, sael, int_step)
        aael = next_acc
        vael = next_vel

        polar = polar_from_cart(vael)
        dvae = float(polar[0])
        psial = float(polar[1])
        thtal = float(polar[2])
        tvl = mat2tr(psial, thtal)

        tav = np.eye(3)
        cphi = math.cos(phiavout)
        sphi = math.sin(phiavout)
        tav[1, 1] = cphi
        tav[2, 2] = cphi
        tav[1, 2] = sphi
        tav[2, 1] = -sphi
        tal = tav @ tvl

        store.set("SAEL", sael)
        store.set("VAEL", vael)
        store.set("AAEL", aael)
        store.set("TAL", tal)
        store.set("TAV", tav)
        store.set("TVL", tvl)
        store.set("dvae", dvae)
        store.set("psial", psial)
        store.set("thtal", thtal)
        store.set("psialx", psial * DEG)
        store.set("thtalx", thtal * DEG)
        store.set("alt", -sael[2])

    def terminate(self, vehicle, ctx):
        pass

