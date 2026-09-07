import math

import numpy as np

from cadac.constants import DEG, R, RAD
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr, polar_from_cart


def _phiavout(store):
    return store.get("phiavout") if "phiavout" in store.names() else 0.0


def _tav_from_phiavout(phiavout):
    tav = np.eye(3)
    cphi = math.cos(phiavout)
    sphi = math.sin(phiavout)
    tav[1, 1] = cphi
    tav[2, 2] = cphi
    tav[1, 2] = sphi
    tav[2, 1] = -sphi
    return tav


class Sam6Flat3Environment:
    name = "environment"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("grav", 0.0, "real", "out", "environment"),
            Field("rho", 0.0, "real", "out", "environment"),
            Field("pdynmc", 0.0, "real", "out", "environment", ("com",)),
            Field("mach", 0.0, "real", "out", "environment", ("scrn", "plot", "com")),
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


class Sam6Flat3Kinematics:
    name = "kinematics"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("time", 0.0, "real", "exec", "kinematics", ("com",)),
            Field("launch_delay", 0.0, "real", "data", "kinematics"),
            Field("launch_epoch", 0.0, "real", "out", "kinematics", ("com",)),
            Field("launch_time", 0.0, "real", "diag", "kinematics"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        store.set("time", ctx.sim_time)
        store.set("launch_epoch", store.get("launch_delay"))

    def execute(self, vehicle, ctx):
        store = vehicle.store
        store.set("launch_time", ctx.sim_time - store.get("launch_epoch"))
        store.set("time", ctx.sim_time)

    def terminate(self, vehicle, ctx):
        pass


class Sam6Flat3Newton:
    name = "newton"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for field in (
            Field("TAL", zeros33, "mat", "out", "newton"),
            Field("TAV", zeros33, "mat", "diag", "newton"),
            Field("TVL", zeros33, "mat", "diag", "newton"),
            Field("dvae", 0.0, "real", "init/out", "newton"),
            Field("SAEL", zeros3, "vec", "state", "newton", ("com",)),
            Field("VAEL", zeros3, "vec", "state", "newton", ("com",)),
            Field("AAEL", zeros3, "vec", "state", "newton"),
            Field("psivlx", 0.0, "real", "init/diag", "newton", ("com",)),
            Field("thtvlx", 0.0, "real", "init/diag", "newton", ("com",)),
            Field("sael1", 0.0, "real", "init", "newton"),
            Field("sael2", 0.0, "real", "init", "newton"),
            Field("sael3", 0.0, "real", "init", "newton"),
            Field("psivl", 0.0, "real", "out", "newton"),
            Field("thtvl", 0.0, "real", "out", "newton"),
            Field("alt", 0.0, "real", "out", "newton", ("com",)),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        dvae = store.get("dvae")
        psivlx = store.get("psivlx")
        thtvlx = store.get("thtvlx")
        sael1 = store.get("sael1")
        sael2 = store.get("sael2")
        sael3 = store.get("sael3")
        phiavout = _phiavout(store)

        psivl = psivlx * RAD
        thtvl = thtvlx * RAD
        vael = np.array(
            [
                dvae * np.cos(thtvl) * np.cos(psivl),
                dvae * np.cos(thtvl) * np.sin(psivl),
                dvae * (-np.sin(thtvl)),
            ]
        )
        tvl = mat2tr(psivl, thtvl)
        tav = _tav_from_phiavout(phiavout)
        tal = tav @ tvl
        sael = np.array([sael1, sael2, sael3])

        store.set("TAL", tal)
        store.set("TAV", tav)
        store.set("TVL", tvl)
        store.set("SAEL", sael)
        store.set("VAEL", vael)
        store.set("psivl", psivl)
        store.set("thtvl", thtvl)
        store.set("alt", -sael[2])

    def execute(self, vehicle, ctx):
        store = vehicle.store
        fspa = store.get("FSPA")
        grav = store.get("grav")
        phiavout = _phiavout(store)
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
        psivl = float(polar[1])
        thtvl = float(polar[2])
        tvl = mat2tr(psivl, thtvl)
        tav = _tav_from_phiavout(phiavout)
        tal = tav @ tvl

        store.set("SAEL", sael)
        store.set("VAEL", vael)
        store.set("AAEL", aael)
        store.set("TAL", tal)
        store.set("TAV", tav)
        store.set("TVL", tvl)
        store.set("dvae", dvae)
        store.set("psivl", psivl)
        store.set("thtvl", thtvl)
        store.set("psivlx", psivl * DEG)
        store.set("thtvlx", thtvl * DEG)
        store.set("alt", -sael[2])

    def terminate(self, vehicle, ctx):
        pass
