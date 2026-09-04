import math

from cadac.constants import R
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.kernel.state import Field


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
