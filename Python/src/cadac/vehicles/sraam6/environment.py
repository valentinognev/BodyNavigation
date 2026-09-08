import math

from cadac.constants import R
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.kernel.state import Field


class Sraam6Environment:
    name = "environment"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("press", 0.0, "real", "out", "environment"),
            Field("rho", 0.0, "real", "out", "environment"),
            Field("vsound", 0.0, "real", "diag", "environment"),
            Field("grav", 0.0, "real", "out", "environment"),
            Field("vmach", 0.0, "real", "out", "environment", ("scrn", "plot", "com")),
            Field("pdynmc", 0.0, "real", "out", "environment", ("plot",)),
            Field("tempk", 0.0, "real", "out", "environment"),
            Field("mfreeze_environ", 0, "int", "save", "environment"),
            Field("pdynmcf", 0.0, "real", "save", "environment"),
            Field("vmachf", 0.0, "real", "save", "environment"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mfreeze_environ = store.get("mfreeze_environ")
        pdynmcf = store.get("pdynmcf")
        vmachf = store.get("vmachf")
        dvbe = store.get("dvbe")
        hbe = store.get("hbe")
        trcond = store.get("trcond")
        trmach = store.get("trmach")
        trdynm = store.get("trdynm")
        mguid = store.get("mguid")

        grav = gravity(hbe)
        rho, press, tempk = atmosphere76(hbe)
        vsound = math.sqrt(1.4 * R * tempk)
        vmach = abs(dvbe / vsound)
        pdynmc = 0.5 * rho * dvbe**2

        if mguid == 6:
            if vmach <= trmach:
                trcond = 2
            if pdynmc <= trdynm:
                trcond = 3

        if "mfreeze" in store:
            mfreeze = store.get("mfreeze")
            if mfreeze == 0:
                mfreeze_environ = 0
            else:
                if mfreeze != mfreeze_environ:
                    mfreeze_environ = mfreeze
                    vmachf = vmach
                    pdynmcf = pdynmc
                vmach = vmachf
                pdynmc = pdynmcf
            store.set("mfreeze_environ", mfreeze_environ)
            store.set("pdynmcf", pdynmcf)
            store.set("vmachf", vmachf)

        store.set("press", press)
        store.set("rho", rho)
        store.set("grav", grav)
        store.set("vmach", vmach)
        store.set("pdynmc", pdynmc)
        store.set("tempk", tempk)
        store.set("vsound", vsound)
        store.set("trcond", trcond)

    def terminate(self, vehicle, ctx):
        pass
