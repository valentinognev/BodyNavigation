import math

from cadac.constants import EARTH_MASS, G, R, REARTH
from cadac.env.us76 import atmosphere76
from cadac.kernel.state import Field


class Sam6Environment:
    name = "environment"

    def define(self, vehicle):
        store = vehicle.store
        scrn_plot_com = ("scrn", "plot", "com")
        for field in (
            Field("press", 0.0, "real", "out", "environment"),
            Field("rho", 0.0, "real", "out", "environment"),
            Field("vsound", 0.0, "real", "diag", "environment"),
            Field("grav", 0.0, "real", "out", "environment"),
            Field("vmach", 0.0, "real", "out", "environment", scrn_plot_com),
            Field("pdynmc", 0.0, "real", "out", "environment", scrn_plot_com),
            Field("tempk", 0.0, "real", "out", "environment"),
            Field("mfreeze_environ", 0, "int", "save", "environment"),
            Field("pdynmcf", 0.0, "real", "save", "environment"),
            Field("machf", 0.0, "real", "save", "environment"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        dvbe = store.get("dvbe")
        alt = store.get("alt")
        mfreeze_environ = store.get("mfreeze_environ")
        pdynmcf = store.get("pdynmcf")
        machf = store.get("machf")

        rad = REARTH + alt
        grav = G * EARTH_MASS / rad**2
        rho, press, tempk = atmosphere76(alt)
        vsound = math.sqrt(1.4 * R * tempk)
        vmach = abs(dvbe / vsound)
        pdynmc = 0.5 * rho * dvbe * dvbe

        if "mguide" in store and "trdynm" in store and "trcond" in store:
            trcond = store.get("trcond")
            guid_term = store.get("mguide") % 10
            if guid_term == 6:
                if pdynmc <= store.get("trdynm"):
                    trcond = 3
            store.set("trcond", trcond)

        if "mfreeze" in store:
            mfreeze = store.get("mfreeze")
            if mfreeze == 0:
                mfreeze_environ = 0
            else:
                if mfreeze != mfreeze_environ:
                    mfreeze_environ = mfreeze
                    machf = vmach
                    pdynmcf = pdynmc
                vmach = machf
                pdynmc = pdynmcf
            store.set("mfreeze_environ", mfreeze_environ)
            store.set("pdynmcf", pdynmcf)
            store.set("machf", machf)

        store.set("press", press)
        store.set("rho", rho)
        store.set("grav", grav)
        store.set("vmach", vmach)
        store.set("pdynmc", pdynmc)
        store.set("tempk", tempk)
        store.set("vsound", vsound)

    def terminate(self, vehicle, ctx):
        pass
