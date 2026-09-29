"""SRAAM5 atmosphere and gravity — Fortran MODULE.FOR subroutine G2."""

from cadac.env.gravity import gravity
from cadac.env.iso62 import iso62
from cadac.kernel.state import Field


class Sraam5Environment:
    name = "environment"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("optmet", 0.0, "real", "data", "environment"),
            Field("press", 0.0, "real", "out", "environment"),
            Field("rho", 0.0, "real", "out", "environment"),
            Field("vsound", 0.0, "real", "diag", "environment"),
            Field("grav", 0.0, "real", "out", "environment"),
            Field("vmach", 0.0, "real", "out", "environment", ("scrn", "plot", "com")),
            Field("pdynmc", 0.0, "real", "out", "environment", ("plot",)),
            Field("tempk", 0.0, "real", "diag", "environment"),
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

        grav = gravity(hbe)
        atm = iso62(hbe, dvbe)
        press = atm["press"]
        rho = atm["rho"]
        tempk = atm["k"]
        vsound = atm["vsound"]
        vmach = atm["mach"]
        pdynmc = atm["pdynmc"]

        if "mguid" in store and "trmach" in store and "trdynm" in store and "trcode" in store:
            trcode = store.get("trcode")
            if store.get("mguid") == 6:
                if vmach <= store.get("trmach"):
                    trcode = 2.0
                if pdynmc <= store.get("trdynm"):
                    trcode = 3.0
            store.set("trcode", trcode)

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

    def terminate(self, vehicle, ctx):
        pass
