"""ROCKET3 atmosphere and gravity — Fortran MODULE.FOR subroutine G2."""

from cadac.env.gravity import gravity
from cadac.env.iso62 import iso62
from cadac.kernel.state import Field


class Rocket3Environment:
    name = "environment"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("optmet", 0.0, "real", "data", "environment"),
            Field("opnoro", 0.0, "real", "data", "environment"),
            Field("mair", 0, "int", "data", "environment"),
            Field("press", 0.0, "real", "out", "environment"),
            Field("rho", 0.0, "real", "out", "environment"),
            Field("vsound", 0.0, "real", "diag", "environment"),
            Field("grav", 0.0, "real", "out", "environment"),
            Field("vmach", 0.0, "real", "out", "environment"),
            Field("pdynmc", 0.0, "real", "out", "environment"),
            Field("tempk", 0.0, "real", "diag", "environment"),
        ):
            if field.name not in store:
                store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        balt = store.get("balt")
        dvbe = store.get("dvbe")
        mair = store.get("mair")

        grav = gravity(balt)

        if mair == 0:
            atm = iso62(balt, dvbe)
            press = atm["press"]
            rho = atm["rho"]
            tempk = atm["k"]
            vsound = atm["vsound"]
            vmach = atm["mach"]
            pdynmc = atm["pdynmc"]
        else:
            raise ValueError(f"unknown mair {mair}")

        store.set("press", press)
        store.set("rho", rho)
        store.set("grav", grav)
        store.set("vmach", vmach)
        store.set("pdynmc", pdynmc)
        store.set("tempk", tempk)
        store.set("vsound", vsound)

    def terminate(self, vehicle, ctx):
        pass
