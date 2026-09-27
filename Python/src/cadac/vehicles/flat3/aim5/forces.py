import numpy as np

from cadac.kernel.state import Field


class Aim5Forces:
    name = "forces"

    def define(self, vehicle):
        store = vehicle.store
        plot = ("scrn", "plot")
        for field in (
            Field("FSPV", (0.0, 0.0, 0.0), "vec", "out", "forces"),
            Field("aax", 0.0, "real", "diag", "forces"),
            Field("alx", 0.0, "real", "diag", "forces", plot),
            Field("anx", 0.0, "real", "diag", "forces", plot),
        ):
            if field.name not in store:
                store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        pdynmc = store.get("pdynmc")
        area = store.get("area")
        caaim = store.get("caaim")
        cyaim = store.get("cyaim")
        cnaim = store.get("cnaim")
        thrust = store.get("thrust")
        mass = store.get("mass")
        grav = store.get("grav")
        fspv0 = (thrust - caaim * pdynmc * area) / mass
        fspv1 = (cyaim * pdynmc * area) / mass
        fspv2 = (-cnaim * pdynmc * area) / mass
        store.set("FSPV", np.array([fspv0, fspv1, fspv2]))
        store.set("aax", fspv0 / grav)
        store.set("alx", fspv1 / grav)
        store.set("anx", -fspv2 / grav)

    def terminate(self, vehicle, ctx):
        pass
