import numpy as np

from cadac.kernel.state import Field


class Aim5AircraftForces:
    name = "forces"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("FSPV", (0.0, 0.0, 0.0), "vec", "out", "forces"),
            Field("acc_longx", 0.0, "real", "data", "forces"),
        ):
            if field.name not in store.names():
                store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        acc_longx = store.get("acc_longx")
        grav = store.get("grav")
        anx = store.get("anx")
        store.set("FSPV", np.array([acc_longx * grav, 0.0, -anx * grav]))

    def terminate(self, vehicle, ctx):
        pass
