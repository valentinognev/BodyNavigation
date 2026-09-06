from math import cos, sin

import numpy as np

from cadac.constants import RAD
from cadac.kernel.state import Field


class Cruise5Forces:
    name = "forces"

    def define(self, vehicle):
        store = vehicle.store
        if "FSPV" not in store.names():
            store.define(
                Field("FSPV", (0.0, 0.0, 0.0), "vec", "out", "forces", ("plot",))
            )

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        pdynmc = store.get("pdynmc")
        area = store.get("area")
        cd = store.get("cd")
        cl = store.get("cl")
        thrust = store.get("thrust")
        mass = store.get("mass")
        alphax = store.get("alphax")
        phimvx = store.get("phimvx")
        alpha = alphax * RAD
        phimv = phimvx * RAD
        fspv1 = (-pdynmc * area * cd + thrust * cos(alpha)) / mass
        fspv2 = sin(phimv) * (pdynmc * area * cl + thrust * sin(alpha)) / mass
        fspv3 = -cos(phimv) * (pdynmc * area * cl + thrust * sin(alpha)) / mass
        store.set("FSPV", np.array([fspv1, fspv2, fspv3]))

    def terminate(self, vehicle, ctx):
        pass
