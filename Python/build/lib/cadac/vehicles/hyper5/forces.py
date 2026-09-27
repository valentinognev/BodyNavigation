from math import cos, sin

import numpy as np

from cadac.constants import RAD
from cadac.kernel.state import Field


class Hyper5Forces:
    name = "forces"

    def define(self, vehicle):
        store = vehicle.store
        if "FSPV" not in store:
            store.define(
                Field("FSPV", (0.0, 0.0, 0.0), "vec", "out", "forces", ("plot",))
            )

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        pdynmc = store.get("pdynmc")
        cl = store.get("cl")
        cd = store.get("cd")
        area = store.get("area")
        thrust = store.get("thrust")
        mass = store.get("mass")
        alphax = store.get("alphax")
        phimvx = store.get("phimvx")
        phimv = phimvx * RAD
        alpha = alphax * RAD
        fspv1 = (-pdynmc * area * cd + thrust * cos(alpha)) / mass
        fspv2 = sin(phimv) * (pdynmc * area * cl + thrust * sin(alpha)) / mass
        fspv3 = -cos(phimv) * (pdynmc * area * cl + thrust * sin(alpha)) / mass
        store.set("FSPV", np.array([fspv1, fspv2, fspv3]))

    def terminate(self, vehicle, ctx):
        pass
