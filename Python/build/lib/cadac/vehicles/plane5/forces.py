import numpy as np

from cadac.constants import RAD
from cadac.kernel.state import Field


class Plane5Forces:
    name = "forces"

    def define(self, vehicle):
        vehicle.store.define(
            Field("FSPV", (0.0, 0.0, 0.0), "vec", "out", "forces", ("plot",))
        )

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        pdynmc = store.get("pdynmc")
        mass = store.get("mass")
        thrust = store.get("thrust")
        cl = store.get("cl")
        cd = store.get("cd")
        area = store.get("area")
        alphax = store.get("alphax")
        phimvx = store.get("phimvx")
        phimv = phimvx * RAD
        alpha = alphax * RAD
        fspv1 = (-pdynmc * area * cd + thrust * np.cos(alpha)) / mass
        fspv2 = np.sin(phimv) * (pdynmc * area * cl + thrust * np.sin(alpha)) / mass
        fspv3 = -np.cos(phimv) * (pdynmc * area * cl + thrust * np.sin(alpha)) / mass
        store.set("FSPV", np.array([fspv1, fspv2, fspv3]))

    def terminate(self, vehicle, ctx):
        pass
