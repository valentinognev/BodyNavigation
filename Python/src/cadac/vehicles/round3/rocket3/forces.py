from math import cos, sin

import numpy as np

from cadac.constants import RAD
from cadac.kernel.state import Field
from cadac.vehicles.round3.rocket3.stubs import StubModule


class Rocket3Forces(StubModule):
    name = "forces"
    _fields = (
        Field("FSPV", (0.0, 0.0, 0.0), "vec", "out", "forces", ("plot",)),
    )

    def execute(self, vehicle, ctx):
        store = vehicle.store
        pdynmc = store.get("pdynmc")
        sref = store.get("sref")
        cd = store.get("cd")
        cl = store.get("cl")
        thrustx = store.get("thrustx")
        vmass = store.get("vmass")
        alphax = store.get("alphax")
        phimvx = store.get("phimvx")

        alpha = alphax * RAD
        phimv = phimvx * RAD
        fd = pdynmc * sref * cd
        fl = pdynmc * sref * cl
        thrust = thrustx * 1000.0
        fapm1 = -fd + thrust * cos(alpha)
        fapm3 = -(fl + thrust * sin(alpha))
        store.set(
            "FSPV",
            np.array(
                [
                    fapm1 / vmass,
                    -sin(phimv) * fapm3 / vmass,
                    cos(phimv) * fapm3 / vmass,
                ],
                dtype=float,
            ),
        )
