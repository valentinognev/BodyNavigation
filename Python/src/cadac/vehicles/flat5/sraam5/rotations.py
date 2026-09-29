"""SRAAM5 kinematic rotations — Fortran MODULE.FOR D2."""

from math import cos, sin

import numpy as np

from cadac.kernel.state import Field

_ZEROS3 = (0.0, 0.0, 0.0)


class Sraam5Rotations:
    """Kinematic Rotation Module (D2)."""

    name = "rotations"

    def define(self, vehicle):
        store = vehicle.store
        store.define(Field("mturn", 0, "int", "data", "rotations"))
        # PHD owned by control for MTURN=1; default here so D2 BTT is safe alone
        if "phd" not in store:
            store.define(Field("phd", 0.0, "real", "state", "control"))
        store.define(Field("WBVB", _ZEROS3, "vec", "diag", "rotations"))

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mturn = store.get("mturn")
        alp = store.get("alp")
        alpd = store.get("alpd")

        if mturn == 0:
            # Skid-to-turn (yaw-to-turn): ALP, ALPD, BETD from autopilot
            betd = store.get("betd")
            wbvb = np.array(
                [betd * sin(alp), alpd, -betd * cos(alp)],
                dtype=float,
            )
        else:
            # Bank-to-turn: ALP, ALPD, PHD from autopilot
            phd = store.get("phd")
            wbvb = np.array(
                [phd * cos(alp), alpd, phd * sin(alp)],
                dtype=float,
            )

        store.set("WBVB", wbvb)

    def terminate(self, vehicle, ctx):
        pass
