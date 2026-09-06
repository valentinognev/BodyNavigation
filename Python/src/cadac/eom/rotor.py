import math

import numpy as np

from cadac.constants import R
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.kernel.state import Field

RPM = 9.5493
RHO_SL = 1.225


class RotorEnvironment:
    name = "environment"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("mwind", 0, "int", "data", "environment"),
            Field("press", 0.0, "real", "out", "environment"),
            Field("rho", 0.0, "real", "out", "environment"),
            Field("vsound", 0.0, "real", "diag", "environment"),
            Field("grav", 0.0, "real", "out", "environment"),
            Field("vmach", 0.0, "real", "out", "environment", ("scrn", "plot", "com")),
            Field("pdynmc", 0.0, "real", "out", "environment", ("scrn", "plot")),
            Field("tempk", 0.0, "real", "out", "environment"),
            Field("dvae", 0.0, "real", "data", "environment"),
            Field("dvael", 0.0, "real", "data", "environment"),
            Field("waltl", 0.0, "real", "data", "environment"),
            Field("dvaeh", 0.0, "real", "data", "environment"),
            Field("walth", 0.0, "real", "data", "environment"),
            Field("vaed3", 0.0, "real", "data", "environment"),
            Field("psiwdx", 0.0, "real", "data", "environment"),
            Field("twind", 0.1, "real", "data", "environment"),
            Field("VAELS", zeros3, "vec", "state", "environment"),
            Field("VAELSD", zeros3, "vec", "state", "environment"),
            Field("VAEL", zeros3, "vec", "out", "environment"),
            Field("dvba", 0.0, "real", "out", "environment"),
            Field("VBAL", zeros3, "vec", "out", "environment"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mwind = store.get("mwind")
        if mwind != 0:
            raise ValueError(f"unknown mwind {mwind}")
        hbe = store.get("hbe")
        vbel = store.get("VBEL")
        rho, press, tempk = atmosphere76(hbe)
        vsound = math.sqrt(1.4 * R * tempk)
        vael = np.zeros(3)
        vbal = vbel - vael
        dvba = float(np.linalg.norm(vbal))
        vmach = abs(dvba / vsound)
        pdynmc = 0.5 * rho * dvba * dvba
        store.set("grav", gravity(hbe))
        store.set("rho", rho)
        store.set("press", press)
        store.set("tempk", tempk)
        store.set("vsound", vsound)
        store.set("VAEL", vael)
        store.set("VBAL", vbal)
        store.set("dvba", dvba)
        store.set("vmach", vmach)
        store.set("pdynmc", pdynmc)

    def terminate(self, vehicle, ctx):
        pass
