import math

import numpy as np

from cadac.constants import R
from cadac.env.us76 import atmosphere76
from cadac.kernel.state import Field
from cadac.math.wgs84 import cad_grav84


class Round6Environment:
    name = "environment"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("mair", 0, "int", "data", "environment"),
            Field("press", 0.0, "real", "out", "environment"),
            Field("rho", 0.0, "real", "out", "environment"),
            Field("vsound", 0.0, "real", "diag", "environment"),
            Field("vmach", 0.0, "real", "out", "environment", ("scrn", "plot", "com")),
            Field("pdynmc", 0.0, "real", "out", "environment", ("scrn", "plot")),
            Field("tempk", 0.0, "real", "out", "environment"),
            Field("mfreeze_evrn", 0, "int", "save", "environment"),
            Field("pdynmcf", 0.0, "real", "save", "environment"),
            Field("vmachf", 0.0, "real", "save", "environment"),
            Field("GRAVG", zeros3, "vec", "out", "environment"),
            Field("grav", 0.0, "real", "out", "environment"),
            Field("dvae", 0.0, "real", "data", "environment"),
            Field("dvael", 0.0, "real", "data", "environment"),
            Field("waltl", 0.0, "real", "data", "environment"),
            Field("dvaeh", 0.0, "real", "data", "environment"),
            Field("walth", 0.0, "real", "data", "environment"),
            Field("vaed3", 0.0, "real", "data", "environment"),
            Field("psiwdx", 0.0, "real", "data", "environment"),
            Field("twind", 0.1, "real", "data", "environment"),
            Field("VAEDS", zeros3, "vec", "state", "environment"),
            Field("VAEDSD", zeros3, "vec", "state", "environment"),
            Field("VAED", zeros3, "vec", "out", "environment"),
            Field("dvba", 0.0, "real", "out", "environment"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mair = store.get("mair")
        matmo = mair // 100
        mturb = (mair - matmo * 100) // 10
        mwind = (mair - matmo * 100) % 10
        if matmo != 0 or mturb != 0 or mwind != 0:
            raise ValueError(f"unknown mair {mair}")

        dvba = store.get("dvba")
        vaeds = store.get("VAEDS")
        vaedsd = store.get("VAEDSD")
        time = store.get("time")
        alt = store.get("alt")
        vbed = store.get("VBED")
        sbii = store.get("SBII")

        gravg = cad_grav84(sbii, time)
        grav = float(np.linalg.norm(gravg))

        rho, press, tempk = atmosphere76(alt)
        vsound = math.sqrt(1.4 * R * tempk)

        vmach = abs(dvba / vsound)
        pdynmc = 0.5 * rho * dvba * dvba

        vaed = np.zeros(3)
        vbad = vbed - vaed
        dvba = float(np.linalg.norm(vbad))
        vmach = abs(dvba / vsound)
        pdynmc = 0.5 * rho * dvba * dvba

        names = store.names()
        if "trcode" in names and "mguid" in names:
            if store.get("mguid") == 6:
                trcode = store.get("trcode")
                if vmach <= store.get("trmach"):
                    trcode = 2.0
                if pdynmc <= store.get("trdynm"):
                    trcode = 3.0
                store.set("trcode", trcode)

        if "mfreeze" in names:
            mfreeze = store.get("mfreeze")
            mfreeze_evrn = store.get("mfreeze_evrn")
            pdynmcf = store.get("pdynmcf")
            vmachf = store.get("vmachf")
            if mfreeze == 0:
                mfreeze_evrn = 0
            else:
                if mfreeze != mfreeze_evrn:
                    mfreeze_evrn = mfreeze
                    vmachf = vmach
                    pdynmcf = pdynmc
                vmach = vmachf
                pdynmc = pdynmcf
            store.set("mfreeze_evrn", mfreeze_evrn)
            store.set("pdynmcf", pdynmcf)
            store.set("vmachf", vmachf)

        store.set("VAEDS", vaeds)
        store.set("VAEDSD", vaedsd)
        store.set("press", press)
        store.set("rho", rho)
        store.set("vmach", vmach)
        store.set("pdynmc", pdynmc)
        store.set("GRAVG", gravg)
        store.set("grav", grav)
        store.set("VAED", vaed)
        store.set("dvba", dvba)
        store.set("vsound", vsound)
        store.set("tempk", tempk)

    def terminate(self, vehicle, ctx):
        pass
