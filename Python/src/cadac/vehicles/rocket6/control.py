import numpy as np

from cadac.constants import AGRAV, DEG, RAD
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field


def _sign(variable):
    if variable < 0:
        return -1
    return 1


class Rocket6Control:
    name = "control"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("plot",)
        for field in (
            Field("maut", 0, "int", "data", "control"),
            Field("mfreeze", 0, "int", "data", "control"),
            Field("waclp", 0.0, "real", "data", "control", plot),
            Field("zaclp", 0.0, "real", "data", "control", plot),
            Field("paclp", 0.0, "real", "data", "control", plot),
            Field("delimx", 0.0, "real", "data", "control"),
            Field("drlimx", 0.0, "real", "data", "control"),
            Field("yyd", 0.0, "real", "state", "control"),
            Field("yy", 0.0, "real", "state", "control"),
            Field("zzd", 0.0, "real", "state", "control"),
            Field("zz", 0.0, "real", "state", "control"),
            Field("delecx", 0.0, "real", "out", "control"),
            Field("delrcx", 0.0, "real", "out", "control"),
            Field("alcomx_actual", 0.0, "real", "diag", "control", plot),
            Field("ancomx_actual", 0.0, "real", "diag", "control", plot),
            Field("GAINFP", zeros3, "vec", "diag", "control", plot),
            Field("gainl", 0.0, "real", "data", "control"),
            Field("gkp", 0.0, "real", "diag", "control"),
            Field("gkphi", 0.0, "real", "diag", "control"),
            Field("isetc2", 0.0, "real", "init", "control"),
            Field("wacly", 0.0, "real", "data", "control", plot),
            Field("zacly", 0.0, "real", "data", "control"),
            Field("pacly", 0.0, "real", "data", "control"),
            Field("GAINFY", zeros3, "vec", "diag", "control"),
            Field("factwaclp", 0.0, "real", "data", "control"),
            Field("factwacly", 0.0, "real", "data", "control"),
            Field("alcomx", 0.0, "real", "data", "control", plot),
            Field("ancomx", 0.0, "real", "data", "control", plot),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        maut = store.get("maut")
        if maut not in (0, 53):
            raise ValueError(f"unknown maut {maut}")

        delecx = 0.0
        delrcx = 0.0
        delimx = store.get("delimx")
        drlimx = store.get("drlimx")
        mprop = store.get("mprop")
        gnmax = store.get("gnmax")
        gymax = store.get("gymax")
        alcomx = store.get("alcomx")
        ancomx = store.get("ancomx")
        dt = ctx.int_step

        mauty = maut // 10
        mautp = maut % 10

        if mauty == 5:
            if alcomx > gymax:
                alcomx = gymax
            if alcomx < -gymax:
                alcomx = -gymax
            if mprop:
                delrcx = self.control_yaw_accel(vehicle, alcomx, dt)
        if mautp == 3:
            if ancomx > gnmax:
                ancomx = gnmax
            if ancomx < -gnmax:
                ancomx = -gnmax
            if mprop:
                delecx = self.control_normal_accel(vehicle, ancomx, dt)

        if abs(delecx) > delimx:
            delecx = delimx * _sign(delecx)
        if abs(delrcx) > drlimx:
            delrcx = drlimx * _sign(delrcx)

        store.set("delecx", delecx)
        store.set("delrcx", delrcx)
        store.set("alcomx_actual", alcomx)
        store.set("ancomx_actual", ancomx)

    def control_normal_accel(self, vehicle, ancomx, int_step):
        store = vehicle.store
        zaclp = store.get("zaclp")
        factwaclp = store.get("factwaclp")
        pdynmc = store.get("pdynmc")
        dla = store.get("dla")
        dma = store.get("dma")
        dmq = store.get("dmq")
        dmde = store.get("dmde")
        dvbec = store.get("dvbec")
        qqcx = store.get("qqcx")
        fspcb = np.asarray(store.get("FSPCB"), dtype=float)
        zzd = store.get("zzd")
        zz = store.get("zz")

        waclp = (0.1 + 0.5e-5 * (pdynmc - 20e3)) * (1 + factwaclp)
        paclp = 0.7 + 1e-5 * (pdynmc - 20e3) * (1 + factwaclp)

        gainfb3 = waclp * waclp * paclp / (dla * dmde)
        gainfb2 = (2 * zaclp * waclp + paclp + dmq - dla / dvbec) / dmde
        gainfb1 = (
            waclp * waclp
            + 2.0 * zaclp * waclp * paclp
            + dma
            + dmq * dla / dvbec
            - gainfb2 * dmde * dla / dvbec
        ) / (dla * dmde)

        fspb3 = fspcb[2]
        zzd_new = AGRAV * ancomx + fspb3
        zz = integrate(zzd_new, zzd, zz, int_step)
        zzd = zzd_new
        dqc = -gainfb1 * (-fspb3) - gainfb2 * qqcx * RAD + gainfb3 * zz
        delecx = dqc * DEG

        store.set("zzd", zzd)
        store.set("zz", zz)
        store.set("GAINFP", (gainfb1, gainfb2, gainfb3))
        store.set("waclp", waclp)
        store.set("zaclp", zaclp)
        store.set("paclp", paclp)
        return delecx

    def control_yaw_accel(self, vehicle, alcomx, int_step):
        store = vehicle.store
        zacly = store.get("zacly")
        factwacly = store.get("factwacly")
        pdynmc = store.get("pdynmc")
        dyb = store.get("dyb")
        dnb = store.get("dnb")
        dnr = store.get("dnr")
        dndr = store.get("dndr")
        dvbe = store.get("dvbe")
        rrcx = store.get("rrcx")
        fspcb = np.asarray(store.get("FSPCB"), dtype=float)
        yyd = store.get("yyd")
        yy = store.get("yy")

        wacly = (0.1 + 0.5e-5 * (pdynmc - 20e3)) * (1 + factwacly)
        pacly = 0.7 + 1e-5 * (pdynmc - 20e3) * (1 + factwacly)

        gainfb3 = -wacly * wacly * pacly / (dyb * dndr)
        gainfb2 = (2 * zacly * wacly + pacly + dnr + dyb / dvbe) / dndr
        gainfb1 = (
            -wacly * wacly
            - 2.0 * zacly * wacly * pacly
            + dnb
            + dnr * dyb / dvbe
            - gainfb2 * dndr * dnb / dvbe
        ) / (dyb * dndr)

        fspb2 = fspcb[1]
        yyd_new = AGRAV * alcomx - fspb2
        yy = integrate(yyd_new, yyd, yy, int_step)
        yyd = yyd_new
        drc = -gainfb1 * fspb2 - gainfb2 * rrcx * RAD + gainfb3 * yy
        drcx = drc * DEG

        store.set("yyd", yyd)
        store.set("yy", yy)
        store.set("GAINFY", (gainfb1, gainfb2, gainfb3))
        store.set("wacly", wacly)
        store.set("zacly", zacly)
        store.set("pacly", pacly)
        return drcx

    def terminate(self, vehicle, ctx):
        pass
