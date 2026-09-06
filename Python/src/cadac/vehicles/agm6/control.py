from math import sqrt

from cadac.constants import DEG, RAD
from cadac.kernel.state import Field

SMALL = 1.e-7


def _sign(variable):
    if variable < 0:
        return -1
    return 1


class Agm6Control:
    name = "control"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("plot",)
        for field in (
            Field("maut", 0, "int", "data", "control"),
            Field("mfreeze", 0, "int", "data", "control"),
            Field("wacl", 0.0, "real", "data", "control", plot),
            Field("zacl", 0.0, "real", "data", "control"),
            Field("pacl", 0.0, "real", "data", "control"),
            Field("alimit", 0.0, "real", "data", "control"),
            Field("dqlimx", 0.0, "real", "data", "control"),
            Field("drlimx", 0.0, "real", "data", "control"),
            Field("dplimx", 0.0, "real", "data", "control"),
            Field("phicomx", 0.0, "real", "data", "control", plot),
            Field("wrcl", 0.0, "real", "data", "control"),
            Field("zrcl", 0.0, "real", "data", "control"),
            Field("yyd", 0.0, "real", "state", "control"),
            Field("yy", 0.0, "real", "state", "control"),
            Field("zzd", 0.0, "real", "state", "control"),
            Field("zz", 0.0, "real", "state", "control"),
            Field("dpcx", 0.0, "real", "out", "control", plot),
            Field("dqcx", 0.0, "real", "out", "control", plot),
            Field("drcx", 0.0, "real", "out", "control", plot),
            Field("GAINFB", zeros3, "vec", "diag", "control"),
            Field("gainp", 0.0, "real", "data", "control"),
            Field("gkp", 0.0, "real", "diag", "control"),
            Field("gkphi", 0.0, "real", "diag", "control"),
            Field("zetlagr", 0.0, "real", "data", "control"),
            Field("qqcomx", 0.0, "real", "data", "control", plot),
            Field("rrcomx", 0.0, "real", "data", "control", plot),
            Field("zrate", 0.0, "real", "diag", "control"),
            Field("grate", 0.0, "real", "diag", "control"),
            Field("wnlagr", 0.0, "real", "diag", "control"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        pass

    def control_roll(self, vehicle):
        store = vehicle.store
        dplimx = store.get("dplimx")
        phicomx = store.get("phicomx")
        wrcl = store.get("wrcl")
        zrcl = store.get("zrcl")
        dlp = store.get("dlp")
        dld = store.get("dld")
        wbecb = store.get("WBECB")
        phiblcx = store.get("phiblcx")
        gkp = (2. * zrcl * wrcl + dlp) / dld
        gkphi = wrcl * wrcl / dld
        pp = wbecb[0]
        ephi = gkphi * (phicomx - phiblcx) * RAD
        dpc = ephi - gkp * pp
        dpcx = dpc * DEG
        if abs(dpcx) > dplimx:
            dpcx = dplimx * _sign(dpcx)
        store.set("dpcx", dpcx)
        store.set("gkp", gkp)
        store.set("gkphi", gkphi)

    def control_rate(self, vehicle):
        store = vehicle.store
        zetlagr = store.get("zetlagr")
        qqcomx = store.get("qqcomx")
        rrcomx = store.get("rrcomx")
        dvbe = store.get("dvbe")
        dna = store.get("dna")
        dnd = store.get("dnd")
        dma = store.get("dma")
        dmq = store.get("dmq")
        dmd = store.get("dmd")
        wbecb = store.get("WBECB")
        zrate = dna / dvbe - dma * dnd / (dvbe * dmd)
        aa = dna / dvbe - dmq
        bb = -dma - dmq * dna / dvbe
        dum1 = (aa - 2 * zetlagr * zetlagr * zrate)
        dum2 = aa * aa - 4 * zetlagr * zetlagr * bb
        radix = dum1 * dum1 - dum2
        if radix < 0:
            radix = SMALL
        if abs(dmd) < SMALL:
            dmd = SMALL * _sign(dmd)
        grate = -(-dum1 + sqrt(radix)) / dmd
        dum3 = grate * dmd * zrate
        radix = bb + dum3
        if radix < 0:
            radix = SMALL
        wnlagr = sqrt(radix)
        qq = wbecb[1]
        rr = wbecb[2]
        dqcx = DEG * grate * qq - qqcomx
        drcx = DEG * grate * rr - rrcomx
        store.set("dqcx", dqcx)
        store.set("drcx", drcx)
        store.set("zrate", zrate)
        store.set("grate", grate)
        store.set("wnlagr", wnlagr)

    def terminate(self, vehicle, ctx):
        pass
