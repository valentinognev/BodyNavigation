from math import sqrt

from cadac.constants import DEG, RAD
from cadac.kernel.state import Field

SMALL = 1e-7


def _sign(variable):
    if variable < 0.0:
        return -1
    return 1


class Sraam6Control:
    name = "control"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("maut", 0, "int", "data", "control"),
            Field("mfreeze", 0, "int", "data", "control"),
            Field("wacl", 0.0, "real", "data", "control"),
            Field("zacl", 0.0, "real", "data", "control"),
            Field("pacl", 0.0, "real", "data", "control"),
            Field("alimit", 0.0, "real", "data", "control"),
            Field("dqlimx", 0.0, "real", "data", "control"),
            Field("drlimx", 0.0, "real", "data", "control"),
            Field("dplimx", 0.0, "real", "data", "control"),
            Field("phicomx", 0.0, "real", "data", "control"),
            Field("wrcl", 0.0, "real", "data", "control"),
            Field("zrcl", 0.0, "real", "data", "control"),
            Field("yyd", 0.0, "real", "state", "control"),
            Field("yy", 0.0, "real", "state", "control"),
            Field("zzd", 0.0, "real", "state", "control"),
            Field("zz", 0.0, "real", "state", "control"),
            Field("dpcx", 0.0, "real", "out", "control"),
            Field("dqcx", 0.0, "real", "out", "control"),
            Field("drcx", 0.0, "real", "out", "control"),
            Field("GAINFB", zeros3, "vec", "diag", "control"),
            Field("gainp", 0.0, "real", "data", "control"),
            Field("gkp", 0.0, "real", "diag", "control"),
            Field("gkphi", 0.0, "real", "diag", "control"),
            Field("fspb2m", 0.0, "real", "diag", "control"),
            Field("fspb2mt", 0.0, "real", "diag", "control"),
            Field("fspb3m", 0.0, "real", "diag", "control"),
            Field("fspb3mt", 0.0, "real", "diag", "control"),
            Field("qqxm", 0.0, "real", "diag", "control"),
            Field("qqxmt", 0.0, "real", "diag", "control"),
            Field("rrxm", 0.0, "real", "diag", "control"),
            Field("rrxmt", 0.0, "real", "diag", "control"),
            Field("dqcxm", 0.0, "real", "diag", "control"),
            Field("dqcxmt", 0.0, "real", "diag", "control"),
            Field("drcxm", 0.0, "real", "diag", "control"),
            Field("drcxmt", 0.0, "real", "diag", "control"),
            Field("isetc2", 0.0, "real", "init", "control"),
            Field("factwacl", 0.0, "real", "data", "control"),
            Field("factzacl", 0.0, "real", "data", "control"),
            Field("zetlagr", 0.0, "real", "data", "control"),
            Field("ratelimx", 0.0, "real", "data", "control"),
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
        phicomx = store.get("phicomx")
        wrcl = store.get("wrcl")
        zrcl = store.get("zrcl")
        phiblx = store.get("phiblx")
        dlp = store.get("dlp")
        dld = store.get("dld")
        pp = store.get("pp")
        gkp = (2.0 * zrcl * wrcl + dlp) / dld
        gkphi = wrcl * wrcl / dld
        ephi = gkphi * (phicomx - phiblx) * RAD
        dpc = ephi - gkp * pp
        dpcx = dpc * DEG
        store.set("dpcx", dpcx)
        store.set("gkp", gkp)
        store.set("gkphi", gkphi)

    def control_rate(self, vehicle):
        store = vehicle.store
        zetlagr = store.get("zetlagr")
        qq = store.get("qq")
        rr = store.get("rr")
        dvbe = store.get("dvbe")
        dna = store.get("dna")
        dnd = store.get("dnd")
        dma = store.get("dma")
        dmq = store.get("dmq")
        dmd = store.get("dmd")
        zrate = dna / dvbe - dma * dnd / (dvbe * dmd)
        aa = dna / dvbe - dmq
        bb = -dma - dmq * dna / dvbe
        dum1 = aa - 2.0 * zetlagr * zetlagr * zrate
        dum2 = aa * aa - 4.0 * zetlagr * zetlagr * bb
        radix = dum1 * dum1 - dum2
        if radix < 0.0:
            radix = 0.0
        if abs(dmd) < SMALL:
            dmd = SMALL * _sign(dmd)
        grate = (-dum1 + sqrt(radix)) / (-dmd)
        dum3 = grate * dmd * zrate
        radix = bb + dum3
        if radix < 0.0:
            radix = 0.0
        wnlagr = sqrt(radix)
        dqcx = DEG * grate * qq
        drcx = DEG * grate * rr
        store.set("dqcx", dqcx)
        store.set("drcx", drcx)
        store.set("zrate", zrate)
        store.set("grate", grate)
        store.set("wnlagr", wnlagr)

    def terminate(self, vehicle, ctx):
        pass
