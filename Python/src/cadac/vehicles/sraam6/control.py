from math import atan2, cos, sin, sqrt

from cadac.constants import AGRAV, DEG, RAD
from cadac.kernel.integrate import integrate
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

    def control_accel(self, vehicle, ctx):
        store = vehicle.store
        alimit = store.get("alimit")
        gainp = store.get("gainp")
        factwacl = store.get("factwacl")
        factzacl = store.get("factzacl")
        pdynmc = store.get("pdynmc")
        ancomx = store.get("ancomx")
        alcomx = store.get("alcomx")
        dna = store.get("dna")
        dma = store.get("dma")
        dmq = store.get("dmq")
        dmd = store.get("dmd")
        fspb = store.get("FSPB")
        dvbe = store.get("dvbe")
        qq = store.get("qq")
        rr = store.get("rr")
        yyd = store.get("yyd")
        yy = store.get("yy")
        zzd = store.get("zzd")
        zz = store.get("zz")
        dt = ctx.int_step

        aa = sqrt(alcomx * alcomx + ancomx * ancomx)
        if aa > alimit:
            aa = alimit
        if abs(ancomx) < SMALL and abs(alcomx) < SMALL:
            phi = 0.0
        else:
            phi = atan2(ancomx, alcomx)
        alcomx = aa * cos(phi)
        ancomx = aa * sin(phi)

        wacl = (0.013 * sqrt(pdynmc) + 7.1) * (factwacl + 1)
        zacl = (0.559e-3 * sqrt(pdynmc) + 0.232) * (factzacl + 1)
        pacl = 14

        gainfb3 = wacl * wacl * pacl / (dna * dmd)
        gainfb2 = (2.0 * zacl * wacl + pacl + dmq - dna / dvbe) / dmd
        gainfb1 = (
            wacl * wacl
            + 2.0 * zacl * wacl * pacl
            + dma
            + dmq * dna / dvbe
            - gainfb2 * dmd * dna / dvbe
        ) / (dna * dmd) - gainp

        fspb3 = fspb[2]
        zzd_new = AGRAV * ancomx + fspb3
        zz = integrate(zzd_new, zzd, zz, dt)
        zzd = zzd_new
        dqc = -gainfb1 * (-fspb3) - gainfb2 * qq + gainfb3 * zz
        dqcx = dqc * DEG

        fspb2 = fspb[1]
        yyd_new = AGRAV * alcomx - fspb2
        yy = integrate(yyd_new, yyd, yy, dt)
        yyd = yyd_new
        drc = -gainfb1 * fspb2 - gainfb2 * rr + gainfb3 * yy
        drcx = drc * DEG

        store.set("yyd", yyd)
        store.set("yy", yy)
        store.set("zzd", zzd)
        store.set("zz", zz)
        store.set("dqcx", dqcx)
        store.set("drcx", drcx)
        store.set("wacl", wacl)
        store.set("zacl", zacl)
        store.set("pacl", pacl)
        store.set("GAINFB", (gainfb1, gainfb2, gainfb3))

    def terminate(self, vehicle, ctx):
        pass
