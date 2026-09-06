from math import atan2, cos, sin, sqrt

from cadac.constants import AGRAV, DEG, RAD
from cadac.kernel.integrate import integrate
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
        store = vehicle.store
        maut = store.get("maut")
        if maut == 0:
            return
        if maut not in (1, 2, 3):
            raise ValueError(f"unknown maut {maut}")
        self.control_roll(vehicle)
        if maut == 2:
            self.control_rate(vehicle)
        if maut == 3:
            self.control_accel(vehicle, ctx.int_step)

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

    def control_accel(self, vehicle, int_step):
        store = vehicle.store
        wacl = store.get("wacl")
        zacl = store.get("zacl")
        pacl = store.get("pacl")
        alimit = store.get("alimit")
        dqlimx = store.get("dqlimx")
        drlimx = store.get("drlimx")
        gainp = store.get("gainp")
        dvbe = store.get("dvbe")
        ancomx = store.get("ancomx")
        alcomx = store.get("alcomx")
        dna = store.get("dna")
        dma = store.get("dma")
        dmq = store.get("dmq")
        dmd = store.get("dmd")
        fspcb = store.get("FSPCB")
        wbecb = store.get("WBECB")
        yyd = store.get("yyd")
        yy = store.get("yy")
        zzd = store.get("zzd")
        zz = store.get("zz")

        aa = sqrt(alcomx * alcomx + ancomx * ancomx)
        if aa > alimit:
            aa = alimit
        if abs(ancomx) < SMALL and abs(alcomx) < SMALL:
            phi = 0.0
        else:
            phi = atan2(ancomx, alcomx)
        alcomx = aa * cos(phi)
        ancomx = aa * sin(phi)

        gainfb3 = wacl * wacl * pacl / (dna * dmd)
        gainfb2 = (2 * zacl * wacl + pacl + dmq - dna / dvbe) / dmd
        gainfb1 = (
            wacl * wacl
            + 2 * zacl * wacl * pacl
            + dma
            + dmq * dna / dvbe
            - gainfb2 * dna * dmd / dvbe
        ) / (dna * dmd) - gainp

        qq = wbecb[1]
        fspb3 = fspcb[2]
        zzd_new = AGRAV * ancomx + fspb3
        zz = integrate(zzd_new, zzd, zz, int_step)
        zzd = zzd_new
        dqc = -gainfb1 * (-fspb3) - gainfb2 * qq + gainfb3 * zz + gainp * zzd
        dqcx = dqc * DEG

        rr = wbecb[2]
        fspb2 = fspcb[1]
        yyd_new = AGRAV * alcomx - fspb2
        yy = integrate(yyd_new, yyd, yy, int_step)
        yyd = yyd_new
        drc = -gainfb1 * fspb2 - gainfb2 * rr + gainfb3 * yy + gainp * yyd
        drcx = drc * DEG

        if abs(dqcx) > dqlimx:
            dqcx = dqlimx * _sign(dqcx)
        if abs(drcx) > drlimx:
            drcx = drlimx * _sign(drcx)

        store.set("yyd", yyd)
        store.set("yy", yy)
        store.set("zzd", zzd)
        store.set("zz", zz)
        store.set("dqcx", dqcx)
        store.set("drcx", drcx)
        store.set("GAINFB", (gainfb1, gainfb2, gainfb3))

    def terminate(self, vehicle, ctx):
        pass
