from math import atan2, cos, fabs, sin, sqrt

from cadac.constants import AGRAV, DEG, RAD
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field

SMALL = 1e-7


def _sign(variable):
    if variable < 0:
        return -1
    return 1


class Sam6Control:
    name = "control"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("plot",)
        scrn = ("scrn",)
        for field in (
            Field("maut", 0, "int", "data", "control"),
            Field("mfreeze", 0, "int", "data", "control"),
            Field("wacl", 0.0, "real", "diag", "control", plot),
            Field("zacl", 0.0, "real", "diag", "control", plot),
            Field("pacl", 0.0, "real", "diag", "control", plot),
            Field("alimitx", 0.0, "real", "data", "control"),
            Field("dqlimx", 0.0, "real", "data", "control"),
            Field("drlimx", 0.0, "real", "data", "control"),
            Field("dplimx", 0.0, "real", "data", "control"),
            Field("phicomx", 0.0, "real", "data", "control"),
            Field("wrcl", 0.0, "real", "diag", "control", plot),
            Field("zrcl", 0.0, "real", "data", "control"),
            Field("yyd", 0.0, "real", "state", "control"),
            Field("yy", 0.0, "real", "state", "control"),
            Field("zzd", 0.0, "real", "state", "control"),
            Field("zz", 0.0, "real", "state", "control"),
            Field("dpcx", 0.0, "real", "out", "control", scrn),
            Field("dqcx", 0.0, "real", "out", "control", scrn),
            Field("drcx", 0.0, "real", "out", "control", scrn),
            Field("tp", 0.0, "real", "data", "control"),
            Field("GAINFB", zeros3, "vec", "diag", "control"),
            Field("gainp", 0.0, "real", "data", "control"),
            Field("dqcx_rcs", 0.0, "real", "out", "control"),
            Field("drcx_rcs", 0.0, "real", "out", "control"),
            Field("qqcomx", 0.0, "real", "data", "control"),
            Field("rrcomx", 0.0, "real", "data", "control"),
            Field("gkp", 0.0, "real", "diag", "control"),
            Field("gkphi", 0.0, "real", "diag", "control"),
            Field("factwrcl", 0.0, "real", "data", "control"),
            Field("zetlagr", 0.0, "real", "data", "control"),
            Field("zrate", 0.0, "real", "diag", "control", plot),
            Field("grate", 0.0, "real", "diag", "control"),
            Field("wnlagr", 0.0, "real", "diag", "control", plot),
            Field("wacl_bias", 0.0, "real", "data", "control"),
            Field("pacl_bias", 0.0, "real", "data", "control"),
            Field("zacl_bias", 0.0, "real", "data", "control"),
            Field("ancomx_test", 0.0, "real", "data", "control", plot),
            Field("alcomx_test", 0.0, "real", "data", "control", plot),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        maut = store.get("maut")
        if maut == 0:
            return
        if maut == 4:
            raise ValueError(f"unknown maut {maut}")
        self.control_roll(vehicle)
        if maut == 2:
            self.control_rate(vehicle)
        if maut == 3:
            self.control_accel(vehicle, ctx.int_step)

    def control_roll(self, vehicle):
        store = vehicle.store
        phicomx = store.get("phicomx")
        zrcl = store.get("zrcl")
        tp = store.get("tp")
        factwrcl = store.get("factwrcl")
        dlp = store.get("dlp")
        dld = store.get("dld")
        thtblcx = store.get("thtblcx")
        wbecb = store.get("WBECB")
        phiblcx = store.get("phiblcx")
        wrcl = -0.8 * dlp * (1 + factwrcl)
        gkp = (2 * zrcl * wrcl + dlp) / dld
        gkphi = wrcl * wrcl / dld
        pp = wbecb[0]
        ephi = gkphi * (phicomx - phiblcx) * RAD
        dpc = ephi - gkp * pp
        if abs(thtblcx) > 88:
            kp = (1 / tp + dlp) / dld
            dpcx = kp * pp * DEG
        else:
            dpcx = dpc * DEG
        store.set("dpcx", dpcx)
        store.set("wrcl", wrcl)
        store.set("gkp", gkp)
        store.set("gkphi", gkphi)

    def control_rate(self, vehicle):
        store = vehicle.store
        zetlagr = store.get("zetlagr")
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
        dum1 = aa - 2 * zetlagr * zetlagr * zrate
        dum2 = aa * aa - 4 * zetlagr * zetlagr * bb
        radix = dum1 * dum1 - dum2
        if radix < 0:
            radix = 0
        if abs(dmd) < SMALL:
            dmd = SMALL * _sign(dmd)
        grate = (-dum1 + sqrt(radix)) / dmd
        dum3 = grate * dmd * zrate
        radix = bb + dum3
        if radix < 0:
            radix = 0
        wnlagr = sqrt(radix)
        qq = wbecb[1]
        rr = wbecb[2]
        dqcx = DEG * grate * qq
        drcx = DEG * grate * rr
        store.set("dqcx", dqcx)
        store.set("drcx", drcx)
        store.set("dqcx_rcs", dqcx)
        store.set("drcx_rcs", drcx)
        store.set("zrate", zrate)
        store.set("grate", grate)
        store.set("wnlagr", wnlagr)

    def control_accel(self, vehicle, int_step):
        store = vehicle.store
        alimitx = store.get("alimitx")
        gainp = store.get("gainp")
        wacl_bias = store.get("wacl_bias")
        pacl_bias = store.get("pacl_bias")
        zacl_bias = store.get("zacl_bias")
        ancomx = store.get("ancomx") + store.get("ancomx_test")
        alcomx = store.get("alcomx") + store.get("alcomx_test")
        dvbe = store.get("dvbe")
        dna = store.get("dna")
        dma = store.get("dma")
        dmq = store.get("dmq")
        dmd = store.get("dmd")
        dlnd = store.get("dlnd")
        realq1 = store.get("realq1")
        realq2 = store.get("realq2")
        dnr = store.get("dnr")
        dyb = store.get("dyb")
        dnb = store.get("dnb")
        fspcb = store.get("FSPCB")
        wbecb = store.get("WBECB")
        yyd = store.get("yyd")
        yy = store.get("yy")
        zzd = store.get("zzd")
        zz = store.get("zz")
        aa = sqrt(alcomx * alcomx + ancomx * ancomx)
        if aa > alimitx:
            aa = alimitx
        if fabs(ancomx) < SMALL and fabs(alcomx) < SMALL:
            phi = 0
        else:
            phi = atan2(ancomx, alcomx)
        alcomx = aa * cos(phi)
        ancomx = aa * sin(phi)
        zacl = 0.7 * (1 + zacl_bias)
        wacl = fabs(realq1) * (1 + wacl_bias)
        pacl = (fabs(realq2) + 35) * (1 + pacl_bias)
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
        gainfb3 = -wacl * wacl * pacl / (dyb * dlnd)
        gainfb2 = (2 * zacl * wacl + pacl + dnr + dyb / dvbe) / dlnd
        gainfb1 = (
            -wacl * wacl
            - 2 * zacl * wacl * pacl
            + dnb
            + dnr * dyb / dvbe
            - gainfb2 * dyb * dlnd / dvbe
        ) / (dyb * dlnd) - gainp
        rr = wbecb[2]
        fspb2 = fspcb[1]
        yyd_new = AGRAV * alcomx - fspb2
        yy = integrate(yyd_new, yyd, yy, int_step)
        yyd = yyd_new
        drc = -gainfb1 * fspb2 - gainfb2 * rr + gainfb3 * yy + gainp * yyd
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
