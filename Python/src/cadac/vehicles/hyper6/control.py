from math import sqrt

from cadac.constants import DEG, RAD
from cadac.kernel.state import Field

SMALL = 1.0e-7


def _sign(variable):
    if variable < 0:
        return -1
    return 1


class Hyper6Control:
    name = "control"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("plot",)
        for field in (
            Field("maut", 0, "int", "data", "control"),
            Field("mroll", 0, "int", "data", "control"),
            Field("mfreeze", 0, "int", "data", "control"),
            Field("waclp", 0.0, "real", "data", "control"),
            Field("zaclp", 0.0, "real", "data", "control"),
            Field("paclp", 0.0, "real", "data", "control"),
            Field("alimitx", 0.0, "real", "data", "control"),
            Field("dalimx", 0.0, "real", "data", "control"),
            Field("delimx", 0.0, "real", "data", "control"),
            Field("drlimx", 0.0, "real", "data", "control"),
            Field("philimx", 0.0, "real", "data", "control"),
            Field("wrcl", 0.0, "real", "data", "control"),
            Field("zrcl", 0.0, "real", "data", "control"),
            Field("yyd", 0.0, "real", "state", "control"),
            Field("yy", 0.0, "real", "state", "control"),
            Field("zzd", 0.0, "real", "state", "control"),
            Field("zz", 0.0, "real", "state", "control"),
            Field("delacx", 0.0, "real", "out", "control", plot),
            Field("delecx", 0.0, "real", "out", "control", plot),
            Field("delrcx", 0.0, "real", "out", "control", plot),
            Field("alcomx", 0.0, "real", "data", "control", plot),
            Field("ancomx", 0.0, "real", "data", "control", plot),
            Field("GAINFP", zeros3, "vec", "diag", "control"),
            Field("gainp", 0.0, "real", "data", "control"),
            Field("gainl", 0.0, "real", "data", "control"),
            Field("altcom", 0.0, "real", "data", "control"),
            Field("gainalt", 0.0, "real", "data", "control"),
            Field("gainaltrate", 0.0, "real", "data", "control"),
            Field("altrate", 0.0, "real", "diag", "control"),
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
            Field("tp", 0.0, "real", "data", "control"),
            Field("zetlagr", 0.0, "real", "data", "control"),
            Field("psivdcomx", 0.0, "real", "data", "control"),
            Field("facthead", 0.0, "real", "data", "control"),
            Field("gainpsi", 0.0, "real", "diag", "control"),
            Field("phicomx", 0.0, "real", "data", "control", plot),
            Field("pcomx", 0.0, "real", "data", "control"),
            Field("qcomx", 0.0, "real", "data", "control"),
            Field("rcomx", 0.0, "real", "data", "control"),
            Field("thtvdcomx", 0.0, "real", "data", "control"),
            Field("zrate", 0.0, "real", "diag", "control"),
            Field("grate", 0.0, "real", "diag", "control"),
            Field("wnlagr", 0.0, "real", "diag", "control"),
            Field("pgam", 0.0, "real", "data", "control"),
            Field("wgam", 0.0, "real", "data", "control"),
            Field("zgam", 0.0, "real", "data", "control"),
            Field("GAINGAM", zeros3, "vec", "diag", "control"),
            Field("gainff", 0.0, "real", "diag", "control"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        pass

    def control_roll(self, vehicle, phicomx):
        store = vehicle.store
        wrcl = store.get("wrcl")
        zrcl = store.get("zrcl")
        phibdcx = store.get("phibdcx")
        ppcx = store.get("ppcx")
        dllp = store.get("dllp")
        dllda = store.get("dllda")
        gkp = (2.0 * zrcl * wrcl + dllp) / dllda
        gkphi = wrcl * wrcl / dllda
        ephi = gkphi * (phicomx - phibdcx) * RAD
        dpc = ephi - gkp * ppcx * RAD
        delacx = dpc * DEG
        store.set("gkp", gkp)
        store.set("gkphi", gkphi)
        return delacx

    def control_roll_rate(self, vehicle, pcomx):
        store = vehicle.store
        tp = store.get("tp")
        dllp = store.get("dllp")
        dllda = store.get("dllda")
        ppcx = store.get("ppcx")
        kp = (1 / tp + dllp) / dllda
        delacx = kp * (pcomx - ppcx)
        return delacx

    def control_pitch_rate(self, vehicle, qcomx):
        store = vehicle.store
        zetlagr = store.get("zetlagr")
        dla = store.get("dla")
        dlde = store.get("dlde")
        dma = store.get("dma")
        dmq = store.get("dmq")
        dmde = store.get("dmde")
        qqcx = store.get("qqcx")
        dvbec = store.get("dvbec")
        zrate = dla / dvbec - dma * dlde / (dvbec * dmde)
        aa = dla / dvbec - dmq
        bb = -dma - dmq * dla / dvbec
        dum1 = aa - 2.0 * zetlagr * zetlagr * zrate
        dum2 = aa * aa - 4.0 * zetlagr * zetlagr * bb
        radix = dum1 * dum1 - dum2
        if radix < 0.0:
            radix = 0.0
        if abs(dmde) < SMALL:
            dmde = SMALL * _sign(dmde)
        grate = (-dum1 + sqrt(radix)) / (-dmde)
        delecx = grate * (qqcx - qcomx)
        return delecx

    def control_yaw_rate(self, vehicle, rcomx):
        store = vehicle.store
        zetlagr = store.get("zetlagr")
        dyb = store.get("dyb")
        dydr = store.get("dydr")
        dnb = store.get("dnb")
        dnr = store.get("dnr")
        dndr = store.get("dndr")
        rrcx = store.get("rrcx")
        dvbec = store.get("dvbec")
        zrate = -dyb / dvbec + dnb * dydr / (dvbec * dndr)
        aa = -dyb / dvbec - dnr
        bb = dnb + dyb * dnr / dvbec
        dum1 = aa - 2.0 * zetlagr * zetlagr * zrate
        dum2 = aa * aa - 4.0 * zetlagr * zetlagr * bb
        radix = dum1 * dum1 - dum2
        if radix < 0.0:
            radix = 0.0
        if abs(dndr) < SMALL:
            dndr = SMALL * _sign(dndr)
        grate = (-dum1 + sqrt(radix)) / (-dndr)
        dum3 = grate * dndr * zrate
        radix = bb + dum3
        if radix < 0.0:
            radix = 0.0
        wnlagr = sqrt(radix)
        delrcx = grate * (rrcx - rcomx)
        store.set("zrate", zrate)
        store.set("grate", grate)
        store.set("wnlagr", wnlagr)
        return delrcx

    def terminate(self, vehicle, ctx):
        pass
