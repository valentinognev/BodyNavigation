from math import cos, sqrt

import numpy as np

from cadac.constants import AGRAV, DEG, RAD
from cadac.kernel.integrate import integrate
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
        store = vehicle.store
        maut = store.get("maut")
        if maut == 0:
            return
        mauty = maut // 10
        mautp = maut % 10
        # Yaw 2 is the existing rate path only beside a ported pitch digit
        # (2/3/4/5), which keeps maut 22/24 and rejects yaw-rate-only maut 20.
        # Pitch 1 is roll-only (no pitch path); pitch 2 is pitch-rate SAS.
        yaw_ok = mauty in (0, 3, 4) or (mauty == 2 and mautp in (2, 3, 4, 5))
        pitch_ok = mautp in (0, 1, 2, 3, 4, 5)
        if not (yaw_ok and pitch_ok):
            raise ValueError(f"unknown maut {maut}")

        delacx = 0.0
        delecx = 0.0
        delrcx = 0.0
        mroll = store.get("mroll")
        dalimx = store.get("dalimx")
        delimx = store.get("delimx")
        drlimx = store.get("drlimx")
        philimx = store.get("philimx")
        ancomx = store.get("ancomx")
        phicomx = store.get("phicomx")
        pcomx = store.get("pcomx")
        qcomx = store.get("qcomx")
        rcomx = store.get("rcomx")
        thtvdcomx = store.get("thtvdcomx")

        if mauty == 2:
            delrcx = self.control_yaw_rate(vehicle, rcomx)
        if mautp == 2:
            delecx = self.control_pitch_rate(vehicle, qcomx)
        if mauty == 3:
            phicomx = self.control_lateral_accel(vehicle, store.get("alcomx"))
            delrcx = self.control_yaw_rate(vehicle, rcomx)
        if mautp == 3:
            gmax = store.get("gmax")
            gminx = store.get("gminx")
            if ancomx > gmax:
                ancomx = gmax
            if ancomx < gminx:
                ancomx = gminx
            delecx = self.control_normal_accel(vehicle, ancomx, ctx.int_step)
        if mautp == 4:
            delecx = self.control_gamma(vehicle, thtvdcomx)
        if mauty == 4:
            phicomx = self.control_heading(vehicle, store.get("psivdcomx"))
            delrcx = self.control_yaw_rate(vehicle, rcomx)
        if mautp == 5:
            ancomx = self.control_altitude(vehicle, store.get("altcom"))
            gmax = store.get("gmax")
            gminx = store.get("gminx")
            if ancomx > gmax:
                ancomx = gmax
            if ancomx < gminx:
                ancomx = gminx
            delecx = self.control_normal_accel(vehicle, ancomx, ctx.int_step)

        if mroll == 0:
            if abs(phicomx) > philimx:
                phicomx = philimx * _sign(phicomx)
            delacx = self.control_roll(vehicle, phicomx)
        elif mroll == 1:
            delacx = self.control_roll_rate(vehicle, pcomx)
        else:
            raise ValueError(f"unknown mroll {mroll}")

        if abs(delacx) > dalimx:
            delacx = dalimx * _sign(delacx)
        if abs(delecx) > delimx:
            delecx = delimx * _sign(delecx)
        if abs(delrcx) > drlimx:
            delrcx = drlimx * _sign(delrcx)

        store.set("delacx", delacx)
        store.set("delecx", delecx)
        store.set("delrcx", delrcx)
        store.set("ancomx", ancomx)
        store.set("phicomx", phicomx)

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

    def control_gamma(self, vehicle, thtvdcomx):
        store = vehicle.store
        pgam = store.get("pgam")
        wgam = store.get("wgam")
        zgam = store.get("zgam")
        dvbe = store.get("dvbe")
        dla = store.get("dla")
        dlde = store.get("dlde")
        dma = store.get("dma")
        dmq = store.get("dmq")
        dmde = store.get("dmde")
        qqcx = store.get("qqcx")
        dvbec = store.get("dvbec")
        thtvdcx = store.get("thtvdcx")
        thtbdcx = store.get("thtbdcx")

        if dvbec == 0:
            dvbec = dvbe

        aa = np.array(
            [
                [dmq, dma, -dma],
                [1.0, 0.0, 0.0],
                [0.0, dla / dvbec, -dla / dvbec],
            ],
            dtype=float,
        )
        bb = np.array([dmde, 0.0, dlde / dvbec], dtype=float)

        am = 2.0 * zgam * wgam + pgam
        bm = wgam * wgam + 2.0 * zgam * wgam * pgam
        cm = wgam * wgam * pgam
        v11 = dmde
        v12 = 0.0
        v13 = dlde / dvbec
        v21 = dmde * dla / dvbec - dlde * dma / dvbec
        v22 = dmde
        v23 = -dmq * dlde / dvbec
        v31 = 0.0
        v32 = v21
        v33 = v21
        dp = np.array(
            [
                [v11, v12, v13],
                [v21, v22, v23],
                [v31, v32, v33],
            ],
            dtype=float,
        )
        dd = np.array(
            [am + dmq - dla / dvbec, bm + dma + dmq * dla / dvbec, cm],
            dtype=float,
        )
        gaingam = np.linalg.inv(dp) @ dd
        dum33 = aa - np.outer(bb, gaingam)
        dum3 = np.linalg.inv(dum33) @ bb
        hh = np.array([0.0, 0.0, 1.0], dtype=float)
        gainff = -1.0 / (hh @ dum3)

        thtc = gainff * thtvdcomx * RAD
        qqf = gaingam[0] * qqcx * RAD
        thtbgf = gaingam[1] * thtbdcx * RAD
        thtugf = gaingam[2] * thtvdcx * RAD
        delec = thtc - (qqf + thtbgf + thtugf)
        delecx = delec * DEG

        store.set("GAINGAM", gaingam)
        store.set("gainff", gainff)
        return delecx

    def control_normal_accel(self, vehicle, ancomx, int_step):
        store = vehicle.store
        waclp = store.get("waclp")
        zaclp = store.get("zaclp")
        paclp = store.get("paclp")
        gainp = store.get("gainp")
        dla = store.get("dla")
        dma = store.get("dma")
        dmq = store.get("dmq")
        dmde = store.get("dmde")
        dvbec = store.get("dvbec")
        qqcx = store.get("qqcx")
        fspcb = store.get("FSPCB")
        zzd = store.get("zzd")
        zz = store.get("zz")

        gainfb3 = waclp * waclp * paclp / (dla * dmde)
        gainfb2 = (2.0 * zaclp * waclp + paclp + dmq - dla / dvbec) / dmde
        gainfb1 = (
            waclp * waclp
            + 2.0 * zaclp * waclp * paclp
            + dma
            + dmq * dla / dvbec
            - gainfb2 * dmde * dla / dvbec
        ) / (dla * dmde) - gainp

        fspb3 = fspcb[2]
        zzd_new = AGRAV * ancomx + fspb3
        zz = integrate(zzd_new, zzd, zz, int_step)
        zzd = zzd_new
        dqc = -gainfb1 * (-fspb3) - gainfb2 * qqcx * RAD + gainfb3 * zz + gainp * zzd
        delecx = dqc * DEG

        store.set("zzd", zzd)
        store.set("zz", zz)
        store.set("GAINFP", (gainfb1, gainfb2, gainfb3))
        return delecx

    def control_lateral_accel(self, vehicle, alcomx):
        store = vehicle.store
        gainl = store.get("gainl")
        fspcb = store.get("FSPCB")
        fspb3 = fspcb[2]
        phicomx = -DEG * gainl * alcomx * _sign(fspb3)
        return phicomx

    def control_heading(self, vehicle, psivdcomx):
        store = vehicle.store
        wrcl = store.get("wrcl")
        zrcl = store.get("zrcl")
        facthead = store.get("facthead")
        grav = store.get("grav")
        dvbec = store.get("dvbec")
        psivdcx = store.get("psivdcx")
        gainpsi = (dvbec / grav) * zrcl * wrcl * (1.0 - zrcl * zrcl) * (1.0 + facthead)
        phicomx = gainpsi * (psivdcomx - psivdcx)
        store.set("gainpsi", gainpsi)
        return phicomx

    def control_altitude(self, vehicle, altcom):
        store = vehicle.store
        gainalt = store.get("gainalt")
        gainaltrate = store.get("gainaltrate")
        grav = store.get("grav")
        altc = store.get("altc")
        vbecd = store.get("VBECD")
        phibdcx = store.get("phibdcx")
        altrate = -vbecd[2]
        eh = gainalt * (altcom - altc)
        if phibdcx == 0:
            phibdcx = SMALL
        ancomx = (1.0 / cos(phibdcx * RAD)) * (gainaltrate * (eh - altrate) + grav) / AGRAV
        store.set("altrate", altrate)
        return ancomx

    def terminate(self, vehicle, ctx):
        pass
