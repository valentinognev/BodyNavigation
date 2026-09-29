from math import atan2, cos, exp, fabs, sin, sqrt, tan

import numpy as np

from cadac.constants import AGRAV, DEG, RAD
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr, polar_from_cart, skew

SMALL = 1e-7


class Sam6Guidance:
    name = "guidance"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("plot",)
        scrn_plot = ("scrn", "plot")
        for field in (
            Field("mguide", 0, "int", "data", "guidance"),
            Field("ancomx", 0.0, "real", "out", "guidance", scrn_plot),
            Field("alcomx", 0.0, "real", "out", "guidance", scrn_plot),
            Field("gnav_comp", 0.0, "real", "diag", "guidance", plot),
            Field("grav_bias", 1.0, "real", "data", "guidance"),
            Field("apnyx", 0.0, "real", "diag", "guidance"),
            Field("apnzx", 0.0, "real", "diag", "guidance"),
            Field("adely", 0.0, "real", "diag", "guidance"),
            Field("adelz", 0.0, "real", "diag", "guidance"),
            Field("allx", 0.0, "real", "diag", "guidance", plot),
            Field("annx", 0.0, "real", "diag", "guidance", plot),
            Field("epchta", 0.0, "real", "save", "guidance"),
            Field("gnav", 0.0, "real", "data", "guidance"),
            Field("gnd", 0.0, "real", "state", "guidance"),
            Field("gn", 0.0, "real", "state", "guidance", plot),
            Field("tgnav", 0.0, "real", "data", "guidance"),
            Field("dcvel", 0.0, "real", "diag", "guidance", plot),
            Field("line_gain", 0.0, "real", "data", "guidance"),
            Field("nl_gain_fact", 0.0, "real", "data", "guidance"),
            Field("decrement", 0.0, "real", "data", "guidance"),
            Field("psiflx", 0.0, "real", "diag", "guidance", plot),
            Field("thtflx", 0.0, "real", "data", "guidance"),
            Field("ip_sltrange", 0.0, "real", "diag", "guidance", plot),
            Field("nl_gain", 0.0, "real", "diag", "guidance"),
            Field("VBEO", zeros3, "vec", "diag", "guidance"),
            Field("VBEF", zeros3, "vec", "diag", "guidance"),
            Field("SIBLC", zeros3, "vec", "out", "guidance"),
            Field("WOELC", zeros3, "vec", "out", "guidance"),
            Field("tgoc", 0.0, "real", "diag", "guidance"),
            Field("dtbc", 0.0, "real", "diag", "guidance"),
            Field("dvtbc", 0.0, "real", "diag", "guidance", plot),
            Field("psiobcx", 0.0, "real", "diag", "guidance"),
            Field("thtobcx", 0.0, "real", "diag", "guidance"),
            Field("UTBLC", zeros3, "vec", "out", "guidance"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mguide = store.get("mguide")
        thtflx = store.get("thtflx")
        gmax = store.get("gmax")
        guid_mid = mguide // 10
        guid_term = mguide % 10

        acbx = np.zeros(3)
        siblc = np.zeros(3)
        psiflx = 0.0
        if guid_mid >= 2:
            siel = _ip_siel(ctx)
            sbelc = store.get("SBELC")
            siblc = siel - sbelc
            if guid_mid == 2:
                sbel0 = np.array(
                    [store.get("sbel1"), store.get("sbel2"), store.get("sbel3")],
                    dtype=float,
                )
                sibl0 = siel - sbel0
                polar = polar_from_cart(sibl0)
                psiflx = float(polar[1]) * DEG
                acbx = self.guidance_line(vehicle, siblc, psiflx, thtflx)
            if guid_mid == 3:
                acbx = self.guidance_mid_pronav(vehicle, siblc)
        if guid_term == 6:
            acbx = self.guidance_term_comp(vehicle, ctx.int_step)
        if guid_term == 7:
            acbx = self.guidance_term_pronav(vehicle, ctx.int_step)

        allx = float(acbx[1])
        annx = float(-acbx[2])
        aa = sqrt(allx * allx + annx * annx)
        if aa > gmax:
            aa = gmax
        if fabs(annx) < SMALL and fabs(allx) < SMALL:
            phi = 0.0
        else:
            phi = atan2(annx, allx)
        alcomx = aa * cos(phi)
        ancomx = aa * sin(phi)

        store.set("ancomx", ancomx)
        store.set("alcomx", alcomx)
        store.set("allx", allx)
        store.set("annx", annx)
        store.set("psiflx", psiflx)
        store.set("SIBLC", siblc)

    def guidance_line(self, vehicle, siblc, psiflx, thtflx):
        store = vehicle.store
        line_gain = store.get("line_gain")
        nl_gain_fact = store.get("nl_gain_fact")
        decrement = store.get("decrement")
        grav = store.get("grav")
        vbelc = store.get("VBELC")
        tblc = store.get("TBLC")
        thtvlcx = store.get("thtvlcx")
        psivlcx = store.get("psivlcx")

        tfl = mat2tr(psiflx * RAD, thtflx * RAD)
        polar = polar_from_cart(siblc)
        ip_sltrange = float(polar[0])
        psiol = float(polar[1])
        thtol = float(polar[2])
        tol = mat2tr(psiol, thtol)
        vbeo = tol @ vbelc
        vbef = tfl @ vbelc
        nl_gain = nl_gain_fact * (1.0 - exp(-ip_sltrange / decrement))
        algv = np.array(
            [
                grav * sin(thtvlcx * RAD),
                line_gain * (-vbeo[1] + nl_gain * vbef[1]),
                line_gain * (-vbeo[2] + nl_gain * vbef[2])
                - grav * cos(thtvlcx * RAD),
            ],
            dtype=float,
        )
        tvl = mat2tr(psivlcx * RAD, thtvlcx * RAD)
        tbv = tblc @ tvl.T
        acbx = tbv @ algv * (1.0 / AGRAV)

        store.set("ip_sltrange", ip_sltrange)
        store.set("nl_gain", nl_gain)
        store.set("VBEO", vbeo)
        store.set("VBEF", vbef)
        return acbx

    def guidance_mid_pronav(self, vehicle, siblc):
        store = vehicle.store
        gnav = store.get("gnav")
        tblc = np.asarray(store.get("TBLC"), dtype=float)
        vbelc = np.asarray(store.get("VBELC"), dtype=float)
        siblc = np.asarray(siblc, dtype=float)

        dtbc = float(np.linalg.norm(siblc))
        utblc = siblc * (1.0 / dtbc)
        utbbc = tblc @ utblc
        polar = polar_from_cart(utbbc)
        psiobcx = float(polar[1]) * DEG
        thtobcx = float(polar[2]) * DEG
        dvtbc = fabs(float(utblc @ vbelc))
        tgoc = dtbc / dvtbc
        woelc = skew(utblc) @ vbelc * (1.0 / dtbc)
        acbx = tblc @ (skew(woelc) @ utblc) * gnav * dvtbc * (1.0 / AGRAV)

        store.set("WOELC", woelc)
        store.set("UTBLC", utblc)
        store.set("tgoc", tgoc)
        store.set("dtbc", dtbc)
        store.set("dvtbc", dvtbc)
        store.set("psiobcx", psiobcx)
        store.set("thtobcx", thtobcx)
        return np.asarray(acbx, dtype=float).reshape(3)

    def guidance_term_comp(self, vehicle, int_step):
        store = vehicle.store
        grav_bias = store.get("grav_bias")
        gnav = store.get("gnav")
        tgnav = store.get("tgnav")
        sbel = store.get("SBEL")
        vbel = store.get("VBEL")
        stel = store.get("STEL")
        vtel = store.get("VTEL")
        thtpb = store.get("thtpb")
        psipb = store.get("psipb")
        sigdy = store.get("sigdy")
        sigdz = store.get("sigdz")
        tblc = store.get("TBLC")
        fspcb = store.get("FSPCB")
        gnd = store.get("gnd")
        gn = store.get("gn")

        sbtl = sbel - stel
        dbt = float(np.linalg.norm(sbtl))
        dcvel = float(sbtl @ (vbel - vtel)) / dbt
        fspcb1 = float(fspcb[0])
        adely = fspcb1 * tan(psipb) / AGRAV
        adelz = fspcb1 * tan(thtpb) / (cos(psipb) * AGRAV)
        gravb = tblc @ np.array([0.0, 0.0, grav_bias], dtype=float)
        if tgnav:
            gnd_new = (gnav - gn) / tgnav
            gn = float(integrate(gnd_new, gnd, gn, int_step))
            gnd = gnd_new
        else:
            gn = gnav
        gnav_comp = -gn * dcvel
        apnyx = gnav_comp * sigdz / (cos(psipb) * AGRAV)
        apnzx = (
            gnav_comp
            * (sigdz * tan(thtpb) * tan(psipb) + sigdy / cos(thtpb))
            / AGRAV
        )
        allx = apnyx + adely - gravb[1]
        annx = apnzx + adelz + gravb[2]
        acbx = np.array([0.0, allx, -annx], dtype=float)

        store.set("gnd", gnd)
        store.set("gn", gn)
        store.set("gnav_comp", gnav_comp)
        store.set("apnyx", apnyx)
        store.set("apnzx", apnzx)
        store.set("adely", adely)
        store.set("adelz", adelz)
        store.set("dcvel", dcvel)
        return acbx

    def guidance_term_pronav(self, vehicle, int_step):
        store = vehicle.store
        grav_bias = store.get("grav_bias")
        gnav = store.get("gnav")
        tgnav = store.get("tgnav")
        tblc = store.get("TBLC")
        ddab = store.get("ddab")
        psisb = store.get("psisb")
        thtsb = store.get("thtsb")
        lamdqb = store.get("lamdqb")
        lamdrb = store.get("lamdrb")
        gnd = store.get("gnd")
        gn = store.get("gn")

        gravb = tblc @ np.array([0.0, 0.0, grav_bias], dtype=float)
        if tgnav:
            gnd_new = (gnav - gn) / tgnav
            gn = float(integrate(gnd_new, gnd, gn, int_step))
            gnd = gnd_new
        else:
            gn = gnav
        gnav_comp = -gn * ddab
        apnyx = gnav_comp * lamdrb / (cos(psisb) * AGRAV)
        apnzx = (
            gnav_comp
            * (lamdrb * tan(thtsb) * tan(psisb) + lamdqb / cos(thtsb))
            / AGRAV
        )
        allx = apnyx - gravb[1]
        annx = apnzx + gravb[2]
        acbx = np.array([0.0, allx, -annx], dtype=float)

        store.set("gnd", gnd)
        store.set("gn", gn)
        store.set("gnav_comp", gnav_comp)
        store.set("apnyx", apnyx)
        store.set("apnzx", apnzx)
        return acbx

    def terminate(self, vehicle, ctx):
        pass


def _missile_index(ctx):
    combus = ctx.combus
    if not combus:
        return 0
    return sum(
        1
        for i, packet in enumerate(combus)
        if packet.type == "MISSILE6" and i < ctx.vehicle_slot
    )


def _radar_packet(ctx):
    combus = ctx.combus
    if not combus:
        return None
    radar = None
    for packet in combus:
        if packet.type == "RADAR0":
            radar = packet
    return radar


def _ip_siel(ctx):
    radar = _radar_packet(ctx)
    k = _missile_index(ctx)
    if radar is None:
        return np.zeros(3)
    value = radar.vars.get(f"SIEL{k + 1}")
    if value is None:
        return np.zeros(3)
    return np.asarray(value, dtype=float)
