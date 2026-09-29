from math import acos, asin, cos, exp, hypot, sin, sqrt, tan

import numpy as np

from cadac.constants import DEG, EPS, RAD
from cadac.kernel.state import Field
from cadac.math.earth import cadine
from cadac.math.frames import cadtbv, mat2tr, polar_from_cart, skew


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _angle(vec1, vec2):
    scalar = float(vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2])
    abs1 = sqrt(float(vec1[0] ** 2 + vec1[1] ** 2 + vec1[2] ** 2))
    abs2 = sqrt(float(vec2[0] ** 2 + vec2[1] ** 2 + vec2[2] ** 2))
    dum = abs1 * abs2
    if dum > EPS:
        argument = scalar / dum
    else:
        argument = 1.0
    if argument > 1.0:
        argument = 1.0
    if argument < -1.0:
        argument = -1.0
    return acos(argument)


class Hyper5Guidance:
    name = "guidance"

    def define(self, vehicle):
        store = vehicle.store
        plot = ("scrn", "plot")
        for field in (
            Field("mguidance", 0, "int", "data", "guidance", ("scrn",)),
            Field("wp_lonx", 0.0, "real", "data", "guidance"),
            Field("wp_latx", 0.0, "real", "data", "guidance"),
            Field("wp_alt", 0.0, "real", "data", "guidance"),
            Field("point_gain", 0.0, "real", "data", "guidance"),
            Field("pronav_gain", 0.0, "real", "data", "guidance"),
            Field("line_gain", 0.0, "real", "data", "guidance"),
            Field("nl_gain_fact", 1.0, "real", "data", "guidance"),
            Field("decrement", 0.0, "real", "data", "guidance"),
            Field("psifgx", 0.0, "real", "data", "guidance"),
            Field("thtfgx", 0.0, "real", "data", "guidance"),
            Field("bias", 0.0, "real", "data", "guidance"),
            Field("nl_gain", 0.0, "real", "diag", "guidance"),
            Field("VBEF", (0.0, 0.0, 0.0), "vec", "diag", "guidance"),
            Field("wp_sltrange", 999999.0, "real", "diag", "guidance", plot),
            Field("VBEO", (0.0, 0.0, 0.0), "vec", "diag", "guidance"),
            Field("wp_grdrange", 999999.0, "real", "diag", "guidance", plot),
            Field("SWBG", (0.0, 0.0, 0.0), "vec", "out", "guidance"),
            Field("rad_min", 0.0, "real", "diag", "guidance"),
            Field("wp_flag", 0, "int", "diag", "guidance"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mguidance = store.get("mguidance")
        alcomx = 0.0
        ancomx = 0.0
        if mguidance == 0:
            return
        if mguidance == 30:
            grav = store.get("grav")
            phicx = store.get("phicx")
            algv = self.guidance_line(vehicle)
            alcomx = algv[1] / grav
        elif mguidance == 43:
            grav = store.get("grav")
            phicx = store.get("phicx")
            algv = self.guidance_line(vehicle)
            apgv = self.guidance_point(vehicle)
            alcomx = apgv[1] / grav
            ancomx = -algv[2] / grav
        elif mguidance == 40:
            grav = store.get("grav")
            phicx = store.get("phicx")
            apgv = self.guidance_point(vehicle)
            alcomx = apgv[1] / grav
        elif mguidance == 44:
            grav = store.get("grav")
            phicx = store.get("phicx")
            apgv = self.guidance_point(vehicle)
            alcomx = apgv[1] / grav
            ancomx = -apgv[2] / grav
        elif mguidance == 66:
            grav = store.get("grav")
            phicx = store.get("phicx")
            apnb = self.guidance_pronav(vehicle)
            alcomx = apnb[1] / grav
            ancomx = -apnb[2] / grav
        elif mguidance == 3:
            grav = store.get("grav")
            phicx = store.get("phicx")
            algv = self.guidance_line(vehicle)
            alcomx = 0.0
            ancomx = -algv[2] / grav
        elif mguidance == 33:
            grav = store.get("grav")
            phicx = store.get("phicx")
            algv = self.guidance_line(vehicle)
            alcomx = algv[1] / grav
            ancomx = -algv[2] / grav
        elif mguidance == 60:
            grav = store.get("grav")
            phicx = store.get("phicx")
            apnb = self.guidance_pronav(vehicle)
            alcomx = apnb[1] / grav
            ancomx = 0.0
        elif mguidance == 6:
            grav = store.get("grav")
            phicx = store.get("phicx")
            apnb = self.guidance_pronav(vehicle)
            alcomx = 0.0
            ancomx = -apnb[2] / grav
        elif mguidance == 70:
            phicx = self.guidance_arc(vehicle)
        else:
            raise ValueError(f"unknown mguidance {mguidance}")
        anposlimx = store.get("anposlimx")
        anneglimx = store.get("anneglimx")
        allimx = store.get("allimx")
        if ancomx > anposlimx:
            ancomx = anposlimx
        if ancomx < anneglimx:
            ancomx = anneglimx
        if alcomx > allimx:
            alcomx = allimx
        if alcomx < -allimx:
            alcomx = -allimx
        store.set("phicx", phicx)
        store.set("ancomx", ancomx)
        store.set("alcomx", alcomx)

    def guidance_line(self, vehicle):
        store = vehicle.store
        line_gain = store.get("line_gain")
        nl_gain_fact = store.get("nl_gain_fact")
        decrement = store.get("decrement")
        wp_lonx = store.get("wp_lonx")
        wp_latx = store.get("wp_latx")
        wp_alt = store.get("wp_alt")
        psifgx = store.get("psifgx")
        thtfgx = store.get("thtfgx")
        time = store.get("time")
        grav = store.get("grav")
        tig = store.get("tig")
        thtvgx = store.get("thtvgx")
        vbeg = store.get("vbeg")
        sbii = store.get("sbii")
        philimx = store.get("philimx")

        tfg = mat2tr(psifgx * RAD, thtfgx * RAD)
        swii = cadine(wp_lonx * RAD, wp_latx * RAD, wp_alt, time)
        swbg = tig.T @ (swii - sbii)
        polar = polar_from_cart(swbg)
        wp_sltrange = polar[0]
        tog = mat2tr(float(polar[1]), float(polar[2]))
        wp_grdrange = hypot(float(swbg[0]), float(swbg[1]))
        vbeo = tog @ vbeg
        vbef = tfg @ vbeg
        nl_gain = nl_gain_fact * (1 - exp(-wp_sltrange / decrement))
        algv = np.array(
            [
                grav * sin(thtvgx * RAD),
                line_gain * (-vbeo[1] + nl_gain * vbef[1]),
                line_gain * (-vbeo[2] + nl_gain * vbef[2])
                - grav * cos(thtvgx * RAD),
            ]
        )
        dvbe = sqrt(float(vbeg[0] ** 2 + vbeg[1] ** 2 + vbeg[2] ** 2))
        rad_min = dvbe * dvbe / (grav * tan(philimx * RAD))
        if wp_grdrange < 2 * rad_min:
            sh = np.array([swbg[0], swbg[1], 0.0])
            vh = np.array([vbeg[0], vbeg[1], 0.0])
            wp_flag = _sign(float(vh @ sh))
        else:
            wp_flag = 0

        store.set("wp_sltrange", wp_sltrange)
        store.set("nl_gain", nl_gain)
        store.set("VBEO", vbeo)
        store.set("VBEF", vbef)
        store.set("wp_grdrange", wp_grdrange)
        store.set("SWBG", swbg)
        store.set("rad_min", rad_min)
        store.set("wp_flag", wp_flag)
        return algv

    def guidance_point(self, vehicle):
        store = vehicle.store
        wp_lonx = store.get("wp_lonx")
        wp_latx = store.get("wp_latx")
        wp_alt = store.get("wp_alt")
        point_gain = store.get("point_gain")
        time = store.get("time")
        grav = store.get("grav")
        tig = store.get("tig")
        thtvgx = store.get("thtvgx")
        vbeg = store.get("vbeg")
        sbii = store.get("sbii")
        philimx = store.get("philimx")

        swii = cadine(wp_lonx * RAD, wp_latx * RAD, wp_alt, time)
        swbg = tig.T @ (swii - sbii)
        polar = polar_from_cart(swbg)
        wp_sltrange = polar[0]
        psiog = polar[1]
        thtog = polar[2]
        tog = mat2tr(psiog, thtog)
        wp_grdrange = hypot(float(swbg[0]), float(swbg[1]))
        vbeo = tog @ vbeg
        apgv1 = grav * sin(thtvgx * RAD)
        apgv2 = point_gain * (-vbeo[1])
        apgv3 = point_gain * (-vbeo[2]) - grav * cos(thtvgx * RAD)
        apgv = np.array([apgv1, apgv2, apgv3])
        dvbe = sqrt(float(vbeg[0] ** 2 + vbeg[1] ** 2 + vbeg[2] ** 2))
        rad_min = dvbe * dvbe / (grav * tan(philimx * RAD))
        if wp_grdrange < 2 * rad_min:
            sh = np.array([swbg[0], swbg[1], 0.0])
            vh = np.array([vbeg[0], vbeg[1], 0.0])
            if float(vh @ sh) < 0:
                wp_flag = -1
            else:
                wp_flag = 1
        else:
            wp_flag = 0

        store.set("wp_sltrange", wp_sltrange)
        store.set("VBEO", vbeo)
        store.set("wp_grdrange", wp_grdrange)
        store.set("SWBG", swbg)
        store.set("rad_min", rad_min)
        store.set("wp_flag", wp_flag)
        return apgv

    def guidance_pronav(self, vehicle):
        store = vehicle.store
        pronav_gain = store.get("pronav_gain")
        bias = store.get("bias")
        grav = store.get("grav")
        tbg = store.get("TBG")
        woeb = store.get("WOEB")
        closing_speed = store.get("closing_speed")
        utbb = store.get("UTBB")
        grav_g = np.array([0.0, 0.0, grav + bias])
        return skew(woeb) @ utbb * (pronav_gain * closing_speed) - tbg @ grav_g

    def guidance_arc(self, vehicle):
        store = vehicle.store
        wp_lonx = store.get("wp_lonx")
        wp_latx = store.get("wp_latx")
        wp_alt = store.get("wp_alt")
        time = store.get("time")
        fspv = store.get("FSPV")
        grav = store.get("grav")
        tig = store.get("tig")
        dvbe = store.get("dvbe")
        vbeg = store.get("vbeg")
        sbii = store.get("sbii")
        alphax = store.get("alphax")
        phimvx = store.get("phimvx")
        philimx = store.get("philimx")

        swii = cadine(wp_lonx * RAD, wp_latx * RAD, wp_alt, time)
        swbg = tig.T @ (swii - sbii)
        swbg1 = float(swbg[0])
        swbg2 = float(swbg[1])
        sh = np.array([swbg1, swbg2, 0.0])
        dwbh = sqrt(swbg1 * swbg1 + swbg2 * swbg2)
        vbeg1 = float(vbeg[0])
        vbeg2 = float(vbeg[1])
        vh = np.array([vbeg1, vbeg2, 0.0])
        uv = skew(vh) @ sh
        psiwvx = DEG * _angle(vh, sh)
        zz = np.array([0.0, 0.0, 1.0])
        psiwvx = psiwvx * _sign(float(uv[0] * zz[0] + uv[1] * zz[1] + uv[2] * zz[2]))
        alpha = alphax * RAD
        phimv = phimvx * RAD
        tbv = cadtbv(phimv, alpha)
        fspb = tbv @ fspv
        fspb3 = float(fspb[2])
        argument = 0.0
        if abs(psiwvx) < 90:
            num = -2 * dvbe * dvbe * sin(psiwvx * RAD)
            denom = fspb3 * dwbh
            if denom != 0:
                argument = num / denom
            if abs(argument) <= 1.0 and abs(asin(argument)) < philimx * RAD:
                phicx = DEG * asin(argument)
            else:
                phicx = philimx * _sign(argument)
        else:
            phicx = philimx * _sign(psiwvx)
        rad_min = dvbe * dvbe / (grav * tan(philimx * RAD))
        if dwbh < 0.2 * rad_min:
            wp_flag = _sign(float(vh[0] * sh[0] + vh[1] * sh[1] + vh[2] * sh[2]))
        else:
            wp_flag = 0
        wp_grdrange = dwbh

        store.set("SWBG", swbg)
        store.set("wp_grdrange", wp_grdrange)
        store.set("rad_min", rad_min)
        store.set("wp_flag", wp_flag)
        return phicx

    def terminate(self, vehicle, ctx):
        pass
