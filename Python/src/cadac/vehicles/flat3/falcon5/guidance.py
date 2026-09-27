from math import cos, exp, hypot, sin, sqrt, tan

import numpy as np

from cadac.constants import RAD
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr, polar_from_cart


class Plane5Guidance:
    name = "guidance"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("mguidance", 0, "int", "data", "guidance", ("scrn",)),
            Field("swel1", 0.0, "real", "data", "guidance"),
            Field("swel2", 0.0, "real", "data", "guidance"),
            Field("swel3", 0.0, "real", "data", "guidance"),
            Field("line_gain", 0.0, "real", "data", "guidance"),
            Field("nl_gain_fact", 1.0, "real", "data", "guidance"),
            Field("decrement", 0.0, "real", "data", "guidance"),
            Field("psiflx", 0.0, "real", "data", "guidance"),
            Field("thtflx", 0.0, "real", "data", "guidance"),
            Field("point_gain", 0.0, "real", "data", "guidance"),
            Field("wp_sltrange", 999999.0, "real", "dia", "guidance"),
            Field("nl_gain", 0.0, "real", "dia", "guidance"),
            Field("VBEO", (0.0, 0.0, 0.0), "vec", "dia", "guidance"),
            Field("VBEF", (0.0, 0.0, 0.0), "vec", "dia", "guidance"),
            Field("wp_grdrange", 999999.0, "real", "dia", "guidance", ("scrn", "plot")),
            Field("SWBL", (0.0, 0.0, 0.0), "vec", "out", "guidance"),
            Field("rad_min", 0.0, "real", "dia", "guidance"),
            Field("write", 0, "int", "save", "guidance", ("scrn", "plot")),
            Field("wp_flag", 0, "int", "dia", "guidance", ("plot", "scrn")),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mguidance = store.get("mguidance")
        grav = store.get("grav")
        phicx = store.get("phicx")
        anposlimx = store.get("anposlimx")
        anneglimx = store.get("anneglimx")
        allimx = store.get("allimx")
        if mguidance == 0:
            return
        if mguidance == 30:
            algv = self.guidance_line(vehicle)
            alcomx = algv[1] / grav
            ancomx = 0.0
        elif mguidance == 33:
            algv = self.guidance_line(vehicle)
            alcomx = algv[1] / grav
            ancomx = -algv[2] / grav
        elif mguidance == 40:
            apgv = self.guidance_point(vehicle)
            alcomx = apgv[1] / grav
            ancomx = 0.0
        else:
            raise ValueError(f"unknown mguidance {mguidance}")
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

    def guidance_point(self, vehicle):
        store = vehicle.store
        swel1 = store.get("swel1")
        swel2 = store.get("swel2")
        swel3 = store.get("swel3")
        point_gain = store.get("point_gain")
        write = store.get("write")
        grav = store.get("grav")
        thtvlx = store.get("thtvlx")
        vbel = store.get("VBEL")
        sbel = store.get("SBEL")
        philimx = store.get("philimx")

        swel = np.array([swel1, swel2, swel3])
        swbl = swel - sbel
        polar = polar_from_cart(swbl)
        wp_sltrange = polar[0]
        psiol = polar[1]
        thtol = polar[2]
        tol = mat2tr(psiol, thtol)
        wp_grdrange = hypot(swbl[0], swbl[1])
        vbeo = tol @ vbel
        apgv1 = grav * sin(thtvlx * RAD)
        apgv2 = point_gain * (-vbeo[1])
        apgv3 = point_gain * (-vbeo[2]) - grav * cos(thtvlx * RAD)
        apgv = np.array([apgv1, apgv2, apgv3])
        dvbe = sqrt(float(vbel[0] ** 2 + vbel[1] ** 2 + vbel[2] ** 2))
        rad_min = dvbe * dvbe / (grav * tan(philimx * RAD))
        if wp_grdrange < 2 * rad_min:
            sh = np.array([swbl[0], swbl[1], 0.0])
            vh = np.array([vbel[0], vbel[1], 0.0])
            if float(vh @ sh) < 0:
                wp_flag = -1
            else:
                wp_flag = 1
            if wp_flag == 1:
                write = 1
        else:
            wp_flag = 0

        store.set("write", write)
        store.set("wp_sltrange", wp_sltrange)
        store.set("VBEO", vbeo)
        store.set("wp_grdrange", wp_grdrange)
        store.set("SWBL", swbl)
        store.set("rad_min", rad_min)
        store.set("wp_flag", wp_flag)
        return apgv

    def guidance_line(self, vehicle):
        store = vehicle.store
        line_gain = store.get("line_gain")
        nl_gain_fact = store.get("nl_gain_fact")
        decrement = store.get("decrement")
        swel1 = store.get("swel1")
        swel2 = store.get("swel2")
        swel3 = store.get("swel3")
        psiflx = store.get("psiflx")
        thtflx = store.get("thtflx")
        write = store.get("write")
        grav = store.get("grav")
        thtvlx = store.get("thtvlx")
        vbel = store.get("VBEL")
        sbel = store.get("SBEL")
        philimx = store.get("philimx")

        tfl = mat2tr(psiflx * RAD, thtflx * RAD)
        swel = np.array([swel1, swel2, swel3])
        swbl = swel - sbel
        polar = polar_from_cart(swbl)
        wp_sltrange = polar[0]
        psiol = polar[1]
        thtol = polar[2]
        tol = mat2tr(psiol, thtol)
        wp_grdrange = hypot(swbl[0], swbl[1])
        vbeo = tol @ vbel
        vbef = tfl @ vbel
        nl_gain = nl_gain_fact * (1 - exp(-wp_sltrange / decrement))
        algv1 = grav * sin(thtvlx * RAD)
        algv2 = line_gain * (-vbeo[1] + nl_gain * vbef[1])
        algv3 = line_gain * (-vbeo[2] + nl_gain * vbef[2]) - grav * cos(thtvlx * RAD)
        algv = np.array([algv1, algv2, algv3])
        dvbe = sqrt(float(vbel[0] ** 2 + vbel[1] ** 2 + vbel[2] ** 2))
        rad_min = dvbe * dvbe / (grav * tan(philimx * RAD))
        if wp_grdrange < 2 * rad_min:
            sh = np.array([swbl[0], swbl[1], 0.0])
            vh = np.array([vbel[0], vbel[1], 0.0])
            if float(vh @ sh) < 0:
                wp_flag = -1
            else:
                wp_flag = 1
            if wp_flag == 1:
                write = 1
        else:
            wp_flag = 0

        store.set("write", write)
        store.set("wp_sltrange", wp_sltrange)
        store.set("nl_gain", nl_gain)
        store.set("VBEO", vbeo)
        store.set("VBEF", vbef)
        store.set("wp_grdrange", wp_grdrange)
        store.set("SWBL", swbl)
        store.set("rad_min", rad_min)
        store.set("wp_flag", wp_flag)
        return algv

    def terminate(self, vehicle, ctx):
        pass
