from math import cos, hypot, sin, sqrt, tan

import numpy as np

from cadac.constants import RAD
from cadac.kernel.state import Field
from cadac.math.earth import cadine
from cadac.math.frames import mat2tr, polar_from_cart


def _skew(vec):
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


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
            Field("bias", 0.0, "real", "data", "guidance"),
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
        pass

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
        return _skew(woeb) @ utbb * (pronav_gain * closing_speed) - tbg @ grav_g

    def terminate(self, vehicle, ctx):
        pass
