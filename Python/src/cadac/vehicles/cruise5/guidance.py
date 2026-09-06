from math import cos, exp, hypot, sin, sqrt, tan

import numpy as np

from cadac.constants import RAD
from cadac.kernel.state import Field
from cadac.math.earth import cadine
from cadac.math.frames import mat2tr, polar_from_cart


def _sign(variable):
    if variable < 0:
        return -1
    return 1


class Cruise5Guidance:
    name = "guidance"

    def define(self, vehicle):
        store = vehicle.store
        plot = ("scrn", "plot")
        for field in (
            Field("mguidance", 0, "int", "data", "guidance", ("scrn",)),
            Field("pronav_gain", 0.0, "real", "data", "guidance"),
            Field("line_gain", 0.0, "real", "data", "guidance"),
            Field("nl_gain_fact", 1.0, "real", "data", "guidance"),
            Field("decrement", 0.0, "real", "data", "guidance"),
            Field("wp_lonx", 0.0, "real", "data", "guidance"),
            Field("wp_latx", 0.0, "real", "data", "guidance"),
            Field("wp_alt", 0.0, "real", "data", "guidance"),
            Field("psifgx", 0.0, "real", "data", "guidance"),
            Field("thtfgx", 0.0, "real", "data", "guidance"),
            Field("point_gain", 0.0, "real", "data", "guidance"),
            Field("wp_sltrange", 999999.0, "real", "diag", "guidance"),
            Field("nl_gain", 0.0, "real", "diag", "guidance"),
            Field("VBEO", (0.0, 0.0, 0.0), "vec", "diag", "guidance"),
            Field("VBEF", (0.0, 0.0, 0.0), "vec", "diag", "guidance"),
            Field("wp_grdrange", 999999.0, "real", "diag", "guidance", plot),
            Field("SWBG", (0.0, 0.0, 0.0), "vec", "out", "guidance"),
            Field("rad_min", 0.0, "real", "diag", "guidance"),
            Field("rad_geometric", 0.0, "real", "diag", "guidance"),
            Field("wp_flag", 0, "int", "diag", "guidance", ("plot",)),
        ):
            if field.name not in store.names():
                store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        pass

    def terminate(self, vehicle, ctx):
        pass

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
        wp_sltrange = float(polar[0])
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
        dvbe = sqrt(float(vbeg @ vbeg))
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
