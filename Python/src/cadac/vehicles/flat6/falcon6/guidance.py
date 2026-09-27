from math import cos, exp, sin, sqrt, tan

import numpy as np

from cadac.constants import RAD
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr, polar_from_cart


def _sign(variable):
    if variable < 0.0:
        return -1
    return 1


class Plane6Guidance:
    name = "guidance"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("mguid", 0, "int", "data", "guidance", ("scrn",)),
            Field("line_gain", 0.0, "real", "data", "guidance"),
            Field("nl_gain_fact", 0.0, "real", "data", "guidance"),
            Field("decrement", 0.0, "real", "data", "guidance"),
            Field("swel1", 0.0, "real", "data", "guidance"),
            Field("swel2", 0.0, "real", "data", "guidance"),
            Field("swel3", 0.0, "real", "data", "guidance"),
            Field("psiflx", 0.0, "real", "data", "guidance"),
            Field("thtflx", 0.0, "real", "data", "guidance"),
            Field("dwb", 0.0, "real", "diag", "guidance"),
            Field("nl_gain", 0.0, "real", "diag", "guidance", ("scrn", "plot")),
            Field("VBEO", zeros3, "vec", "diag", "guidance"),
            Field("VBEF", zeros3, "vec", "diag", "guidance"),
            Field("dwbh", 0.0, "real", "diag", "guidance", ("scrn", "plot")),
            Field("SWBL", zeros3, "vec", "diag", "guidance"),
            Field("turn_min", 0.0, "real", "diag", "guidance"),
            Field("wp_flag", 0, "int", "diag", "guidance", ("plot",)),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mguid = store.get("mguid")
        if mguid == 0:
            return
        if mguid not in (30, 33):
            raise ValueError(f"unknown mguid {mguid}")

        swel = np.array(
            [store.get("swel1"), store.get("swel2"), store.get("swel3")],
            dtype=float,
        )
        sbel = np.asarray(store.get("SBEL"), dtype=float)
        vbel = np.asarray(store.get("VBEL"), dtype=float)
        grav = store.get("grav")
        dvbe = store.get("dvbe")
        philimx = store.get("philimx")
        swbl = swel - sbel
        dwb = sqrt(float(swbl @ swbl))
        dwbh = sqrt(swbl[0] * swbl[0] + swbl[1] * swbl[1])
        turn_min = dvbe * dvbe / (grav * tan(philimx * RAD))
        wp_flag = 1
        if dwbh < (2.0 * turn_min):
            closing = swbl[0] * vbel[0] + swbl[1] * vbel[1]
            wp_flag = _sign(closing)

        algv = self.guidance_line(vehicle, swbl)
        mguidl = mguid // 10
        mguidp = mguid % 10
        alcomx = 0.0
        ancomx = 0.0
        if mguidl == 3:
            alcomx = float(algv[1]) / grav
        if mguidp == 3:
            ancomx = -float(algv[2]) / grav

        store.set("dwb", dwb)
        store.set("dwbh", dwbh)
        store.set("SWBL", swbl)
        store.set("turn_min", turn_min)
        store.set("wp_flag", int(wp_flag))
        store.set("alcomx", alcomx)
        store.set("ancomx", ancomx)

    def guidance_line(self, vehicle, swbl):
        store = vehicle.store
        line_gain = store.get("line_gain")
        nl_gain_fact = store.get("nl_gain_fact")
        decrement = store.get("decrement")
        grav = store.get("grav")
        vbel = np.asarray(store.get("VBEL"), dtype=float)
        thtvlx = store.get("thtvlx")
        tfl = mat2tr(store.get("psiflx") * RAD, store.get("thtflx") * RAD)
        polar = polar_from_cart(swbl)
        wp_sltrange = float(polar[0])
        tol = mat2tr(float(polar[1]), float(polar[2]))
        vbeo = tol @ vbel
        vbef = tfl @ vbel
        nl_gain = nl_gain_fact * (1.0 - exp(-wp_sltrange / decrement))
        algv1 = grav * sin(thtvlx * RAD)
        algv2 = line_gain * (-vbeo[1] + nl_gain * vbef[1])
        algv3 = line_gain * (-vbeo[2] + nl_gain * vbef[2]) - grav * cos(thtvlx * RAD)
        store.set("dwb", wp_sltrange)
        store.set("nl_gain", nl_gain)
        store.set("VBEO", np.asarray(vbeo, dtype=float))
        store.set("VBEF", np.asarray(vbef, dtype=float))
        return np.array([algv1, algv2, algv3], dtype=float)

    def terminate(self, vehicle, ctx):
        pass
