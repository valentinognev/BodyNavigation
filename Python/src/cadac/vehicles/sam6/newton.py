import math

import numpy as np

from cadac.constants import DEG, RAD
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr


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


class Sam6Newton:
    name = "newton"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        scrn_plot_com = ("scrn", "plot", "com")
        scrn_com = ("scrn", "com")
        scrn_plot = ("scrn", "plot")
        for field in (
            Field("VBEBD", zeros3, "vec", "state", "newton"),
            Field("VBEB", zeros3, "vec", "state", "newton"),
            Field("SBELD", zeros3, "vec", "state", "newton"),
            Field("SBEL", zeros3, "vec", "state", "newton", scrn_plot_com),
            Field("sbel1", 0.0, "real", "data", "newton"),
            Field("sbel2", 0.0, "real", "data", "newton"),
            Field("sbel3", 0.0, "real", "data", "newton"),
            Field("FSPB", zeros3, "vec", "out", "newton"),
            Field("VBEL", zeros3, "vec", "out", "newton", scrn_com),
            Field("dvbe", 0.0, "real", "in/out", "newton", scrn_plot),
            Field("alpha0x", 0.0, "real", "data", "newton"),
            Field("beta0x", 0.0, "real", "data", "newton"),
            Field("alt", 0.0, "real", "out", "newton", scrn_plot_com),
            Field("hbe", 0.0, "real", "out", "newton", scrn_plot),
            Field("psivlx", 0.0, "real", "diag", "newton", scrn_plot),
            Field("thtvlx", 0.0, "real", "diag", "newton", scrn_plot),
            Field("anx", 0.0, "real", "diag", "newton", scrn_plot),
            Field("ayx", 0.0, "real", "diag", "newton", scrn_plot),
            Field("ATB", zeros3, "vec", "diag", "newton"),
            Field("mfreeze_newt", 0, "int", "save", "newton"),
            Field("dvbef", 0.0, "real", "save", "newton"),
            Field("SLEL", zeros3, "vec", "out", "newton"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        sbel1 = store.get("sbel1")
        sbel2 = store.get("sbel2")
        sbel3 = store.get("sbel3")
        dvbe = store.get("dvbe")
        alpha0x = store.get("alpha0x")
        beta0x = store.get("beta0x")
        tbl = store.get("TBL")
        salp = math.sin(alpha0x * RAD)
        calp = math.cos(alpha0x * RAD)
        sbet = math.sin(beta0x * RAD)
        cbet = math.cos(beta0x * RAD)
        vbeb = np.array(
            [calp * cbet * dvbe, sbet * dvbe, salp * cbet * dvbe], dtype=float
        )
        vbel = tbl.T @ vbeb
        sbel = np.array([sbel1, sbel2, sbel3], dtype=float)
        alt = -float(sbel[2])
        store.set("VBEB", vbeb)
        store.set("SBEL", sbel)
        store.set("VBEL", vbel)
        store.set("SLEL", np.array(sbel, copy=True))
        store.set("alt", alt)
        store.set("hbe", alt)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mfreeze_newt = store.get("mfreeze_newt")
        dvbef = store.get("dvbef")
        grav = store.get("grav")
        tbl = store.get("TBL")
        fapb = store.get("FAPB")
        wbeb = store.get("WBEB")
        mass = store.get("mass")
        vbebd = store.get("VBEBD")
        vbeb = store.get("VBEB")
        sbeld = store.get("SBELD")
        sbel = store.get("SBEL")
        int_step = ctx.int_step

        atb = _skew(wbeb) @ vbeb
        gravl = np.array([0.0, 0.0, grav], dtype=float)
        fspb = fapb * (1.0 / mass)
        vbebd_new = fspb - atb + tbl @ gravl
        vbeb = integrate(vbebd_new, vbebd, vbeb, int_step)
        vbebd = vbebd_new
        vbel = tbl.T @ vbeb
        sbeld_new = vbel
        sbel = integrate(sbeld_new, sbeld, sbel, int_step)
        sbeld = sbeld_new

        vbel1 = float(vbel[0])
        vbel2 = float(vbel[1])
        vbel3 = float(vbel[2])
        if vbel1 == 0.0 and vbel2 == 0.0:
            psivl = 0.0
        else:
            psivl = math.atan2(vbel2, vbel1)
        thtvl = math.atan2(-vbel3, math.sqrt(vbel1 * vbel1 + vbel2 * vbel2))
        psivlx = psivl * DEG
        thtvlx = thtvl * DEG

        dvbe = float(np.linalg.norm(vbel))
        alt = -float(sbel[2])
        hbe = alt
        anx = -float(fspb[2]) / grav
        ayx = float(fspb[1]) / grav

        if "mfreeze" in store:
            mfreeze = store.get("mfreeze")
            if mfreeze == 0:
                mfreeze_newt = 0
            else:
                if mfreeze != mfreeze_newt:
                    mfreeze_newt = mfreeze
                    dvbef = dvbe
                dvbe = dvbef

        store.set("VBEBD", vbebd)
        store.set("VBEB", vbeb)
        store.set("SBELD", sbeld)
        store.set("SBEL", sbel)
        store.set("mfreeze_newt", mfreeze_newt)
        store.set("dvbef", dvbef)
        store.set("FSPB", fspb)
        store.set("VBEL", vbel)
        store.set("dvbe", dvbe)
        store.set("alt", alt)
        store.set("hbe", hbe)
        store.set("psivlx", psivlx)
        store.set("thtvlx", thtvlx)
        store.set("anx", anx)
        store.set("ayx", ayx)
        store.set("ATB", atb)

    def terminate(self, vehicle, ctx):
        pass
