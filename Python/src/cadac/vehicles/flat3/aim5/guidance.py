from math import atan2, cos, exp, sin, sqrt

import numpy as np

from cadac.kernel.state import Field
from cadac.math.frames import skew

SMALL = 1e-7


class Aim5Guidance:
    name = "guidance"

    def define(self, vehicle):
        store = vehicle.store
        plot = ("scrn", "plot")
        for field in (
            Field("mguid", 0, "int", "data", "guidance"),
            Field("gnav", 0.0, "real", "data", "guidance"),
            Field("tgo_manvr", 0.0, "real", "data", "guidance"),
            Field("amp_manvr", 0.0, "real", "data", "guidance"),
            Field("frq_manvr", 0.0, "real", "data", "guidance"),
            Field("tgo63_manvr", 0.0, "real", "data", "guidance"),
            Field("annx", 0.0, "real", "diag", "guidance"),
            Field("allx", 0.0, "real", "diag", "guidance"),
            Field("an_manvr", 0.0, "real", "diag", "guidance"),
            Field("al_manvr", 0.0, "real", "diag", "guidance"),
            Field("ancomx", 0.0, "real", "out", "guidance", plot),
            Field("alcomx", 0.0, "real", "out", "guidance", plot),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mguid = store.get("mguid")
        gnav = store.get("gnav")
        tgo_manvr = store.get("tgo_manvr")
        amp_manvr = store.get("amp_manvr")
        frq_manvr = store.get("frq_manvr")
        tgo63_manvr = store.get("tgo63_manvr")
        grav = store.get("grav")
        gmax = store.get("gmax")
        dvta = store.get("dvta")
        tgo_aim = store.get("tgo_aim")
        utaa = store.get("UTAA")
        woea = store.get("WOEA")

        guid_manvr = mguid // 10
        guid_mode = mguid % 10
        if guid_mode not in {0, 1} or guid_manvr not in {0, 1}:
            raise ValueError(
                f"unsupported mguid={mguid}: guid_mode={guid_mode} guid_manvr={guid_manvr}"
            )

        annx = 0.0
        allx = 0.0
        an_manvr = 0.0
        al_manvr = 0.0
        if guid_mode == 1:
            apna = skew(woea) @ utaa * (gnav * abs(dvta))
            annx = -apna[2] / grav
            allx = apna[1] / grav
        if guid_manvr == 1 and tgo_aim < tgo_manvr:
            amp = amp_manvr * (1.0 - exp(-tgo_aim / tgo63_manvr))
            an_manvr = amp * sin(frq_manvr * tgo_aim)
            al_manvr = amp * cos(frq_manvr * tgo_aim)
            annx += an_manvr
            allx += al_manvr

        aax = sqrt(allx**2 + annx**2)
        if aax > gmax:
            aax = gmax
        if abs(annx) < SMALL or abs(allx) < SMALL:
            phi = 0.0
        else:
            phi = atan2(annx, allx)
        alcomx = aax * cos(phi)
        ancomx = aax * sin(phi)

        store.set("annx", annx)
        store.set("allx", allx)
        store.set("an_manvr", an_manvr)
        store.set("al_manvr", al_manvr)
        store.set("ancomx", ancomx)
        store.set("alcomx", alcomx)

    def terminate(self, vehicle, ctx):
        pass
