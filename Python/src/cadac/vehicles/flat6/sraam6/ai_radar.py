"""SRAAM6 AI acquisition radar — Fortran MODULE.FOR S2 (Band D).

Maps Fortran ST1EL/ST2EL/VT1EL onto C++-style STEL/SSEL/VTEL. On MNAV=3,
overwrites STEL/VTEL with biased ST1CEL/VT1CEL so existing guidance can latch.
"""

import numpy as np

from cadac.kernel.state import Field
from cadac.math.frames import cart_from_pol, polar_from_cart

_ZEROS3 = (0.0, 0.0, 0.0)


class Sraam6AiRadar:
    name = "ai_radar"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("ntag", 0, "int", "data", "ai_radar"),
            Field("dtimtu", 0.0, "real", "data", "ai_radar"),
            Field("dtimup", 0.0, "real", "data", "ai_radar"),
            Field("biastd", 0.0, "real", "data", "ai_radar"),
            Field("randtd", 0.0, "real", "data", "ai_radar"),
            Field("biasta", 0.0, "real", "data", "ai_radar"),
            Field("randta", 0.0, "real", "data", "ai_radar"),
            Field("biaste", 0.0, "real", "data", "ai_radar"),
            Field("randte", 0.0, "real", "data", "ai_radar"),
            Field("EVT1EL", _ZEROS3, "vec", "data", "ai_radar"),
            Field("ST1CEL", _ZEROS3, "vec", "out", "ai_radar"),
            Field("VT1CEL", _ZEROS3, "vec", "out", "ai_radar"),
            Field("ai_iset1", 0, "int", "save", "ai_radar"),
            Field("ai_iset2", 0, "int", "save", "ai_radar"),
            Field("epchtai", 0.0, "real", "save", "ai_radar"),
            Field("epchup", 0.0, "real", "save", "ai_radar"),
        ):
            if field.name not in store:
                store.define(field)
        # mnav may already be defined by guidance; define if absent (unit harness)
        if "mnav" not in store:
            store.define(Field("mnav", 0, "int", "out", "ai_radar"))

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mguid = int(store.get("mguid"))
        if mguid == 6:
            return

        # Fortran ST1EL←STEL (target), ST2EL←SSEL (shooter)
        st1el = np.asarray(store.get("STEL"), dtype=float)
        st2el = np.asarray(store.get("SSEL"), dtype=float)
        vt1el = np.asarray(store.get("VTEL"), dtype=float)

        st2t1l = st2el - st1el
        polar = polar_from_cart(st2t1l)
        dt2t1 = float(polar[0])
        azt2t1 = float(polar[1])
        elt2t1 = float(polar[2])

        ntag = int(store.get("ntag"))
        if ntag == 0:
            return

        time = float(ctx.sim_time)
        ai_iset1 = int(store.get("ai_iset1"))
        ai_iset2 = int(store.get("ai_iset2"))
        epchtai = float(store.get("epchtai"))
        epchup = float(store.get("epchup"))
        mnav = int(store.get("mnav"))
        dtimtu = float(store.get("dtimtu"))
        dtimup = float(store.get("dtimup"))

        if ntag == 1:
            ai_iset1 = 0
            epchtai = time

        if time >= epchtai and ai_iset1 == 0:
            mnav = 2
            ai_iset1 = 1
            ai_iset2 = 0
            ntag = ntag + 1
            epchup = time + dtimtu

            dt2t1r = dt2t1 + float(store.get("biastd")) + float(store.get("randtd"))
            azt2tr = azt2t1 + float(store.get("biasta")) + float(store.get("randta"))
            elt2tr = elt2t1 + float(store.get("biaste")) + float(store.get("randte"))
            st2t1l = cart_from_pol(dt2t1r, azt2tr, elt2tr)
            st1cel = st2el - st2t1l
            evt1el = np.asarray(store.get("EVT1EL"), dtype=float)
            vt1cel = vt1el + evt1el
            store.set("ST1CEL", st1cel)
            store.set("VT1CEL", vt1cel)

        if time >= epchup and ai_iset2 == 0:
            mnav = 3
            ai_iset1 = 0
            ai_iset2 = 1
            epchtai = time + dtimup - dtimtu
            # Guidance latches STEL/VTEL on mnav==3 (C++ path); feed AI stores
            store.set("STEL", np.asarray(store.get("ST1CEL"), dtype=float).copy())
            store.set("VTEL", np.asarray(store.get("VT1CEL"), dtype=float).copy())

        store.set("ntag", ntag)
        store.set("mnav", mnav)
        store.set("ai_iset1", ai_iset1)
        store.set("ai_iset2", ai_iset2)
        store.set("epchtai", epchtai)
        store.set("epchup", epchup)

    def terminate(self, vehicle, ctx):
        pass
