from math import acos, cos, sin, sqrt

import numpy as np

from cadac.constants import DEG, RAD, REARTH
from cadac.kernel.state import Field
from cadac.math.frames import polar_from_cart


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


class Hyper5Seeker:
    name = "seeker"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("mseeker", 0, "int", "data/save", "seeker", ("scrn",)),
            Field("acq_range", 0.0, "real", "data", "seeker"),
            Field("range_go", 0.0, "real", "out", "seeker", ("plot", "scrn")),
            Field("STBG", zeros3, "vec", "out", "seeker", ("plot",)),
            Field("WOEB", zeros3, "vec", "out", "seeker"),
            Field("closing_speed", 0.0, "real", "out", "seeker"),
            Field("time_go", 0.0, "real", "out", "seeker", ("plot", "scrn")),
            Field("psisbx", 0.0, "real", "out", "seeker", ("plot", "scrn")),
            Field("thtsbx", 0.0, "real", "out", "seeker", ("plot", "scrn")),
            Field("targ_com_slot", 0, "int", "save", "seeker"),
            Field("UTBB", zeros3, "vec", "out", "seeker"),
            Field("acquisition", 0, "int", "init/save", "seeker", ("scrn",)),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def seeker_grnd_ranges(self, vehicle, combus):
        store = vehicle.store
        lon_c = store.get("lonx") * RAD
        lat_c = store.get("latx") * RAD
        slots = []
        ranges = []
        for i, packet in enumerate(combus):
            if packet.type != "TARGET3":
                continue
            lon_t = packet.vars["lonx"] * RAD
            lat_t = packet.vars["latx"] * RAD
            dum = sin(lat_t) * sin(lat_c) + cos(lat_t) * cos(lat_c) * cos(
                lon_t - lon_c
            )
            slots.append(i)
            ranges.append(REARTH * acos(dum))
        return slots, ranges

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mseeker = store.get("mseeker")
        if mseeker == 0:
            return
        if mseeker not in (1, 3):
            raise ValueError(f"unknown mseeker {mseeker}")

        acq_range = store.get("acq_range")
        acquisition = store.get("acquisition")
        targ_com_slot = store.get("targ_com_slot")
        combus = ctx.combus

        if not acquisition and mseeker == 1:
            slots, ranges = self.seeker_grnd_ranges(vehicle, combus)
            for slot, range_m in zip(slots, ranges):
                if range_m < acq_range:
                    acquisition = 1
                    mseeker = 3
                    targ_com_slot = slot

        if mseeker == 3:
            packet = combus[targ_com_slot]
            tgt_vbeg = packet.vars["vbeg"]
            tgt_sbii = packet.vars["sbii"]
            tig = store.get("tig")
            vbeg = store.get("vbeg")
            sbii = store.get("sbii")
            tbg = store.get("TBG")

            stbi = tgt_sbii - sbii
            tgi = tig.T
            stbg = tgi @ stbi
            range_go = sqrt(float(stbg[0] ** 2 + stbg[1] ** 2 + stbg[2] ** 2))
            inv_dtb = 1.0 / range_go
            utbg = stbg * inv_dtb
            vtbg = tgt_vbeg - vbeg
            woeb = tbg @ _skew(utbg) @ vtbg * inv_dtb
            vbtg = vtbg * (-1.0)
            closing_speed = float(utbg @ vbtg)
            time_go = range_go / closing_speed
            utbb = tbg @ utbg
            polar = polar_from_cart(utbb)
            psisbx = polar[1] * DEG
            thtsbx = polar[2] * DEG

            store.set("range_go", range_go)
            store.set("STBG", stbg)
            store.set("WOEB", woeb)
            store.set("closing_speed", closing_speed)
            store.set("time_go", time_go)
            store.set("psisbx", psisbx)
            store.set("thtsbx", thtsbx)
            store.set("UTBB", utbb)

        store.set("targ_com_slot", targ_com_slot)
        store.set("acquisition", acquisition)
        store.set("mseeker", mseeker)
