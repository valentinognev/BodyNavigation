from math import sqrt

import numpy as np

from cadac.kernel.state import Field


class Hyper5Intercept:
    name = "intercept"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("write", 1, "int", "save", "intercept"),
            Field("miss", 0.0, "real", "diag", "intercept"),
            Field("hit_time", 0.0, "real", "diag", "intercept"),
            Field("MISS_G", zeros3, "vec", "diag", "intercept"),
            Field("time_m", 0.0, "real", "save", "intercept"),
            Field("SBTGM", zeros3, "vec", "save", "intercept"),
            Field("STMEG", zeros3, "vec", "save", "intercept"),
            Field("SBMEG", zeros3, "vec", "save", "intercept"),
            Field("halt", 0, "int", "data", "intercept"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def terminate(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        halt = store.get("halt")
        write = store.get("write")
        time_m = store.get("time_m")
        sbtgm = store.get("SBTGM")
        stmeg = store.get("STMEG")
        sbmeg = store.get("SBMEG")
        time = store.get("time")
        alt = store.get("alt")
        sbeg = store.get("sbeg")
        mseeker = store.get("mseeker")
        range_go = store.get("range_go")
        stbg = store.get("STBG")
        closing_speed = store.get("closing_speed")
        targ_com_slot = store.get("targ_com_slot")
        hit_time = 0.0
        miss_g = np.zeros(3)
        miss = 0.0
        if halt and write:
            write = 0
            vehicle.health = 0
            ctx.combus[ctx.vehicle_slot].status = 0
        if (alt <= 0) and write:
            write = 0
            vehicle.health = 0
            ctx.combus[ctx.vehicle_slot].status = 0
        if mseeker == 3:
            if range_go < 1000:
                steg = ctx.combus[targ_com_slot].vars["sbeg"]
                sbtg = stbg * (-1.0)
                if (closing_speed < 0) and write:
                    write = 0
                    sbbmg = sbeg - sbmeg
                    sttmg = steg - stmeg
                    int_step = ctx.int_step
                    hit_time = time_m - int_step * float(
                        (sbbmg - sttmg) @ sbtgm
                    ) / float(sbbmg @ sbbmg)
                    tau = hit_time - time_m
                    miss_g = (sbbmg - sttmg) * (tau / int_step) + sbtgm
                    miss = sqrt(
                        float(miss_g[0] ** 2 + miss_g[1] ** 2 + miss_g[2] ** 2)
                    )
                    vehicle.health = 0
                    ctx.combus[ctx.vehicle_slot].status = 0
                sbtgm = sbtg
                stmeg = steg
                sbmeg = sbeg
                time_m = time
        store.set("write", write)
        store.set("time_m", time_m)
        store.set("SBTGM", sbtgm)
        store.set("STMEG", stmeg)
        store.set("SBMEG", sbmeg)
        store.set("miss", miss)
        store.set("hit_time", hit_time)
        store.set("MISS_G", miss_g)
