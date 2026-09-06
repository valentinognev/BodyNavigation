from math import sqrt

import numpy as np

from cadac.kernel.state import Field


class Cruise5Intercept:
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
        ):
            if field.name not in store.names():
                store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def terminate(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        write = store.get("write")
        time_m = store.get("time_m")
        sbtgm = store.get("SBTGM")
        stmeg = store.get("STMEG")
        sbmeg = store.get("SBMEG")
        time = store.get("time")
        alt = store.get("alt")
        sbeg = store.get("sbeg")
        mguidance = store.get("mguidance")
        wp_alt = store.get("wp_alt")
        swbg = store.get("SWBG")
        wp_flag = store.get("wp_flag")
        mseeker = store.get("mseeker")
        range_go = store.get("range_go")
        stbg = store.get("STBG")
        closing_speed = store.get("closing_speed")
        targ_com_slot = store.get("targ_com_slot")
        hit_time = 0.0
        miss_g = np.zeros(3)
        miss = 0.0
        if (alt <= 0) and write:
            write = 0
            vehicle.health = 0
            ctx.combus[ctx.vehicle_slot].status = 0
        if mguidance == 70 or mguidance == 40 or mguidance == 30:
            if wp_flag == -1:
                swbg1 = float(swbg[0])
                swbg2 = float(swbg[1])
                dwbh = sqrt(swbg1 * swbg1 + swbg2 * swbg2)
        if mguidance == 33 or mguidance == 43:
            if (alt <= wp_alt) and write:
                write = 0
                miss = sqrt(
                    float(swbg[0] ** 2 + swbg[1] ** 2 + swbg[2] ** 2)
                )
                vehicle.health = 0
                ctx.combus[ctx.vehicle_slot].status = 0
        if mseeker == 3:
            if range_go < 100:
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
                    ctx.combus[targ_com_slot].status = -1
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
