from math import sqrt

import numpy as np

from cadac.constants import RAD
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr


def _optional(store, name, default=0):
    if name in store.names():
        return store.get(name)
    return default


def _vec3(vars_, primary, fallback):
    if primary in vars_:
        return np.array(vars_[primary], dtype=float, copy=True)
    if fallback in vars_:
        return np.array(vars_[fallback], dtype=float, copy=True)
    return np.zeros(3)


class Agm6Intercept:
    name = "intercept"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("plot",)
        scrn = ("scrn",)
        scrn_plot = ("scrn", "plot")
        for field in (
            Field("mterm", 0, "int", "data", "intercept"),
            Field("write", 1, "int", "init", "intercept"),
            Field("miss", 0.0, "real", "diag", "intercept", plot),
            Field("hit_time", 0.0, "real", "diag", "intercept"),
            Field("MISS_P", zeros3, "vec", "diag", "intercept", plot),
            Field("time_m", 0.0, "real", "save", "intercept"),
            Field("SBMTP", zeros3, "vec", "save", "intercept"),
            Field("mode", 0, "int", "diag", "intercept", scrn),
            Field("dbt", 0.0, "real", "diag", "intercept", scrn_plot),
            Field("psiplx", 0.0, "real", "/diag/data", "intercept", plot),
            Field("thtplx", 0.0, "real", "diag/data", "intercept", plot),
            Field("critmax", 100.0, "real", "diag/data", "intercept", plot),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        psiplx = store.get("psiplx")
        thtplx = store.get("thtplx")
        time = _optional(store, "time", ctx.sim_time)
        halt = _optional(store, "halt", 0)
        stop = _optional(store, "stop", 0)
        sbel = np.asarray(store.get("SBEL"), dtype=float)
        tgt_num = _optional(store, "tgt_num", 0)
        mprop = _optional(store, "mprop", 0)
        trcond = _optional(store, "trcond", 0)
        mseek = _optional(store, "mseek", 0)
        mguid = _optional(store, "mguid", 0)
        maut = _optional(store, "maut", 0)
        write = store.get("write")
        time_m = store.get("time_m")
        sbmtp = np.array(store.get("SBMTP"), dtype=float, copy=True)

        guid_mid = mguid // 10
        guid_term = mguid % 10
        mode = 10000 * mseek + 1000 * guid_mid + 100 * guid_term + 10 * maut + mprop

        stel = np.zeros(3)
        count = 0
        for packet in ctx.combus or ():
            if packet is None or packet.type != "TARGET3":
                continue
            count += 1
            if count == tgt_num:
                stel = _vec3(packet.vars, "SAEL", "SBEL")
                break

        stbl = stel - sbel
        dbt = sqrt(float(stbl[0] ** 2 + stbl[1] ** 2 + stbl[2] ** 2))
        hit_time = 0.0
        miss_p = np.zeros(3)
        miss = 0.0
        slot = ctx.vehicle_slot

        if trcond and stop:
            vehicle.health = 0
            ctx.combus[slot].status = 0
        if halt:
            vehicle.health = 0
            ctx.combus[slot].status = 0
        alt = -sbel[2]
        if (alt <= 0) and write:
            write = 0
            vehicle.health = 0
            ctx.combus[slot].status = 0

        if guid_mid == 4 or guid_term == 5 or guid_term == 6:
            sbtl = sbel - stel
            dbt = sqrt(float(sbtl[0] ** 2 + sbtl[1] ** 2 + sbtl[2] ** 2))
            if dbt < 100:
                tpl = mat2tr(psiplx * RAD, thtplx * RAD)
                sbtp = tpl @ sbtl
                sbtp3 = sbtp[2]
                if sbtp3 > 0:
                    write = 0
                    sbbmp = sbtp - sbmtp
                    stbmp = sbmtp * (-1.0)
                    dum = stbmp[2] / sbbmp[2]
                    miss_p = sbbmp * dum - stbmp
                    miss = sqrt(
                        float(miss_p[0] ** 2 + miss_p[1] ** 2 + miss_p[2] ** 2)
                    )
                    hit_time = dum * ctx.int_step + time_m
                    vehicle.health = 0
                    ctx.combus[slot].status = 0
                time_m = time
                sbmtp = sbtp

        store.set("write", write)
        store.set("hit_time", hit_time)
        store.set("time_m", time_m)
        store.set("SBMTP", sbmtp)
        store.set("miss", miss)
        store.set("MISS_P", miss_p)
        store.set("mode", mode)
        store.set("dbt", dbt)

    def terminate(self, vehicle, ctx):
        pass
