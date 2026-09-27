from math import sqrt

import numpy as np

from cadac.constants import DEG, RAD
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr, polar_from_cart

_ZEROS3 = (0.0, 0.0, 0.0)


class Sraam6Intercept:
    name = "intercept"

    def define(self, vehicle):
        store = vehicle.store
        plot = ("plot",)
        scrn = ("scrn",)
        for field in (
            Field("mterm", 0, "int", "data", "intercept"),
            Field("write", 1, "int", "init", "intercept"),
            Field("miss", 0.0, "real", "diag", "intercept", plot),
            Field("hit_time", 0.0, "real", "diag", "intercept"),
            Field("MISS", _ZEROS3, "vec", "diag", "intercept", plot),
            Field("time_m", 0.0, "real", "save", "intercept"),
            Field("SBTLM", _ZEROS3, "vec", "save", "intercept"),
            Field("STMEL", _ZEROS3, "vec", "save", "intercept"),
            Field("SBMEL", _ZEROS3, "vec", "save", "intercept"),
            Field("mode", 0, "int", "diag", "intercept", scrn),
            Field("psiptx", 0.0, "real", "diag/data", "intercept", plot),
            Field("thtptx", 0.0, "real", "diag/data", "intercept", plot),
            Field("critmax", 200.0, "real", "diag/data", "intercept", plot),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def terminate(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mterm = store.get("mterm")
        if mterm not in (0, 1, 2):
            raise ValueError(f"unknown mterm {mterm}")
        psiptx = store.get("psiptx")
        thtptx = store.get("thtptx")
        time = store.get("time")
        halt = store.get("halt")
        stop = store.get("stop")
        lconv = store.get("lconv")
        sbel = np.asarray(store.get("SBEL"), dtype=float)
        vbel = np.asarray(store.get("VBEL"), dtype=float)
        stel = np.asarray(store.get("STEL"), dtype=float)
        vtel = np.asarray(store.get("VTEL"), dtype=float)
        tgt_com_slot = store.get("tgt_com_slot")
        mprop = store.get("mprop")
        trcond = store.get("trcond")
        mseek = store.get("mseek")
        mguid = store.get("mguid")
        maut = store.get("maut")
        write = store.get("write")
        time_m = store.get("time_m")
        sbtlm = np.asarray(store.get("SBTLM"), dtype=float)
        stmel = np.asarray(store.get("STMEL"), dtype=float)
        sbmel = np.asarray(store.get("SBMEL"), dtype=float)

        hit_time = 0.0
        miss_vec = np.zeros(3)
        miss = 0.0

        guid_mid = mguid // 10
        guid_term = mguid % 10
        mode = 10000 * mseek + 1000 * guid_mid + 100 * guid_term + 10 * maut + mprop

        stbl = stel - sbel
        dbt = float(np.sqrt(float(stbl @ stbl)))

        if trcond and stop:
            lconv = 4
            vehicle.health = 0
            ctx.combus[ctx.vehicle_slot].status = 0

        if halt:
            lconv = 5
            vehicle.health = 0
            ctx.combus[ctx.vehicle_slot].status = 0

        alt = -float(sbel[2])
        if (alt <= 0.0) and write:
            write = 0
            lconv = 3
            vehicle.health = 0
            ctx.combus[ctx.vehicle_slot].status = 0

        if mseek >= 3:
            if dbt < 100.0:
                utbl = stbl * (1.0 / dbt)
                vtbel = vtel - vbel
                closing_speed = float(utbl @ vtbel)
                sbtl = stbl * (-1.0)
                int_step = ctx.int_step
                if (closing_speed > 0.0) and write:
                    write = 0
                    sbbml = sbel - sbmel
                    sttml = stel - stmel
                    hit_time = time_m - int_step * float(
                        (sbbml - sttml) @ sbtlm
                    ) / float(sbbml @ sbbml)
                    if mterm == 0:
                        lconv = 2
                        tau = hit_time - time_m
                        miss_vec = (sbbml - sttml) * (tau / int_step) + sbtlm
                        miss = sqrt(
                            float(
                                miss_vec[0] ** 2
                                + miss_vec[1] ** 2
                                + miss_vec[2] ** 2
                            )
                        )
                    if mterm > 0:
                        lconv = 2
                        polar = polar_from_cart(vtel)
                        ttl = mat2tr(float(polar[1]), float(polar[2]))
                        vtbet = ttl @ vtbel
                        if mterm == 1:
                            vbtet = vtbet * (-1.0)
                            polar = polar_from_cart(vbtet)
                            psiptx = float(polar[1]) * DEG
                            thtptx = float(polar[2]) * DEG - 90.0
                        tpt = mat2tr(psiptx * RAD, thtptx * RAD)
                        tpl = tpt @ ttl
                        sbtp = tpl @ sbtl
                        sbbmp = tpl @ sbbml
                        stbmp = sbbmp - sbtp
                        ww = float(stbmp[2] / sbbmp[2])
                        miss_vec = sbbmp * ww - stbmp
                        miss = sqrt(
                            float(
                                miss_vec[0] ** 2
                                + miss_vec[1] ** 2
                                + miss_vec[2] ** 2
                            )
                        )
                    vehicle.health = 0
                    ctx.combus[ctx.vehicle_slot].status = 0
                    ctx.combus[tgt_com_slot].status = 0
                sbtlm = np.array(sbtl, dtype=float, copy=True)
                stmel = np.array(stel, dtype=float, copy=True)
                sbmel = np.array(sbel, dtype=float, copy=True)
                time_m = time

        store.set("write", write)
        store.set("hit_time", hit_time)
        store.set("time_m", time_m)
        store.set("SBTLM", sbtlm)
        store.set("STMEL", stmel)
        store.set("SBMEL", sbmel)
        store.set("lconv", lconv)
        store.set("miss", miss)
        store.set("MISS", miss_vec)
        store.set("mode", mode)
        store.set("psiptx", psiptx)
        store.set("thtptx", thtptx)
