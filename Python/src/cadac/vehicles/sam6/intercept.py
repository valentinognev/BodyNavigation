from math import sqrt

import numpy as np

from cadac.constants import DEG, RAD
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr, polar_from_cart


class Sam6Intercept:
    name = "intercept"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("plot",)
        scrn_plot = ("scrn", "plot")
        for field in (
            Field("mterm", 0, "int", "data", "intercept"),
            Field("write", 1, "int", "init", "intercept"),
            Field("miss", 0.0, "real", "diag", "intercept", plot),
            Field("hit_time", 0.0, "real", "diag", "intercept"),
            Field("MISS", zeros3, "vec", "diag", "intercept", plot),
            Field("time_m", 0.0, "real", "save", "intercept"),
            Field("SBTLM", zeros3, "vec", "save", "intercept"),
            Field("STMEL", zeros3, "vec", "save", "intercept"),
            Field("SBMEL", zeros3, "vec", "save", "intercept"),
            Field("mode", 0, "int", "diag", "intercept", scrn_plot),
            Field("dbt", 0.0, "real", "diag", "intercept", scrn_plot),
            Field("psiptx", 0.0, "real", "diag/data", "intercept", plot),
            Field("thtptx", 0.0, "real", "diag/data", "intercept", plot),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def terminate(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mterm = store.get("mterm")
        if mterm == 2:
            raise ValueError(f"unknown mterm {mterm}")
        write = store.get("write")
        psiptx = store.get("psiptx")
        thtptx = store.get("thtptx")
        time_m = store.get("time_m")
        sbtlm = store.get("SBTLM")
        stmel = store.get("STMEL")
        sbmel = store.get("SBMEL")
        time = store.get("time")
        stop = store.get("stop")
        alt = store.get("alt")
        hbe = store.get("hbe")
        sbel = store.get("SBEL")
        vbel = store.get("VBEL")
        stel = store.get("STEL")
        vtel = store.get("VTEL")
        tgt_slot = store.get("tgt_slot")
        mseek = store.get("mseek")
        trcond = store.get("trcond")
        mguide = store.get("mguide")
        ip_sltrange = store.get("ip_sltrange")
        siblc = store.get("SIBLC")
        maut = store.get("maut")
        mprop = store.get("mprop")
        int_step = ctx.int_step

        guid_mid = mguide // 10
        guid_term = mguide % 10
        skr_type = mseek // 10
        skr_mode = mseek % 10
        mode = (
            100000 * skr_type
            + 10000 * skr_mode
            + 1000 * guid_mid
            + 100 * guid_term
            + 10 * maut
            + mprop
        )

        stbl = stel - sbel
        dbt = sqrt(float(stbl[0] ** 2 + stbl[1] ** 2 + stbl[2] ** 2))
        hit_time = 0.0
        miss_vec = np.zeros(3)
        miss = 0.0

        if trcond and stop:
            vehicle.health = 0
            ctx.combus[ctx.vehicle_slot].status = 0

        if ip_sltrange < 500:
            with np.errstate(divide="ignore", invalid="ignore"):
                uibl = siblc * np.divide(1.0, ip_sltrange)
            closing_speed = float(uibl @ vbel)
            if (closing_speed < 0) and write:
                write = 0
                vehicle.health = 0
                ctx.combus[ctx.vehicle_slot].status = 0

        if (alt <= 0 or hbe <= 0) and write:
            write = 0
            vehicle.health = 0
            ctx.combus[ctx.vehicle_slot].status = 0

        if skr_mode == 4:
            if dbt < 500:
                with np.errstate(divide="ignore", invalid="ignore"):
                    utbl = stbl * np.divide(1.0, dbt)
                vtbel = vtel - vbel
                closing_speed = float(utbl @ vtbel)
                sbtl = stbl * (-1.0)
                if (closing_speed > 0) and write:
                    write = 0
                    sbbml = sbel - sbmel
                    sttml = stel - stmel
                    hit_time = time_m - int_step * float(
                        (sbbml - sttml) @ sbtlm
                    ) / float(sbbml @ sbbml)
                    if mterm == 0:
                        tau = hit_time - time_m
                        miss_vec = (sbbml - sttml) * (tau / int_step) + sbtlm
                        miss = sqrt(
                            float(
                                miss_vec[0] ** 2
                                + miss_vec[1] ** 2
                                + miss_vec[2] ** 2
                            )
                        )
                    if mterm >= 1:
                        polar = polar_from_cart(vtel)
                        ttl = mat2tr(float(polar[1]), float(polar[2]))
                        if mterm == 1:
                            vtbet = ttl @ vtbel
                            vbtet = vtbet * (-1.0)
                            polar_asp = polar_from_cart(vbtet)
                            psiptx = float(polar_asp[1]) * DEG
                            thtptx = float(polar_asp[2]) * DEG - 90.0
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
                    ctx.combus[tgt_slot].status = 0
                sbtlm = sbtl
                stmel = stel
                sbmel = sbel
                time_m = time

        store.set("write", write)
        store.set("hit_time", hit_time)
        store.set("time_m", time_m)
        store.set("SBTLM", sbtlm)
        store.set("STMEL", stmel)
        store.set("SBMEL", sbmel)
        store.set("miss", miss)
        store.set("MISS", miss_vec)
        store.set("mode", mode)
        store.set("dbt", dbt)
        store.set("psiptx", psiptx)
        store.set("thtptx", thtptx)
