"""HYPER6 intercept — port of Hyper::intercept (intercept.cpp)."""

from math import sqrt

import numpy as np

from cadac.kernel.state import Field
from cadac.math.frames import cadac_matmul, skew

_ZEROS3 = (0.0, 0.0, 0.0)
_PLOT = ("plot",)
_SCRN_PLOT = ("scrn", "plot")


def _univec3(vec):
    v1 = float(vec[0])
    v2 = float(vec[1])
    v3 = float(vec[2])
    scale = sqrt(v1 * v1 + v2 * v2 + v3 * v3)
    if scale == 0.0:
        return np.zeros(3)
    return np.array([v1 / scale, v2 / scale, v3 / scale])


def _absolute(vec):
    return sqrt(
        float(vec[0]) * float(vec[0])
        + float(vec[1]) * float(vec[1])
        + float(vec[2]) * float(vec[2])
    )


def _dot(a, b):
    return (
        float(a[0]) * float(b[0])
        + float(a[1]) * float(b[1])
        + float(a[2]) * float(b[2])
    )


def _vec(store, name):
    return np.asarray(store.get(name), dtype=float).copy()


class Hyper6Intercept:
    name = "intercept"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("mintercept", 0, "int", "data", "intercept"),
            Field("write", 1, "int", "init", "intercept"),
            Field("miss", 0.0, "real", "diag", "intercept", _PLOT),
            Field("hit_time", 0.0, "real", "diag", "intercept"),
            Field("MISS_I", _ZEROS3, "vec", "diag", "intercept", _PLOT),
            Field("time_m", 0.0, "real", "save", "intercept"),
            Field("SBTIM", _ZEROS3, "vec", "save", "intercept"),
            Field("STMII", _ZEROS3, "vec", "save", "intercept"),
            Field("SBMII", _ZEROS3, "vec", "save", "intercept"),
            Field("event", 0, "int", "diag", "intercept", _SCRN_PLOT),
            Field("dbt", 0.0, "real", "diag", "intercept", _SCRN_PLOT),
            Field("MISS_H", _ZEROS3, "vec", "diag", "intercept", _PLOT),
            Field("MISS_L", _ZEROS3, "vec", "diag", "intercept", _PLOT),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def terminate(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        write = int(store.get("write"))
        time = float(store.get("time"))
        alt = float(store.get("alt"))
        sbii = _vec(store, "SBII")
        vbii = _vec(store, "VBII")
        stii = _vec(store, "STII")
        vtii = _vec(store, "VTII")
        mprop = int(store.get("mprop"))
        mrcs_moment = int(store.get("mrcs_moment"))
        mseek = int(store.get("mseek"))
        mguide = int(store.get("mguide"))
        wp_alt = float(store.get("wp_alt"))
        swbd = _vec(store, "SWBD")
        wp_flag = int(store.get("wp_flag"))
        maut = int(store.get("maut"))
        time_m = float(store.get("time_m"))
        sbtim = _vec(store, "SBTIM")
        stmii = _vec(store, "STMII")
        sbmii = _vec(store, "SBMII")
        int_step = float(ctx.int_step)

        miss = 0.0
        hit_time = 0.0
        miss_i = np.zeros(3)
        miss_h = np.zeros(3)
        miss_l = np.zeros(3)

        mauty = maut // 10
        mautp = maut % 10
        rcs_type = mrcs_moment // 10
        rcs_mode = mrcs_moment % 10
        event = int(
            mseek * 1000000
            + mguide * 100000
            + mauty * 10000
            + mautp * 1000
            + rcs_type * 100
            + rcs_mode * 10
            + mprop
        )

        stbi = stii - sbii
        dbt = _absolute(stbi)

        if alt <= 0 and write:
            write = 0
            vehicle.health = 0
            if ctx.combus is not None:
                ctx.combus[ctx.vehicle_slot].status = 0

        if mguide == 4 or mguide == 30:
            if wp_flag == 1:
                write = 1
            if wp_flag == -1 and write:
                write = 0

        if mguide == 33:
            if wp_flag == 1:
                write = 1
            if alt <= wp_alt and write:
                write = 0
                miss = _absolute(swbd)
                vehicle.health = 0
                if ctx.combus is not None:
                    ctx.combus[ctx.vehicle_slot].status = 0

        if mguide == 6 or mguide == 7 or mguide == 8:
            if wp_flag == -1:
                write = 1
            if dbt < 20e3:
                utbi = stbi * (1.0 / dbt)
                vtbi = vtii - vbii
                closing_speed = _dot(utbi, vtbi)
                sbti = stbi * (-1.0)
                if closing_speed > 0 and write:
                    write = 0
                    sbbmi = sbii - sbmii
                    sttmi = stii - stmii
                    hit_time = time_m - int_step * _dot(sbbmi - sttmi, sbtim) / _dot(
                        sbbmi, sbbmi
                    )
                    tau = hit_time - time_m
                    miss_i = (sbbmi - sttmi) * (tau / int_step) + sbtim
                    miss = _absolute(miss_i)

                    if mguide == 6 or mguide == 7:
                        uh1 = _univec3(stii)
                        uh3 = _univec3(cadac_matmul(skew(stii), vtii))
                        uh2 = cadac_matmul(skew(uh3), uh1)
                        thi = np.vstack((uh1, uh2, uh3))
                        miss_h = cadac_matmul(thi, miss_i)

                    if mguide == 8:
                        ul1i = _univec3(vtii)
                        ul3i = _univec3(stii) * (-1.0)
                        ul2i = cadac_matmul(skew(ul3i), ul1i)
                        tli = np.vstack((ul1i, ul2i, ul3i))
                        miss_l = cadac_matmul(tli, miss_i)

                    vehicle.health = 0
                    if ctx.combus is not None:
                        ctx.combus[ctx.vehicle_slot].status = 0

                sbtim = sbti
                stmii = stii
                sbmii = sbii
                time_m = time

        store.set("write", write)
        store.set("SBTIM", sbtim)
        store.set("STMII", stmii)
        store.set("SBMII", sbmii)
        store.set("time_m", time_m)
        store.set("miss", miss)
        store.set("hit_time", hit_time)
        store.set("MISS_I", miss_i)
        store.set("event", event)
        store.set("dbt", dbt)
        store.set("MISS_H", miss_h)
        store.set("MISS_L", miss_l)
