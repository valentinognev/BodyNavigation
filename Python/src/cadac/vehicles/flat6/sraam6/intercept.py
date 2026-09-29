from math import copysign, sqrt

import numpy as np

from cadac.constants import DEG, RAD
from cadac.kernel.state import Field
from cadac.math.frames import cadac_matmul, mat2tr, polar_from_cart

_ZEROS3 = (0.0, 0.0, 0.0)
# Fortran G4SHAZ PARAMETER(PI=3.14159)
_FTN_PI = 3.14159


def g4shaz(store, sbtl, vbt1l):
    """Fortran G4SHAZ — AFATL-TR-86-32 SHAZAM geometry (YSS/ZSS/DYRB)."""
    if "VT1EL" in store:
        vt1el = np.asarray(store.get("VT1EL"), dtype=float)
    else:
        vt1el = np.asarray(store.get("VTEL"), dtype=float)
    vbel = np.asarray(store.get("VBEL"), dtype=float)
    if "TT1L" in store:
        tt1l = np.asarray(store.get("TT1L"), dtype=float)
    elif "TTL" in store:
        tt1l = np.asarray(store.get("TTL"), dtype=float)
    else:
        polar = polar_from_cart(vt1el)
        tt1l = mat2tr(float(polar[1]), float(polar[2]))

    sbtl = np.asarray(sbtl, dtype=float)
    vbt1l = np.asarray(vbt1l, dtype=float)

    _dvt1e, psiul, thtul = polar_from_cart(vt1el)
    tul = mat2tr(float(psiul), float(thtul))
    vbeu = cadac_matmul(tul, vbel)
    _dvbe, psivu, thtvu = polar_from_cart(vbeu)
    psivu = float(psivu)
    thtvu = float(thtvu)
    aspaz = -copysign((_FTN_PI - abs(psivu)), psivu)
    aspel = -thtvu
    aspazx = aspaz * DEG
    aspelx = aspel * DEG

    vbt1t1 = cadac_matmul(tt1l, vbt1l)
    _dvbt1, psiyt1, thtyt1 = polar_from_cart(vbt1t1)
    psiyt1 = float(psiyt1)
    thtyt1 = float(thtyt1)
    elint = thtyt1
    azint = copysign((_FTN_PI - abs(psiyt1)), psiyt1)
    elintx = elint * DEG
    azintx = azint * DEG

    vbt1u = cadac_matmul(tul, vbt1l)
    _dv2, psizu, thtzu = polar_from_cart(vbt1u)
    tzu = mat2tr(float(psizu), float(thtzu))
    tzl = cadac_matmul(tzu, tul)
    shjz = cadac_matmul(tzl, sbtl)
    yss = -float(shjz[1])
    zss = -float(shjz[2])

    tyt1 = mat2tr(psiyt1, thtyt1)
    tyl = cadac_matmul(tyt1, tt1l)
    shjy = cadac_matmul(tyl, sbtl)
    dyrb = -float(shjy[1])
    dzrb = -float(shjy[2])

    store.set("yss", yss)
    store.set("zss", zss)
    store.set("dyrb", dyrb)
    store.set("dzrb", dzrb)
    store.set("aspazx", aspazx)
    store.set("aspelx", aspelx)
    store.set("azintx", azintx)
    store.set("elintx", elintx)


def g4_dtct_dbtc(sbtp, tpl, exx):
    """Fortran G4 — nav miss DTCT and G&C miss DBTC in intercept plane.

    STCTL = -EXX(1:3); STCTP = TPL*STCTL; DTCT = |STCTP_xy|;
    SBTCP = SBTP - STCTP; DBTC = |SBTCP_xy|.
    """
    stctl = -np.asarray(exx, dtype=float).reshape(-1)[:3]
    stctp = np.asarray(tpl, dtype=float) @ stctl
    dtct = sqrt(float(stctp[0] ** 2 + stctp[1] ** 2))
    sbtcp = np.asarray(sbtp, dtype=float) - stctp
    dbtc = sqrt(float(sbtcp[0] ** 2 + sbtcp[1] ** 2))
    return dtct, dbtc


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
            Field("yss", 0.0, "real", "diag", "intercept", plot),
            Field("zss", 0.0, "real", "diag", "intercept", plot),
            Field("dyrb", 0.0, "real", "diag", "intercept", plot),
            Field("dzrb", 0.0, "real", "diag", "intercept"),
            Field("aspazx", 0.0, "real", "diag", "intercept"),
            Field("aspelx", 0.0, "real", "diag", "intercept"),
            Field("azintx", 0.0, "real", "diag", "intercept"),
            Field("elintx", 0.0, "real", "diag", "intercept"),
            # Fortran G4: nav miss DTCT and G&C miss DBTC (SO5A / MSEEK=5)
            Field("dtct", 0.0, "real", "diag", "intercept", plot),
            Field("dbtc", 0.0, "real", "diag", "intercept", plot),
            Field("EXX", _ZEROS3, "vec", "data", "intercept"),
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
        dtct = float(store.get("dtct"))
        dbtc = float(store.get("dbtc"))

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
                    tpl = None
                    sbtp_plane = None
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
                        sbtp_plane = miss_vec
                        miss = sqrt(
                            float(
                                miss_vec[0] ** 2
                                + miss_vec[1] ** 2
                                + miss_vec[2] ** 2
                            )
                        )
                        # Fortran: SBTL = TLP*SBTP; VBT1L = VBEL-VT1EL; G4SHAZ
                        sbtl_local = tpl.T @ miss_vec
                        vbt1l = vbel - vtel
                        g4shaz(store, sbtl_local, vbt1l)
                    # Fortran G4 always builds TPL/SBTP then DTCT/DBTC at CPA
                    # (MTERM gates G4SHAZ only; no MSEEK if).
                    if tpl is None:
                        polar = polar_from_cart(vtel)
                        ttl = mat2tr(float(polar[1]), float(polar[2]))
                        vbt1l = vbel - vtel
                        vbt1t1 = ttl @ vbt1l
                        _dv, psiyt1, thtyt1 = polar_from_cart(vbt1t1)
                        psiptx = float(psiyt1) * DEG
                        thtptx = float(thtyt1) * DEG - 90.0
                        tpt = mat2tr(psiptx * RAD, thtptx * RAD)
                        tpl = tpt @ ttl
                        sbtp = tpl @ sbtl
                        sbbmp = tpl @ sbbml
                        stbmp = sbbmp - sbtp
                        ww = float(stbmp[2] / sbbmp[2])
                        sbtp_plane = sbbmp * ww - stbmp
                    exx = store.get("EXX")
                    dtct, dbtc = g4_dtct_dbtc(sbtp_plane, tpl, exx)
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
        store.set("dtct", dtct)
        store.set("dbtc", dbtc)
