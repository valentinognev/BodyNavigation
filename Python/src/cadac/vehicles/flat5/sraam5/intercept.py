"""SRAAM5 intercept — Fortran G4 CPA + G4SHAZ + DTCT/DBTC (MSEEK=5)."""

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


class Sraam5Intercept:
    name = "intercept"

    def define(self, vehicle):
        store = vehicle.store
        plot = ("plot",)
        for field in (
            Field("mterm", 0, "int", "data", "intercept"),
            Field("write", 1, "int", "init", "intercept"),
            Field("yss", 0.0, "real", "diag", "intercept", plot),
            Field("zss", 0.0, "real", "diag", "intercept", plot),
            Field("dyrb", 0.0, "real", "diag", "intercept", plot),
            Field("dzrb", 0.0, "real", "diag", "intercept"),
            Field("aspazx", 0.0, "real", "diag", "intercept"),
            Field("aspelx", 0.0, "real", "diag", "intercept"),
            Field("azintx", 0.0, "real", "diag", "intercept"),
            Field("elintx", 0.0, "real", "diag", "intercept"),
            Field("dtct", 0.0, "real", "diag", "intercept", plot),
            Field("dbtc", 0.0, "real", "diag", "intercept", plot),
            Field("EXX", _ZEROS3, "vec", "data", "intercept"),
            Field("psiptx", 0.0, "real", "diag/data", "intercept", plot),
            Field("thtptx", 0.0, "real", "diag/data", "intercept", plot),
            Field("SBELM", _ZEROS3, "vec", "save", "intercept"),
            Field("time_m", 0.0, "real", "save", "intercept"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def terminate(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        """Fortran G4 CPA; DTCT/DBTC written when MSEEK=5."""
        store = vehicle.store
        mterm = store.get("mterm")
        if mterm not in (0, 1, 2):
            raise ValueError(f"unknown mterm {mterm}")
        write = store.get("write")
        time = store.get("time")
        time_m = store.get("time_m")
        sbel = np.asarray(store.get("SBEL"), dtype=float)
        vbel = np.asarray(store.get("VBEL"), dtype=float)
        st1el = np.asarray(store.get("ST1EL"), dtype=float)
        vt1el = np.asarray(store.get("VT1EL"), dtype=float)
        sbelm = np.asarray(store.get("SBELM"), dtype=float)
        psiptx = store.get("psiptx")
        thtptx = store.get("thtptx")
        lconv = store.get("lconv") if "lconv" in store else 0
        dtct = float(store.get("dtct"))
        dbtc = float(store.get("dbtc"))

        sbt1l = sbel - st1el
        dbt1 = float(np.sqrt(float(sbt1l @ sbt1l)))

        if dbt1 < 50.0:
            vbt1l = vbel - vt1el
            cvel = float(sbt1l @ vbt1l) / dbt1
            if "TT1L" in store:
                tt1l = np.asarray(store.get("TT1L"), dtype=float)
            else:
                polar = polar_from_cart(vt1el)
                tt1l = mat2tr(float(polar[1]), float(polar[2]))

            if mterm < 2:
                vbt1t1 = tt1l @ vbt1l
                _dv, psiyt1, thtyt1 = polar_from_cart(vbt1t1)
                psiptx = float(psiyt1) * DEG
                thtptx = float(thtyt1) * DEG - 90.0
            tpt1 = mat2tr(psiptx * RAD, thtptx * RAD)
            tpl = tpt1 @ tt1l
            sbt1p = tpl @ sbt1l

            if (cvel > 0.0) and write:
                write = 0
                sbbml = sbel - sbelm
                sbbmp = tpl @ sbbml
                stbmp = sbbmp - sbt1p
                ww = float(stbmp[2] / sbbmp[2])
                sbtp = sbbmp * ww - stbmp
                lconv = 2
                sbtl_local = tpl.T @ sbtp
                if mterm > 0:
                    g4shaz(store, sbtl_local, vbt1l)
                # Fortran G4: DTCT/DBTC on every CPA (MTERM gates G4SHAZ only)
                exx = store.get("EXX")
                dtct, dbtc = g4_dtct_dbtc(sbtp, tpl, exx)
                vehicle.health = 0
                if ctx is not None and getattr(ctx, "combus", None):
                    ctx.combus[ctx.vehicle_slot].status = 0

            time_m = time
            sbelm = np.array(sbel, dtype=float, copy=True)

        store.set("write", write)
        store.set("time_m", time_m)
        store.set("SBELM", sbelm)
        store.set("psiptx", psiptx)
        store.set("thtptx", thtptx)
        store.set("dtct", dtct)
        store.set("dbtc", dbtc)
        if "lconv" in store:
            store.set("lconv", lconv)
