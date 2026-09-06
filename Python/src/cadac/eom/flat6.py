import math

import numpy as np

from cadac.constants import DEG, EPS, PI, R
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import mat3tr


def _cadac_sign(variable):
    if variable < 0.0:
        return -1
    return 1


class Flat6Environment:
    name = "environment"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("mwind", 0, "int", "data", "environment"),
            Field("press", 0.0, "real", "out", "environment"),
            Field("rho", 0.0, "real", "out", "environment"),
            Field("vsound", 0.0, "real", "diag", "environment"),
            Field("grav", 0.0, "real", "out", "environment"),
            Field("vmach", 0.0, "real", "out", "environment", ("scrn", "plot", "com")),
            Field("pdynmc", 0.0, "real", "out", "environment", ("scrn", "plot")),
            Field("tempk", 0.0, "real", "out", "environment"),
            Field("VAEL", zeros3, "vec", "out", "environment"),
            Field("dvba", 0.0, "real", "out", "environment", ("plot",)),
            Field("VBAL", zeros3, "vec", "out", "environment"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mwind = store.get("mwind")
        if mwind != 0:
            raise ValueError(f"unknown mwind {mwind}")
        hbe = store.get("hbe")
        vbel = store.get("VBEL")
        rho, press, tempk = atmosphere76(hbe)
        vsound = math.sqrt(1.4 * R * tempk)
        vael = np.zeros(3)
        vbal = vbel - vael
        dvba = float(np.linalg.norm(vbal))
        vmach = abs(dvba / vsound)
        pdynmc = 0.5 * rho * dvba**2
        store.set("grav", gravity(hbe))
        store.set("rho", rho)
        store.set("press", press)
        store.set("tempk", tempk)
        store.set("vsound", vsound)
        store.set("VAEL", vael)
        store.set("VBAL", vbal)
        store.set("dvba", dvba)
        store.set("vmach", vmach)
        store.set("pdynmc", pdynmc)

    def terminate(self, vehicle, ctx):
        pass


class Flat6Kinematics:
    name = "kinematics"

    def define(self, vehicle):
        store = vehicle.store
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for field in (
            Field("ck", 50.0, "real", "data", "kinematics"),
            Field("q0d", 0.0, "real", "state", "kinematics"),
            Field("q0", 0.0, "real", "state", "kinematics"),
            Field("q1d", 0.0, "real", "state", "kinematics"),
            Field("q1", 0.0, "real", "state", "kinematics"),
            Field("q2d", 0.0, "real", "state", "kinematics"),
            Field("q2", 0.0, "real", "state", "kinematics"),
            Field("q3d", 0.0, "real", "state", "kinematics"),
            Field("q3", 0.0, "real", "state", "kinematics"),
            Field("TBL", zeros33, "mat", "out", "kinematics"),
            Field("psibl", 0.0, "real", "diag", "kinematics"),
            Field("thtbl", 0.0, "real", "diag", "kinematics"),
            Field("phibl", 0.0, "real", "diag", "kinematics"),
            Field("psiblx", 0.0, "real", "in/di", "kinematics", ("scrn", "plot")),
            Field("thtblx", 0.0, "real", "in/di", "kinematics", ("scrn", "plot")),
            Field("phiblx", 0.0, "real", "in/di", "kinematics", ("scrn", "plot")),
            Field("alppx", 0.0, "real", "out", "kinematics", ("plot",)),
            Field("phipx", 0.0, "real", "out", "kinematics", ("plot",)),
            Field("alpp", 0.0, "real", "out", "kinematics"),
            Field("phip", 0.0, "real", "out", "kinematics"),
            Field("alphax", 0.0, "real", "diag", "kinematics", ("scrn", "plot")),
            Field("betax", 0.0, "real", "diag", "kinematics", ("scrn", "plot")),
            Field("erq", 0.0, "real", "diag", "kinematics", ("plot",)),
            Field("etbl", 0.0, "real", "diag", "kinematics", ("plot",)),
            Field("TLB", zeros33, "mat", "diag", "kinematics"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        psiblx = store.get("psiblx")
        thtblx = store.get("thtblx")
        phiblx = store.get("phiblx")
        spsi = math.sin(psiblx / (2.0 * DEG))
        cpsi = math.cos(psiblx / (2.0 * DEG))
        stht = math.sin(thtblx / (2.0 * DEG))
        ctht = math.cos(thtblx / (2.0 * DEG))
        sphi = math.sin(phiblx / (2.0 * DEG))
        cphi = math.cos(phiblx / (2.0 * DEG))
        q0 = cpsi * ctht * cphi + spsi * stht * sphi
        q1 = cpsi * ctht * sphi - spsi * stht * cphi
        q2 = cpsi * stht * cphi + spsi * ctht * sphi
        q3 = -cpsi * stht * sphi + spsi * ctht * cphi
        tbl = mat3tr(psiblx / DEG, thtblx / DEG, phiblx / DEG)
        store.set("q0", q0)
        store.set("q1", q1)
        store.set("q2", q2)
        store.set("q3", q3)
        store.set("TBL", tbl)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        ck = store.get("ck")
        dvba = store.get("dvba")
        vbal = store.get("VBAL")
        wbeb = store.get("WBEB")
        q0d = store.get("q0d")
        q0 = store.get("q0")
        q1d = store.get("q1d")
        q1 = store.get("q1")
        q2d = store.get("q2d")
        q2 = store.get("q2")
        q3d = store.get("q3d")
        q3 = store.get("q3")
        int_step = ctx.int_step

        quat_metric = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3
        erq = 1.0 - quat_metric
        pp = wbeb[0]
        qq = wbeb[1]
        rr = wbeb[2]
        new_q0d = 0.5 * (-pp * q1 - qq * q2 - rr * q3) + ck * erq * q0
        new_q1d = 0.5 * (pp * q0 + rr * q2 - qq * q3) + ck * erq * q1
        new_q2d = 0.5 * (qq * q0 - rr * q1 + pp * q3) + ck * erq * q2
        new_q3d = 0.5 * (rr * q0 + qq * q1 - pp * q2) + ck * erq * q3
        q0 = integrate(new_q0d, q0d, q0, int_step)
        q1 = integrate(new_q1d, q1d, q1, int_step)
        q2 = integrate(new_q2d, q2d, q2, int_step)
        q3 = integrate(new_q3d, q3d, q3, int_step)
        q0d = new_q0d
        q1d = new_q1d
        q2d = new_q2d
        q3d = new_q3d

        tbl = np.zeros((3, 3))
        tbl[0, 0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
        tbl[0, 1] = 2.0 * (q1 * q2 + q0 * q3)
        tbl[0, 2] = 2.0 * (q1 * q3 - q0 * q2)
        tbl[1, 0] = 2.0 * (q1 * q2 - q0 * q3)
        tbl[1, 1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
        tbl[1, 2] = 2.0 * (q2 * q3 + q0 * q1)
        tbl[2, 0] = 2.0 * (q1 * q3 + q0 * q2)
        tbl[2, 1] = 2.0 * (q2 * q3 - q0 * q1)
        tbl[2, 2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3

        tlb = tbl.T.copy()
        ubl = tlb @ tbl
        e1 = ubl[0, 0] - 1.0
        e2 = ubl[1, 1] - 1.0
        e3 = ubl[2, 2] - 1.0
        etbl = math.sqrt(e1 * e1 + e2 * e2 + e3 * e3)

        tbl13 = tbl[0, 2]
        tbl11 = tbl[0, 0]
        tbl33 = tbl[2, 2]
        tbl12 = tbl[0, 1]
        tbl23 = tbl[1, 2]
        if math.fabs(tbl13) < 1.0:
            thtbl = math.asin(-tbl13)
            cthtbl = math.cos(thtbl)
        else:
            thtbl = PI / 2.0 * _cadac_sign(-tbl13)
            cthtbl = EPS
        cpsi = tbl11 / cthtbl
        if math.fabs(cpsi) >= 1.0:
            cpsi = (1.0 - EPS) * _cadac_sign(cpsi)
        cphi = tbl33 / cthtbl
        if math.fabs(cphi) >= 1.0:
            cphi = (1.0 - EPS) * _cadac_sign(cphi)
        psibl = math.acos(cpsi) * _cadac_sign(tbl12)
        phibl = math.acos(cphi) * _cadac_sign(tbl23)
        psiblx = DEG * psibl
        thtblx = DEG * thtbl
        phiblx = DEG * phibl

        vbab = tbl @ vbal
        vbab1 = vbab[0]
        vbab2 = vbab[1]
        vbab3 = vbab[2]
        alpha = math.atan2(vbab3, vbab1)
        beta = math.asin(vbab2 / dvba)
        dum = vbab1 / dvba
        if math.fabs(dum) > 1.0:
            dum = 1.0 * _cadac_sign(dum)
        alpp = math.acos(dum)
        if vbab2 == 0.0 and vbab3 == 0.0:
            phip = 0.0
        elif math.fabs(vbab2) < EPS:
            phip = 0.0
            if vbab3 > 0.0:
                phip = 0.0
            if vbab3 < 0.0:
                phip = PI
        else:
            phip = math.atan2(vbab2, vbab3)
        alphax = alpha * DEG
        betax = beta * DEG
        alppx = alpp * DEG
        phipx = phip * DEG

        names = store.names()
        if all(n in names for n in ("trcode", "tralppx", "tralpnx", "trbetx")):
            trcode = store.get("trcode")
            tralppx = store.get("tralppx")
            tralpnx = store.get("tralpnx")
            trbetx = store.get("trbetx")
            if alphax > tralppx:
                trcode = 5
            if alphax < tralpnx:
                trcode = 6
            if math.fabs(betax) > trbetx:
                trcode = 7
            store.set("trcode", trcode)

        store.set("q0d", q0d)
        store.set("q0", q0)
        store.set("q1d", q1d)
        store.set("q1", q1)
        store.set("q2d", q2d)
        store.set("q2", q2)
        store.set("q3d", q3d)
        store.set("q3", q3)
        store.set("TBL", tbl)
        store.set("alphax", alphax)
        store.set("betax", betax)
        store.set("psibl", psibl)
        store.set("thtbl", thtbl)
        store.set("phibl", phibl)
        store.set("psiblx", psiblx)
        store.set("thtblx", thtblx)
        store.set("phiblx", phiblx)
        store.set("alppx", alppx)
        store.set("phipx", phipx)
        store.set("erq", erq)
        store.set("etbl", etbl)
        store.set("TLB", tlb)

    def terminate(self, vehicle, ctx):
        pass
