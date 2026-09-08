import math

import numpy as np

from cadac.constants import DEG, EPS, PI, R, RAD
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import cadac_sign, hypot3, mat2tr, mat3tr, quat_to_dcm, skew


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
        dvba = hypot3(vbal)
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

        tbl = quat_to_dcm(q0, q1, q2, q3)

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
            thtbl = PI / 2.0 * cadac_sign(-tbl13)
            cthtbl = EPS
        cpsi = tbl11 / cthtbl
        if math.fabs(cpsi) >= 1.0:
            cpsi = (1.0 - EPS) * cadac_sign(cpsi)
        cphi = tbl33 / cthtbl
        if math.fabs(cphi) >= 1.0:
            cphi = (1.0 - EPS) * cadac_sign(cphi)
        psibl = math.acos(cpsi) * cadac_sign(tbl12)
        phibl = math.acos(cphi) * cadac_sign(tbl23)
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
            dum = 1.0 * cadac_sign(dum)
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

        if all(n in store for n in ("trcode", "tralppx", "tralpnx", "trbetx")):
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
        store.set("alpp", alpp)
        store.set("phip", phip)
        store.set("erq", erq)
        store.set("etbl", etbl)
        store.set("TLB", tlb)

    def terminate(self, vehicle, ctx):
        pass


class Flat6Euler:
    name = "euler"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("ppx", 0.0, "real", "init/out", "euler", ("plot",)),
            Field("qqx", 0.0, "real", "init/out", "euler", ("plot",)),
            Field("rrx", 0.0, "real", "init/out", "euler", ("plot",)),
            Field("WBEB", zeros3, "vec", "state", "euler"),
            Field("WBEBD", zeros3, "vec", "state", "euler"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        ppx = store.get("ppx")
        qqx = store.get("qqx")
        rrx = store.get("rrx")
        store.set("WBEB", np.array([ppx * RAD, qqx * RAD, rrx * RAD], dtype=float))

    def execute(self, vehicle, ctx):
        store = vehicle.store
        fmb = store.get("FMB")
        ibbb = store.get("IBBB")
        eng_ang_mom = store.get("eng_ang_mom")
        wbeb = store.get("WBEB")
        wbebd = store.get("WBEBD")
        int_step = ctx.int_step
        l_engine = np.array([eng_ang_mom, 0.0, 0.0], dtype=float)
        wacc_next = np.linalg.inv(ibbb) @ (
            fmb - skew(wbeb) @ (ibbb @ wbeb + l_engine)
        )
        wbeb = integrate(wacc_next, wbebd, wbeb, int_step)
        wbebd = wacc_next
        store.set("WBEB", wbeb)
        store.set("WBEBD", wbebd)
        store.set("ppx", wbeb[0] * DEG)
        store.set("qqx", wbeb[1] * DEG)
        store.set("rrx", wbeb[2] * DEG)

    def terminate(self, vehicle, ctx):
        pass


def _flight_path_angles(vbel):
    vbel1 = float(vbel[0])
    vbel2 = float(vbel[1])
    vbel3 = float(vbel[2])
    if vbel1 == 0.0 and vbel2 == 0.0:
        psivl = 0.0
    else:
        psivl = math.atan2(vbel2, vbel1)
    thtvl = math.atan2(-vbel3, math.sqrt(vbel1 * vbel1 + vbel2 * vbel2))
    return psivl, thtvl


class Flat6Newton:
    name = "newton"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("time", 0.0, "real", "exec", "newton", ("scrn", "plot", "com")),
            Field("halt", 0, "int", "exec", "newton"),
            Field("VBEBD", zeros3, "vec", "state", "newton"),
            Field("VBEB", zeros3, "vec", "state", "newton", ("plot",)),
            Field("SBELD", zeros3, "vec", "state", "newton"),
            Field("SBEL", zeros3, "vec", "state", "newton", ("plot", "com")),
            Field("sbel1", 0.0, "real", "data", "newton"),
            Field("sbel2", 0.0, "real", "data", "newton"),
            Field("sbel3", 0.0, "real", "data", "newton"),
            Field("SBELM", zeros3, "vec", "save", "newton"),
            Field("groundrange", 0.0, "real", "diag", "newton"),
            Field("FSPB", zeros3, "vec", "out", "newton"),
            Field("VBEL", zeros3, "vec", "out", "newton", ("com", "scrn", "plot")),
            Field("dvbe", 0.0, "real", "in/out", "newton", ("plot",)),
            Field("alpha0x", 0.0, "real", "data", "newton"),
            Field("beta0x", 0.0, "real", "data", "newton"),
            Field("hbe", 0.0, "real", "out", "newton", ("scrn", "plot")),
            Field("psivlx", 0.0, "real", "diag", "newton", ("scrn", "plot")),
            Field("thtvlx", 0.0, "real", "diag", "newton", ("scrn", "plot")),
            Field("alx", 0.0, "real", "diag", "newton", ("plot",)),
            Field("anx", 0.0, "real", "diag", "newton", ("scrn", "plot")),
            Field("ayx", 0.0, "real", "diag", "newton", ("plot",)),
            Field("ATB", zeros3, "vec", "diag", "newton"),
            Field("mfreeze_newt", 0, "int", "save", "newton"),
            Field("dvbef", 0.0, "real", "save", "newton"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        sbel1 = store.get("sbel1")
        sbel2 = store.get("sbel2")
        sbel3 = store.get("sbel3")
        dvbe = store.get("dvbe")
        alpha0x = store.get("alpha0x")
        beta0x = store.get("beta0x")
        tbl = store.get("TBL")
        salp = math.sin(alpha0x * RAD)
        calp = math.cos(alpha0x * RAD)
        sbet = math.sin(beta0x * RAD)
        cbet = math.cos(beta0x * RAD)
        vbeb = np.array(
            [calp * cbet * dvbe, sbet * dvbe, salp * cbet * dvbe], dtype=float
        )
        vbel = tbl.T @ vbeb
        psivl, thtvl = _flight_path_angles(vbel)
        sbel = np.array([sbel1, sbel2, sbel3], dtype=float)
        store.set("VBEB", vbeb)
        store.set("SBEL", sbel)
        store.set("SBELM", sbel)
        store.set("VBEL", vbel)
        store.set("hbe", -float(sbel[2]))
        store.set("psivlx", psivl * DEG)
        store.set("thtvlx", thtvl * DEG)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mfreeze_newt = store.get("mfreeze_newt")
        dvbef = store.get("dvbef")
        sbelm = store.get("SBELM")
        groundrange = store.get("groundrange")
        grav = store.get("grav")
        tbl = store.get("TBL")
        wbeb = store.get("WBEB")
        fapb = store.get("FAPB")
        vmass = store.get("vmass")
        vbebd = store.get("VBEBD")
        vbeb = store.get("VBEB")
        sbeld = store.get("SBELD")
        sbel = store.get("SBEL")
        int_step = ctx.int_step

        time = ctx.sim_time
        atb = skew(wbeb) @ vbeb
        gravl = np.array([0.0, 0.0, grav], dtype=float)
        fspb = fapb * (1.0 / vmass)
        vbebd_new = fspb - atb + tbl @ gravl
        vbeb = integrate(vbebd_new, vbebd, vbeb, int_step)
        vbebd = vbebd_new
        vbel = tbl.T @ vbeb
        sbeld_new = vbel
        sbel = integrate(sbeld_new, sbeld, sbel, int_step)
        sbeld = sbeld_new

        psivl, thtvl = _flight_path_angles(vbel)
        psivlx = psivl * DEG
        thtvlx = thtvl * DEG
        dvbe = hypot3(vbel)
        hbe = -float(sbel[2])
        anx = -fspb[2] / grav
        ayx = fspb[1] / grav
        tvl = mat2tr(psivl, thtvl)
        tvb = tvl @ tbl.T
        fspv = tvb @ fspb
        alx = fspv[1] / grav

        if "mfreeze" in store:
            mfreeze = store.get("mfreeze")
            if mfreeze == 0:
                mfreeze_newt = 0
            else:
                if mfreeze != mfreeze_newt:
                    mfreeze_newt = mfreeze
                    dvbef = dvbe
                dvbe = dvbef

        del_sbel = np.asarray(sbel - sbelm, dtype=float).copy()
        del_sbel[2] = 0.0
        groundrange = groundrange + hypot3(del_sbel)
        sbelm = sbel

        store.set("VBEBD", vbebd)
        store.set("VBEB", vbeb)
        store.set("SBELD", sbeld)
        store.set("SBEL", sbel)
        store.set("SBELM", sbelm)
        store.set("groundrange", groundrange)
        store.set("mfreeze_newt", mfreeze_newt)
        store.set("dvbef", dvbef)
        store.set("time", time)
        store.set("FSPB", fspb)
        store.set("VBEL", vbel)
        store.set("dvbe", dvbe)
        store.set("hbe", hbe)
        store.set("psivlx", psivlx)
        store.set("thtvlx", thtvlx)
        store.set("alx", alx)
        store.set("anx", anx)
        store.set("ayx", ayx)
        store.set("ATB", atb)

    def terminate(self, vehicle, ctx):
        pass
