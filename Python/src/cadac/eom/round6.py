import math

import numpy as np

from cadac.constants import DEG, EPS, PI, R, RAD, WEII3
from cadac.env.us76 import atmosphere76
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import mat3tr
from cadac.math.wgs84 import cad_grav84, cad_tdi84


def _cadac_sign(variable):
    if variable < 0.0:
        return -1
    return 1


def _skew(vec):
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


class Round6Environment:
    name = "environment"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("mair", 0, "int", "data", "environment"),
            Field("press", 0.0, "real", "out", "environment"),
            Field("rho", 0.0, "real", "out", "environment"),
            Field("vsound", 0.0, "real", "diag", "environment"),
            Field("vmach", 0.0, "real", "out", "environment", ("scrn", "plot", "com")),
            Field("pdynmc", 0.0, "real", "out", "environment", ("scrn", "plot")),
            Field("tempk", 0.0, "real", "out", "environment"),
            Field("mfreeze_evrn", 0, "int", "save", "environment"),
            Field("pdynmcf", 0.0, "real", "save", "environment"),
            Field("vmachf", 0.0, "real", "save", "environment"),
            Field("GRAVG", zeros3, "vec", "out", "environment"),
            Field("grav", 0.0, "real", "out", "environment"),
            Field("dvae", 0.0, "real", "data", "environment"),
            Field("dvael", 0.0, "real", "data", "environment"),
            Field("waltl", 0.0, "real", "data", "environment"),
            Field("dvaeh", 0.0, "real", "data", "environment"),
            Field("walth", 0.0, "real", "data", "environment"),
            Field("vaed3", 0.0, "real", "data", "environment"),
            Field("psiwdx", 0.0, "real", "data", "environment"),
            Field("twind", 0.1, "real", "data", "environment"),
            Field("VAEDS", zeros3, "vec", "state", "environment"),
            Field("VAEDSD", zeros3, "vec", "state", "environment"),
            Field("VAED", zeros3, "vec", "out", "environment"),
            Field("dvba", 0.0, "real", "out", "environment"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mair = store.get("mair")
        matmo = mair // 100
        mturb = (mair - matmo * 100) // 10
        mwind = (mair - matmo * 100) % 10
        if matmo != 0 or mturb != 0 or mwind != 0:
            raise ValueError(f"unknown mair {mair}")

        dvba = store.get("dvba")
        vaeds = store.get("VAEDS")
        vaedsd = store.get("VAEDSD")
        time = store.get("time")
        alt = store.get("alt")
        vbed = store.get("VBED")
        sbii = store.get("SBII")

        gravg = cad_grav84(sbii, time)
        grav = float(np.linalg.norm(gravg))

        rho, press, tempk = atmosphere76(alt)
        vsound = math.sqrt(1.4 * R * tempk)

        vmach = abs(dvba / vsound)
        pdynmc = 0.5 * rho * dvba * dvba

        vaed = np.zeros(3)
        vbad = vbed - vaed
        dvba = float(np.linalg.norm(vbad))
        vmach = abs(dvba / vsound)
        pdynmc = 0.5 * rho * dvba * dvba

        names = store.names()
        if "trcode" in names and "mguid" in names:
            if store.get("mguid") == 6:
                trcode = store.get("trcode")
                if vmach <= store.get("trmach"):
                    trcode = 2.0
                if pdynmc <= store.get("trdynm"):
                    trcode = 3.0
                store.set("trcode", trcode)

        if "mfreeze" in names:
            mfreeze = store.get("mfreeze")
            mfreeze_evrn = store.get("mfreeze_evrn")
            pdynmcf = store.get("pdynmcf")
            vmachf = store.get("vmachf")
            if mfreeze == 0:
                mfreeze_evrn = 0
            else:
                if mfreeze != mfreeze_evrn:
                    mfreeze_evrn = mfreeze
                    vmachf = vmach
                    pdynmcf = pdynmc
                vmach = vmachf
                pdynmc = pdynmcf
            store.set("mfreeze_evrn", mfreeze_evrn)
            store.set("pdynmcf", pdynmcf)
            store.set("vmachf", vmachf)

        store.set("VAEDS", vaeds)
        store.set("VAEDSD", vaedsd)
        store.set("press", press)
        store.set("rho", rho)
        store.set("vmach", vmach)
        store.set("pdynmc", pdynmc)
        store.set("GRAVG", gravg)
        store.set("grav", grav)
        store.set("VAED", vaed)
        store.set("dvba", dvba)
        store.set("vsound", vsound)
        store.set("tempk", tempk)

    def terminate(self, vehicle, ctx):
        pass


class Round6Kinematics:
    name = "kinematics"

    def define(self, vehicle):
        store = vehicle.store
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for field in (
            Field("time", 0.0, "real", "exec", "kinematics", ("scrn", "plot", "com")),
            Field("event_time", 0.0, "real", "exec", "kinematics"),
            Field("int_step_new", 0.0, "real", "data", "kinematics"),
            Field("out_step_fact", 0.0, "real", "data", "kinematics"),
            Field("TBD", zeros33, "mat", "out", "kinematics"),
            Field("TBI", zeros33, "mat", "state", "kinematics"),
            Field("TBID", zeros33, "mat", "state", "kinematics"),
            Field("ortho_error", 0.0, "real", "diag", "kinematics", ("scrn",)),
            Field("psibd", 0.0, "real", "diag", "kinematics", ("plot",)),
            Field("thtbd", 0.0, "real", "diag", "kinematics"),
            Field("phibd", 0.0, "real", "diag", "kinematics"),
            Field("psibdx", 0.0, "real", "in/di", "kinematics", ("scrn", "plot")),
            Field("thtbdx", 0.0, "real", "in/di", "kinematics", ("scrn", "plot")),
            Field("phibdx", 0.0, "real", "in/di", "kinematics", ("scrn", "plot")),
            Field("alppx", 0.0, "real", "out", "kinematics", ("plot",)),
            Field("phipx", 0.0, "real", "out", "kinematics"),
            Field("alphax", 0.0, "real", "init/diag", "kinematics", ("scrn", "plot")),
            Field("betax", 0.0, "real", "diag", "kinematics", ("scrn", "plot")),
            Field("alphaix", 0.0, "real", "diag", "kinematics", ("plot",)),
            Field("betaix", 0.0, "real", "diag", "kinematics", ("plot",)),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        time = ctx.sim_time
        int_step_new = ctx.int_step
        psibdx = store.get("psibdx")
        thtbdx = store.get("thtbdx")
        phibdx = store.get("phibdx")
        lonx = store.get("lonx")
        latx = store.get("latx")
        alt = store.get("alt")
        tbd = mat3tr(psibdx * RAD, thtbdx * RAD, phibdx * RAD)
        tdi = cad_tdi84(lonx * RAD, latx * RAD, alt, time)
        tbi = tbd @ tdi
        store.set("time", time)
        store.set("int_step_new", int_step_new)
        store.set("TBD", tbd)
        store.set("TBI", tbi)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        int_step_new = store.get("int_step_new")
        out_step_fact = store.get("out_step_fact")
        dvba = store.get("dvba")
        wbib = store.get("WBIB")
        lonx = store.get("lonx")
        latx = store.get("latx")
        alt = store.get("alt")
        vbed = store.get("VBED")
        vaed = store.get("VAED")
        vbii = store.get("VBII")
        tbi = store.get("TBI")
        tbid = store.get("TBID")

        time = ctx.sim_time
        ctx.int_step = int_step_new
        ctx.out_fact = out_step_fact
        int_step = ctx.int_step

        tbid_new = (-_skew(wbib)) @ tbi
        tbi = integrate(tbid_new, tbid, tbi, int_step)
        tbid = tbid_new

        unit = np.eye(3)
        ee = unit - tbi @ tbi.T
        tbi = tbi + ee @ tbi * 0.5

        e1 = ee[0, 0]
        e2 = ee[1, 1]
        e3 = ee[2, 2]
        ortho_error = math.sqrt(e1 * e1 + e2 * e2 + e3 * e3)

        tdi = cad_tdi84(lonx * RAD, latx * RAD, alt, time)
        tbd = tbi @ tdi.T
        tbd13 = tbd[0, 2]
        tbd11 = tbd[0, 0]
        tbd33 = tbd[2, 2]
        tbd12 = tbd[0, 1]
        tbd23 = tbd[1, 2]

        if math.fabs(tbd13) < 1.0:
            thtbd = math.asin(-tbd13)
            cthtbd = math.cos(thtbd)
        else:
            thtbd = PI / 2.0 * _cadac_sign(-tbd13)
            cthtbd = EPS
        cpsi = tbd11 / cthtbd
        if math.fabs(cpsi) > 1.0:
            cpsi = 1.0 * _cadac_sign(cpsi)
        cphi = tbd33 / cthtbd
        if math.fabs(cphi) > 1.0:
            cphi = 1.0 * _cadac_sign(cphi)
        psibd = math.acos(cpsi) * _cadac_sign(tbd12)
        phibd = math.acos(cphi) * _cadac_sign(tbd23)
        psibdx = DEG * psibd
        thtbdx = DEG * thtbd
        phibdx = DEG * phibd

        vbab = tbd @ (vbed - vaed)
        vbab1 = vbab[0]
        vbab2 = vbab[1]
        vbab3 = vbab[2]
        alpha = math.atan2(vbab3, vbab1)
        beta = math.asin(vbab2 / dvba)
        alphax = alpha * DEG
        betax = beta * DEG

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
        alppx = alpp * DEG
        phipx = phip * DEG

        vbib = tbi @ vbii
        vbib1 = vbib[0]
        vbib2 = vbib[1]
        vbib3 = vbib[2]
        alphai = math.atan2(vbib3, vbib1)
        dvbi = float(np.linalg.norm(vbib))
        betai = math.asin(vbib2 / dvbi)
        alphaix = alphai * DEG
        betaix = betai * DEG

        store.set("TBI", tbi)
        store.set("TBID", tbid)
        store.set("time", time)
        store.set("event_time", ctx.event_time)
        store.set("int_step_new", int_step_new)
        store.set("TBD", tbd)
        store.set("psibdx", psibdx)
        store.set("thtbdx", thtbdx)
        store.set("phibdx", phibdx)
        store.set("alppx", alppx)
        store.set("phipx", phipx)
        store.set("alphax", alphax)
        store.set("betax", betax)
        store.set("ortho_error", ortho_error)
        store.set("psibd", psibd)
        store.set("thtbd", thtbd)
        store.set("phibd", phibd)
        store.set("alphaix", alphaix)
        store.set("betaix", betaix)

    def terminate(self, vehicle, ctx):
        pass


class Round6Euler:
    name = "euler"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("ppx", 0.0, "real", "out", "euler", ("plot",)),
            Field("qqx", 0.0, "real", "out", "euler", ("plot",)),
            Field("rrx", 0.0, "real", "out", "euler", ("plot",)),
            Field("WBEB", zeros3, "vec", "diag", "euler"),
            Field("WBIB", zeros3, "vec", "state", "euler"),
            Field("WBIBD", zeros3, "vec", "state", "euler"),
            Field("WBII", zeros3, "vec", "out", "euler"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        ppx = store.get("ppx")
        qqx = store.get("qqx")
        rrx = store.get("rrx")
        tbi = store.get("TBI")
        wbeb = np.array([ppx * RAD, qqx * RAD, rrx * RAD], dtype=float)
        weii = np.array([0.0, 0.0, WEII3], dtype=float)
        wbib = wbeb + tbi @ weii
        store.set("WBIB", wbib)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        fmb = store.get("FMB")
        tbi = store.get("TBI")
        ibbb = store.get("IBBB")
        wbib = store.get("WBIB")
        wbibd = store.get("WBIBD")
        int_step = ctx.int_step
        wacc_next = np.linalg.inv(ibbb) @ (fmb - _skew(wbib) @ ibbb @ wbib)
        wbib = integrate(wacc_next, wbibd, wbib, int_step)
        wbibd = wacc_next
        wbii = tbi.T @ wbib
        weii = np.array([0.0, 0.0, WEII3], dtype=float)
        wbeb = wbib - tbi @ weii
        store.set("WBIB", wbib)
        store.set("WBIBD", wbibd)
        store.set("ppx", wbeb[0] * DEG)
        store.set("qqx", wbeb[1] * DEG)
        store.set("rrx", wbeb[2] * DEG)
        store.set("WBEB", wbeb)
        store.set("WBII", wbii)

    def terminate(self, vehicle, ctx):
        pass
