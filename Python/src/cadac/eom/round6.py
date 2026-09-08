"""Zipfel 6-DOF round-Earth equations of motion (CADAC Round6)."""

import math

import numpy as np

from cadac.constants import AGRAV, DEG, EPS, PI, R, RAD, REARTH, WEII3
from cadac.env.us76 import atmosphere76
from cadac.kernel.integrate import integrate
from cadac.kernel.module import ModuleBase
from cadac.kernel.state import Field
from cadac.math.frames import cadac_inverse, cadac_matmul, mat2tr, mat3tr, polar_from_cart, cadac_sign, skew
from cadac.math.wgs84 import cad_geo84_in, cad_grav84, cad_in_geo84, cad_tdi84, cad_tgi84
from cadac.stoch import ROCKET6_MARKOV_COUNT, dryden_white, prepare_for_dryden

FOOT = 3.280834
NMILES = 5.399568e-4


class Round6Environment(ModuleBase):
    """Zipfel 6-DOF round Earth atmosphere (CADAC ``round6_environment``)."""

    name = "environment"
    fields = (
        Field("mair", 0, "int", "data", "environment"),
        Field("warning_flag", 0, "int", "init", "environment"),
        Field("press", 0.0, "real", "out", "environment"),
        Field("rho", 0.0, "real", "out", "environment"),
        Field("vsound", 0.0, "real", "diag", "environment"),
        Field("vmach", 0.0, "real", "out", "environment", ("scrn", "plot", "com")),
        Field("pdynmc", 0.0, "real", "out", "environment", ("scrn", "plot")),
        Field("tempk", 0.0, "real", "out", "environment"),
        Field("mfreeze_evrn", 0, "int", "save", "environment"),
        Field("pdynmcf", 0.0, "real", "save", "environment"),
        Field("vmachf", 0.0, "real", "save", "environment"),
        Field("GRAVG", (0.0, 0.0, 0.0), "vec", "out", "environment"),
        Field("grav", 0.0, "real", "out", "environment"),
        Field("dvae", 0.0, "real", "data", "environment"),
        Field("dvael", 0.0, "real", "data", "environment"),
        Field("waltl", 0.0, "real", "data", "environment"),
        Field("dvaeh", 0.0, "real", "data", "environment"),
        Field("walth", 0.0, "real", "data", "environment"),
        Field("vaed3", 0.0, "real", "data", "environment"),
        Field("psiwdx", 0.0, "real", "data", "environment"),
        Field("twind", 0.1, "real", "data", "environment"),
        Field("VAEDS", (0.0, 0.0, 0.0), "vec", "state", "environment"),
        Field("VAEDSD", (0.0, 0.0, 0.0), "vec", "state", "environment"),
        Field("VAED", (0.0, 0.0, 0.0), "vec", "out", "environment"),
        Field("dvba", 0.0, "real", "out", "environment"),
        Field("markov_value", 0.0, "real", "save", "environment"),
        Field("turb_length", 0.0, "real", "data", "environment"),
        Field("turb_sigma", 0.0, "real", "data", "environment"),
        Field("taux1", 0.0, "real", "state", "environment"),
        Field("taux1d", 0.0, "real", "state", "environment"),
        Field("taux2", 0.0, "real", "state", "environment"),
        Field("taux2d", 0.0, "real", "state", "environment"),
        Field("tau", 0.0, "real", "diag", "environment"),
        Field("gauss_value", 0.0, "real", "diag", "environment"),
        Field("tempc", 0.0, "real", "diag", "environment"),
    )

    def __init__(self, weather_deck=None):
        self.weather_deck = weather_deck

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        store.set("dvba", store.get("dvbe"))

    def _environment_dryden(self, vehicle, dvba, int_step):
        store = vehicle.store
        turb_length = store.get("turb_length")
        turb_sigma = store.get("turb_sigma")
        tbd = store.get("TBD")
        alppx = store.get("alppx")
        phipx = store.get("phipx")
        markov_value = store.get("markov_value")
        taux1 = store.get("taux1")
        taux1d = store.get("taux1d")
        taux2 = store.get("taux2")
        taux2d = store.get("taux2d")
        prepare_for_dryden(markov_count=ROCKET6_MARKOV_COUNT)
        gauss_value = dryden_white(int_step)
        taux1d_new = taux2
        taux1 = integrate(taux1d_new, taux1d, taux1, int_step)
        taux1d = taux1d_new
        vl = dvba / turb_length
        taux2d_new = -vl * vl * taux1 - 2.0 * vl * taux2 + vl * vl * gauss_value
        taux2 = integrate(taux2d_new, taux2d, taux2, int_step)
        taux2d = taux2d_new
        tau = turb_sigma * math.sqrt(1.0 / (vl * PI)) * (
            taux1 + math.sqrt(3.0) * taux2 / vl
        )
        vtab = np.array(
            [
                -tau * math.sin(alppx * RAD),
                tau * math.sin(phipx * RAD) * math.cos(alppx * RAD),
                tau * math.cos(phipx * RAD) * math.cos(alppx * RAD),
            ],
            dtype=float,
        )
        vtad = tbd.T @ vtab
        store.set("markov_value", markov_value)
        store.set("taux1", taux1)
        store.set("taux1d", taux1d)
        store.set("taux2", taux2)
        store.set("taux2d", taux2d)
        store.set("tau", tau)
        store.set("gauss_value", gauss_value)
        return vtad

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mair = store.get("mair")
        matmo = mair // 100
        mturb = (mair - matmo * 100) // 10
        mwind = (mair - matmo * 100) % 10
        mair0 = matmo == 0 and mturb == 0 and mwind == 0
        mair12 = matmo == 0 and mturb == 1 and mwind == 2
        if not mair0 and not mair12:
            raise ValueError(f"unknown mair {mair}")
        if mair12 and self.weather_deck is None:
            raise ValueError("mair 12 requires a weather Datadeck")

        warning_flag = store.get("warning_flag")
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
        tempc = tempk - 273.16
        vsound = math.sqrt(1.4 * R * tempk)

        vmach = abs(dvba / vsound)
        pdynmc = 0.5 * rho * dvba * dvba

        vaed = np.zeros(3)
        if mwind > 0:
            int_step = ctx.int_step
            twind = store.get("twind")
            vaed3 = store.get("vaed3")
            dvw = self.weather_deck.look_up("speed", alt)
            psiwdx = self.weather_deck.look_up("direction", alt)
            vaed_raw = np.array(
                [
                    -dvw * math.cos(psiwdx * RAD),
                    -dvw * math.sin(psiwdx * RAD),
                    vaed3,
                ],
                dtype=float,
            )
            vaedsd_new = (vaed_raw - vaeds) * (1.0 / twind)
            vaeds = integrate(vaedsd_new, vaedsd, vaeds, int_step)
            vaedsd = vaedsd_new
            vaed = vaeds
        if mturb == 1:
            int_step = ctx.int_step
            vtad = self._environment_dryden(vehicle, dvba, int_step)
            vaed = vtad + vaeds

        vbad = vbed - vaed
        dvba = float(np.linalg.norm(vbad))
        vmach = abs(dvba / vsound)
        pdynmc = 0.5 * rho * dvba * dvba

        if "trcode" in store and "mguid" in store:
            if store.get("mguid") == 6:
                trcode = store.get("trcode")
                if vmach <= store.get("trmach"):
                    trcode = 2.0
                if pdynmc <= store.get("trdynm"):
                    trcode = 3.0
                store.set("trcode", trcode)

        if "mfreeze" in store:
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

        store.set("warning_flag", warning_flag)
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
        store.set("tempc", tempc)


class Round6Kinematics(ModuleBase):
    """Zipfel 6-DOF round Earth DCM kinematics (CADAC ``round6_kinematics``)."""

    name = "kinematics"
    fields = (
        Field("time", 0.0, "real", "exec", "kinematics", ("scrn", "plot", "com")),
        Field("event_time", 0.0, "real", "exec", "kinematics"),
        Field("int_step_new", 0.0, "real", "data", "kinematics"),
        Field("out_step_fact", 0.0, "real", "data", "kinematics"),
        Field("TBD", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "out", "kinematics"),
        Field("TBI", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "state", "kinematics"),
        Field("TBID", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "state", "kinematics"),
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
    )

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
        tbi = cadac_matmul(tbd, tdi)
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

        tbid_new = cadac_matmul(-skew(wbib), tbi)
        tbi = integrate(tbid_new, tbid, tbi, int_step)
        tbid = tbid_new

        unit = np.eye(3)
        ee = unit - cadac_matmul(tbi, tbi.T.copy())
        tbi = tbi + cadac_matmul(ee, tbi) * 0.5

        e1 = ee[0, 0]
        e2 = ee[1, 1]
        e3 = ee[2, 2]
        ortho_error = math.sqrt(e1 * e1 + e2 * e2 + e3 * e3)

        tdi = cad_tdi84(lonx * RAD, latx * RAD, alt, time)
        tbd = cadac_matmul(tbi, tdi.T.copy())
        tbd13 = tbd[0, 2]
        tbd11 = tbd[0, 0]
        tbd33 = tbd[2, 2]
        tbd12 = tbd[0, 1]
        tbd23 = tbd[1, 2]

        # C++ `if(fabs(tbd13)<1)`. Numpy TBI·TDIᵀ at vertical launch often
        # yields |tbd13| a few ulps below 1; that asin path divides by ~1e-8
        # cosine and corrupts roll/yaw (RCS Schmitt). Treat 1-1e-14 as |tbd13|>=1.
        if math.fabs(tbd13) < 1.0 - 1e-14:
            thtbd = math.asin(-tbd13)
            cthtbd = math.cos(thtbd)
        else:
            thtbd = PI / 2.0 * cadac_sign(-tbd13)
            cthtbd = EPS
        cpsi = tbd11 / cthtbd
        if math.fabs(cpsi) > 1.0:
            cpsi = 1.0 * cadac_sign(cpsi)
        cphi = tbd33 / cthtbd
        if math.fabs(cphi) > 1.0:
            cphi = 1.0 * cadac_sign(cphi)
        psibd = math.acos(cpsi) * cadac_sign(tbd12)
        phibd = math.acos(cphi) * cadac_sign(tbd23)
        psibdx = DEG * psibd
        thtbdx = DEG * thtbd
        phibdx = DEG * phibd

        vbab = cadac_matmul(tbd, vbed - vaed)
        vbab1 = vbab[0]
        vbab2 = vbab[1]
        vbab3 = vbab[2]
        alpha = math.atan2(vbab3, vbab1)
        beta = math.asin(vbab2 / dvba)
        alphax = alpha * DEG
        betax = beta * DEG

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
        alppx = alpp * DEG
        phipx = phip * DEG

        vbib = cadac_matmul(tbi, vbii)
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


class Round6Euler(ModuleBase):
    """Zipfel 6-DOF round Earth rigid-body Euler (CADAC ``round6_euler``)."""

    name = "euler"
    fields = (
        Field("ppx", 0.0, "real", "out", "euler", ("plot",)),
        Field("qqx", 0.0, "real", "out", "euler", ("plot",)),
        Field("rrx", 0.0, "real", "out", "euler", ("plot",)),
        Field("WBEB", (0.0, 0.0, 0.0), "vec", "diag", "euler"),
        Field("WBIB", (0.0, 0.0, 0.0), "vec", "state", "euler"),
        Field("WBIBD", (0.0, 0.0, 0.0), "vec", "state", "euler"),
        Field("WBII", (0.0, 0.0, 0.0), "vec", "out", "euler"),
    )

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        ppx = store.get("ppx")
        qqx = store.get("qqx")
        rrx = store.get("rrx")
        tbi = store.get("TBI")
        wbeb = np.array([ppx * RAD, qqx * RAD, rrx * RAD], dtype=float)
        weii = np.array([0.0, 0.0, WEII3], dtype=float)
        wbib = wbeb + cadac_matmul(tbi, weii)
        store.set("WBIB", wbib)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        fmb = store.get("FMB")
        tbi = store.get("TBI")
        ibbb = store.get("IBBB")
        wbib = store.get("WBIB")
        wbibd = store.get("WBIBD")
        int_step = ctx.int_step
        # cadac_inverse (adjoint/det) matches C++ Matrix::inverse(); np.linalg.inv
        # is LAPACK and 1 ulp off (UPDATES 0.168.12). Do not replace with numpy.
        wacc_next = cadac_matmul(
            cadac_inverse(ibbb),
            fmb - cadac_matmul(cadac_matmul(skew(wbib), ibbb), wbib),
        )
        wbib = integrate(wacc_next, wbibd, wbib, int_step)
        wbibd = wacc_next
        wbii = cadac_matmul(tbi.T.copy(), wbib)
        weii = np.array([0.0, 0.0, WEII3], dtype=float)
        wbeb = wbib - cadac_matmul(tbi, weii)
        store.set("WBIB", wbib)
        store.set("WBIBD", wbibd)
        store.set("ppx", wbeb[0] * DEG)
        store.set("qqx", wbeb[1] * DEG)
        store.set("rrx", wbeb[2] * DEG)
        store.set("WBEB", wbeb)
        store.set("WBII", wbii)


class Round6Newton(ModuleBase):
    """Zipfel 6-DOF round Earth translational Newton (CADAC ``round6_newton``)."""

    name = "newton"
    fields = (
        Field("minit", 0, "int", "data", "newton"),
        Field("alpha0x", 0.0, "real", "data", "newton"),
        Field("beta0x", 0.0, "real", "data", "newton"),
        Field("lonx", 0.0, "real", "init/diag", "newton", ("scrn", "plot", "com")),
        Field("latx", 0.0, "real", "init/diag", "newton", ("scrn", "plot", "com")),
        Field("alt", 0.0, "real", "init/out", "newton", ("scrn", "plot", "com")),
        Field("TVD", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "out", "newton"),
        Field("TDI", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "init", "newton"),
        Field("dvbe", 0.0, "real", "init/out", "newton", ("scrn", "plot", "com")),
        Field("dvbi", 0.0, "real", "out", "newton", ("scrn", "plot", "com")),
        Field("WEII", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "init", "newton"),
        Field("psivdx", 0.0, "real", "init/out", "newton", ("scrn", "plot", "com")),
        Field("thtvdx", 0.0, "real", "init/out", "newton", ("scrn", "plot", "com")),
        Field("dbi", 0.0, "real", "out", "newton", ("scrn", "plot")),
        Field("TGI", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "init", "newton"),
        Field("VBED", (0.0, 0.0, 0.0), "vec", "out", "newton"),
        Field("altx", 0.0, "real", "diag", "newton"),
        Field("SBII", (0.0, 0.0, 0.0), "vec", "state", "newton", ("com",)),
        Field("VBII", (0.0, 0.0, 0.0), "vec", "state", "newton", ("com",)),
        Field("ABII", (0.0, 0.0, 0.0), "vec", "save", "newton"),
        Field("grndtrck", 0.0, "real", "diag", "newton"),
        Field("FSPB", (0.0, 0.0, 0.0), "vec", "out", "newton"),
        Field("ayx", 0.0, "real", "diag", "newton", ("plot",)),
        Field("anx", 0.0, "real", "diag", "newton", ("plot",)),
        Field("gndtrkmx", 0.0, "real", "diag", "newton"),
        Field("gndtrnmx", 0.0, "real", "diag", "newton"),
        Field("latx_bias", 0.0, "real", "data", "newton"),
        Field("dvbi_bias", 0.0, "real", "data", "newton"),
        Field("dbi_bias", 0.0, "real", "data", "newton"),
        Field("mfreeze_newt", 0, "int", "save", "newton"),
        Field("dvbef", 0.0, "real", "save", "newton"),
        Field("thtvdx_bias", 0.0, "real", "data", "newton"),
        Field("sat_semi", 0.0, "real", "data", "newton"),
        Field("sat_ecc", 0.0, "real", "data", "newton"),
        Field("sat_inclx", 0.0, "real", "data", "newton"),
        Field("sat_lon_anodex", 0.0, "real", "data", "newton"),
        Field("sat_arg_perix", 0.0, "real", "data", "newton"),
        Field("sat_true_anomx", 0.0, "real", "data", "newton"),
        Field("ranglex_l_t", 0.0, "real", "data", "newton"),
        Field("headon_flag", 0, "int", "data", "newton"),
        Field("tgo_insertion", 0.0, "real", "data", "newton"),
    )

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        minit = store.get("minit")
        if minit != 0:
            raise ValueError(f"unknown minit {minit}")
        dvbe = store.get("dvbe")
        lonx = store.get("lonx")
        latx = store.get("latx")
        alt = store.get("alt")
        time = store.get("time")
        psibdx = store.get("psibdx")
        thtbdx = store.get("thtbdx")
        phibdx = store.get("phibdx")
        alpha0x = store.get("alpha0x")
        beta0x = store.get("beta0x")

        weii = np.zeros((3, 3))
        weii[0, 1] = -WEII3
        weii[1, 0] = WEII3

        sbii = cad_in_geo84(lonx * RAD, latx * RAD, alt, time)
        dbi = float(np.linalg.norm(sbii))

        salp = math.sin(alpha0x * RAD)
        calp = math.cos(alpha0x * RAD)
        sbet = math.sin(beta0x * RAD)
        cbet = math.cos(beta0x * RAD)
        vbeb = np.array(
            [calp * cbet * dvbe, sbet * dvbe, salp * cbet * dvbe], dtype=float
        )
        tbd = mat3tr(psibdx * RAD, thtbdx * RAD, phibdx * RAD)
        vbed = cadac_matmul(tbd.T.copy(), vbeb)

        tdi = cad_tdi84(lonx * RAD, latx * RAD, alt, time)
        tgi = cad_tgi84(lonx * RAD, latx * RAD, alt, time)
        vbii = cadac_matmul(tdi.T.copy(), vbed) + cadac_matmul(weii, sbii)
        dvbi = float(np.linalg.norm(vbii))

        polar = polar_from_cart(vbed)
        psivdx = DEG * float(polar[1])
        thtvdx = DEG * float(polar[2])

        store.set("lonx", lonx)
        store.set("latx", latx)
        store.set("TDI", tdi)
        store.set("dvbi", dvbi)
        store.set("WEII", weii)
        store.set("psivdx", psivdx)
        store.set("thtvdx", thtvdx)
        store.set("dbi", dbi)
        store.set("TGI", tgi)
        store.set("VBED", vbed)
        store.set("SBII", sbii)
        store.set("VBII", vbii)
        store.set("psibdx", psibdx)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        tdi = store.get("TDI")
        tgi = store.get("TGI")
        weii = store.get("WEII")
        grndtrck = store.get("grndtrck")
        mfreeze_newt = store.get("mfreeze_newt")
        dvbef = store.get("dvbef")
        sbii = store.get("SBII")
        vbii = store.get("VBII")
        abii = store.get("ABII")
        time = store.get("time")
        gravg = store.get("GRAVG")
        tbi = store.get("TBI")
        fapb = store.get("FAPB")
        vmass = store.get("vmass")
        int_step = ctx.int_step

        fspb = fapb * (1.0 / vmass)
        next_acc = cadac_matmul(tbi.T.copy(), fspb) + cadac_matmul(tgi.T.copy(), gravg)
        next_vel = integrate(next_acc, abii, vbii, int_step)
        sbii = integrate(next_vel, vbii, sbii, int_step)
        abii = next_acc
        vbii = next_vel
        dvbi = float(np.linalg.norm(vbii))
        dbi = float(np.linalg.norm(sbii))

        lon, lat, alt = cad_geo84_in(sbii, time)
        tdi = cad_tdi84(lon, lat, alt, time)
        tgi = cad_tgi84(lon, lat, alt, time)
        lonx = lon * DEG
        latx = lat * DEG
        altx = 0.001 * alt * FOOT

        vbed = cadac_matmul(tdi, vbii - cadac_matmul(weii, sbii))
        polar = polar_from_cart(vbed)
        dvbe = float(polar[0])
        psivdx = DEG * float(polar[1])
        thtvdx = DEG * float(polar[2])
        tvd = mat2tr(psivdx * RAD, thtvdx * RAD)

        ayx = fspb[1] / AGRAV
        anx = -fspb[2] / AGRAV
        grndtrck = (
            grndtrck
            + math.sqrt(vbed[0] * vbed[0] + vbed[1] * vbed[1]) * int_step * REARTH / dbi
        )
        gndtrkmx = 0.001 * grndtrck
        gndtrnmx = NMILES * grndtrck

        if "mfreeze" in store:
            mfreeze = store.get("mfreeze")
            if mfreeze == 0:
                mfreeze_newt = 0
            else:
                if mfreeze != mfreeze_newt:
                    mfreeze_newt = mfreeze
                    dvbef = dvbe
                dvbe = dvbef

        store.set("SBII", sbii)
        store.set("VBII", vbii)
        store.set("ABII", abii)
        store.set("grndtrck", grndtrck)
        store.set("mfreeze_newt", mfreeze_newt)
        store.set("dvbef", dvbef)
        store.set("lonx", lonx)
        store.set("latx", latx)
        store.set("alt", alt)
        store.set("TVD", tvd)
        store.set("TDI", tdi)
        store.set("dvbe", dvbe)
        store.set("dvbi", dvbi)
        store.set("TGI", tgi)
        store.set("VBED", vbed)
        store.set("FSPB", fspb)
        store.set("psivdx", psivdx)
        store.set("thtvdx", thtvdx)
        store.set("dbi", dbi)
        store.set("altx", altx)
        store.set("ayx", ayx)
        store.set("anx", anx)
        store.set("gndtrkmx", gndtrkmx)
        store.set("gndtrnmx", gndtrnmx)
