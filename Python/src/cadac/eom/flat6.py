import math

import numpy as np

from cadac.constants import DEG, EPS, PI, RAD, R
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.kernel.integrate import integrate
from cadac.kernel.module import ModuleBase
from cadac.kernel.state import Field
from cadac.math.frames import (
    cadac_sign,
    hypot3,
    incidence_angles,
    mat2tr,
    mat3tr,
    quat_to_dcm,
    skew,
)
from cadac.stoch import dryden_white


class Flat6Environment(ModuleBase):
    """Zipfel 6-DOF flat Earth atmosphere and gravity (CADAC ``flat6_environment``).

    C++ FALCON6 uses ``mwind`` 0/1/2. Fortran FALCON6 packs
    ``mair=|MTURB|MWIND|MATMO|``; ``mair!=0`` selects that decode. Dryden
    (``MTURB=1``) follows MODULE.FOR ``G2TURB`` (not AGM6's body shortcut).
    Tabular ``MATMO=3`` / ``MWIND=3`` use a WEATHER Datadeck (MODULE.FOR G2).
    """

    name = "environment"
    fields = (
        Field("mwind", 0, "int", "data", "environment"),
        Field("mair", 0, "int", "data", "environment"),
        Field("press", 0.0, "real", "out", "environment"),
        Field("rho", 0.0, "real", "out", "environment"),
        Field("vsound", 0.0, "real", "diag", "environment"),
        Field("grav", 0.0, "real", "out", "environment"),
        Field("vmach", 0.0, "real", "out", "environment", ("scrn", "plot", "com")),
        Field("pdynmc", 0.0, "real", "out", "environment", ("scrn", "plot")),
        Field("tempk", 0.0, "real", "out", "environment"),
        Field("mfreeze_environ", 0, "int", "save", "environment"),
        Field("pdynmcf", 0.0, "real", "save", "environment"),
        Field("vmachf", 0.0, "real", "save", "environment"),
        Field("dvae", 0.0, "real", "data", "environment"),
        Field("dvael", 0.0, "real", "data", "environment"),
        Field("waltl", 0.0, "real", "data", "environment"),
        Field("dvaeh", 0.0, "real", "data", "environment"),
        Field("walth", 0.0, "real", "data", "environment"),
        Field("vaed3", 0.0, "real", "data", "environment"),
        Field("psiwdx", 0.0, "real", "data", "environment"),
        Field("twind", 0.1, "real", "data", "environment"),
        Field("VAELS", (0.0, 0.0, 0.0), "vec", "state", "environment"),
        Field("VAELSD", (0.0, 0.0, 0.0), "vec", "state", "environment"),
        Field("VAEL", (0.0, 0.0, 0.0), "vec", "out", "environment"),
        Field("dvba", 0.0, "real", "out", "environment", ("plot",)),
        Field("VBAL", (0.0, 0.0, 0.0), "vec", "out", "environment"),
        # Fortran G2 / G2TURB (HEAD TURBL/TURBSIG → turb_length/turb_sigma)
        Field("turb_length", 0.0, "real", "data", "environment"),
        Field("turb_sigma", 0.0, "real", "data", "environment"),
        Field("taux1", 0.0, "real", "state", "environment"),
        Field("taux1d", 0.0, "real", "state", "environment"),
        Field("taux2", 0.0, "real", "state", "environment"),
        Field("taux2d", 0.0, "real", "state", "environment"),
        Field("tau", 0.0, "real", "diag", "environment"),
        Field("gauss_value", 0.0, "real", "diag", "environment"),
        Field("VTAG", (0.0, 0.0, 0.0), "vec", "diag", "environment"),
    )

    def __init__(self, weather_deck=None) -> None:
        self.weather_deck = weather_deck

    def _g2turb(self, store, dvba, int_step):
        """FALCON6 Fortran G2TURB — Dryden filter + aeroballistic TAB → geographic VTAG."""
        turb_length = store.get("turb_length")
        turb_sigma = store.get("turb_sigma")
        tbl = np.asarray(store.get("TBL"), dtype=float)
        alpp = store.get("alpp")
        phip = store.get("phip")
        taux1 = store.get("taux1")
        taux1d = store.get("taux1d")
        taux2 = store.get("taux2")
        taux2d = store.get("taux2d")

        gauss_value = dryden_white(int_step)
        # Integrate filter states (CADAC module form of G2TURB ODEs).
        taux1d_new = taux2
        taux1 = integrate(taux1d_new, taux1d, taux1, int_step)
        taux1d = taux1d_new
        vl = dvba / turb_length
        taux2d_new = -vl * vl * taux1 - 2.0 * vl * taux2 + vl * vl * gauss_value
        taux2 = integrate(taux2d_new, taux2d, taux2, int_step)
        taux2d = taux2d_new
        # Fortran: DUM1=SQRT(1/(PI*VL)); DUM2=(1/VL)*SQRT(3/(PI*VL))
        dum1 = math.sqrt(1.0 / (PI * vl))
        dum2 = (1.0 / vl) * math.sqrt(3.0 / (PI * vl))
        tau = turb_sigma * (dum1 * taux1 + dum2 * taux2)

        # VTAA=[0,0,TAU]; TAB from ALPP/PHIP (rad); TAG=TAB*TBL; VTAG=TGA*VTAA
        cosa, sina = math.cos(alpp), math.sin(alpp)
        cosp, sinp = math.cos(phip), math.sin(phip)
        tab = np.array(
            [
                [cosa, sina * sinp, sina * cosp],
                [0.0, cosp, -sinp],
                [-sina, cosa * sinp, cosa * cosp],
            ],
            dtype=float,
        )
        vtaa = np.array([0.0, 0.0, tau], dtype=float)
        tag = tab @ tbl
        vtag = tag.T @ vtaa

        store.set("taux1", taux1)
        store.set("taux1d", taux1d)
        store.set("taux2", taux2)
        store.set("taux2d", taux2d)
        store.set("tau", tau)
        store.set("gauss_value", gauss_value)
        store.set("VTAG", vtag)
        return vtag

    def execute(self, vehicle, ctx) -> None:
        store = vehicle.store
        # Fortran G2: MAIR=|MTURB|MWIND|MATMO|. mair==0 → C++ mwind field path.
        mair = store.get("mair")
        mturb = 0
        matmo = 0
        if mair != 0:
            mturb = int(mair / 100)
            mwind = int((mair - mturb * 100) / 10)
            matmo = mair - mturb * 100 - mwind * 10
            if matmo not in (0, 3) or mturb not in (0, 1) or mwind not in (0, 1, 2, 3):
                raise ValueError(f"unknown mair {mair}")
            if (matmo == 3 or mwind == 3) and self.weather_deck is None:
                raise ValueError("mair tabular MATMO/MWIND=3 requires a weather Datadeck")
        else:
            mwind = store.get("mwind")
            if mwind not in (0, 1, 2):
                raise ValueError(f"unknown mwind {mwind}")

        hbe = store.get("hbe")
        vbel = store.get("VBEL")
        if matmo == 3:
            # Fortran G2 MATMO=3: WEATHER RHX/CTMP/WPRES vs WALT.
            rho = self.weather_deck.look_up("density", hbe)
            press = self.weather_deck.look_up("pressure", hbe)
            tempc = self.weather_deck.look_up("temperature", hbe)
            tempk = tempc + 273.16
            vsound = math.sqrt(1.4 * R * tempk)
        else:
            rho, press, tempk = atmosphere76(hbe)
            vsound = math.sqrt(1.4 * R * tempk)
        vaels = np.array(store.get("VAELS"), dtype=float, copy=True)
        vaelsd = np.array(store.get("VAELSD"), dtype=float, copy=True)
        vael = np.zeros(3)
        if mwind > 0:
            if mwind == 1:
                dvw = store.get("dvae")
                psiwdx = store.get("psiwdx")
            elif mwind == 3:
                # Fortran G2 MWIND=3: WEATHER WVEL/WDIR vs WALT.
                dvw = self.weather_deck.look_up("speed", hbe)
                psiwdx = self.weather_deck.look_up("direction", hbe)
            else:
                # mwind==2 shear (C++ / Fortran)
                dvael = store.get("dvael")
                waltl = store.get("waltl")
                dvaeh = store.get("dvaeh")
                walth = store.get("walth")
                dvw = dvael + (dvaeh - dvael) * (hbe - waltl) / (walth - waltl)
                if hbe < waltl:
                    dvw = 0.0
                if hbe > walth:
                    dvw = 0.0
                psiwdx = store.get("psiwdx")
            vael_raw = np.array(
                [
                    -dvw * math.cos(psiwdx * RAD),
                    -dvw * math.sin(psiwdx * RAD),
                    store.get("vaed3"),
                ],
                dtype=float,
            )
            twind = store.get("twind")
            vaelsd_new = (vael_raw - vaels) * (1.0 / twind)
            vaels = integrate(vaelsd_new, vaelsd, vaels, ctx.int_step)
            vaelsd = vaelsd_new
            vael = np.array(vaels, dtype=float, copy=True)
            store.set("VAELS", vaels)
            store.set("VAELSD", vaelsd)

        # Fortran G2: MTURB=1 → G2TURB(VTAG, previous DVBA); VAEL = VTAG + VAELS
        if mturb == 1:
            dvba_prev = store.get("dvba")
            vtag = self._g2turb(store, dvba_prev, ctx.int_step)
            vael = vtag + vaels

        vbal = vbel - vael
        dvba = hypot3(vbal)
        vmach = abs(dvba / vsound)
        pdynmc = 0.5 * rho * dvba**2

        # FALCON6 termination: mguid==6 → low Mach / dyn. press. (plane trcode)
        if "trcode" in store and "mguid" in store and store.get("mguid") == 6:
            trcode = store.get("trcode")
            if vmach <= store.get("trmach"):
                trcode = 2.0
            if pdynmc <= store.get("trdynm"):
                trcode = 3.0
            store.set("trcode", trcode)

        # FALCON6 autopilot freeze: latch vmach/pdynmc when plane mfreeze present
        if "mfreeze" in store:
            mfreeze = store.get("mfreeze")
            mfreeze_environ = store.get("mfreeze_environ")
            pdynmcf = store.get("pdynmcf")
            vmachf = store.get("vmachf")
            if mfreeze == 0:
                mfreeze_environ = 0
            else:
                if mfreeze != mfreeze_environ:
                    mfreeze_environ = mfreeze
                    vmachf = vmach
                    pdynmcf = pdynmc
                vmach = vmachf
                pdynmc = pdynmcf
            store.set("mfreeze_environ", mfreeze_environ)
            store.set("pdynmcf", pdynmcf)
            store.set("vmachf", vmachf)

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


class Flat6Kinematics(ModuleBase):
    """Zipfel 6-DOF flat Earth quaternion kinematics (CADAC ``flat6_kinematics``)."""

    name = "kinematics"
    fields = (
        Field("ck", 50.0, "real", "data", "kinematics"),
        Field("q0d", 0.0, "real", "state", "kinematics"),
        Field("q0", 0.0, "real", "state", "kinematics"),
        Field("q1d", 0.0, "real", "state", "kinematics"),
        Field("q1", 0.0, "real", "state", "kinematics"),
        Field("q2d", 0.0, "real", "state", "kinematics"),
        Field("q2", 0.0, "real", "state", "kinematics"),
        Field("q3d", 0.0, "real", "state", "kinematics"),
        Field("q3", 0.0, "real", "state", "kinematics"),
        Field("TBL", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "out", "kinematics"),
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
        Field("TLB", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "diag", "kinematics"),
    )

    def initialize(self, vehicle, ctx) -> None:
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

    def execute(self, vehicle, ctx) -> None:
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
        alpha, beta, alpp, phip = incidence_angles(vbab, dvba)
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


class Flat6Euler(ModuleBase):
    """Zipfel 6-DOF flat Earth rigid-body Euler (CADAC ``flat6_euler``)."""

    name = "euler"
    fields = (
        Field("ppx", 0.0, "real", "init/out", "euler", ("plot",)),
        Field("qqx", 0.0, "real", "init/out", "euler", ("plot",)),
        Field("rrx", 0.0, "real", "init/out", "euler", ("plot",)),
        Field("WBEB", (0.0, 0.0, 0.0), "vec", "state", "euler"),
        Field("WBEBD", (0.0, 0.0, 0.0), "vec", "state", "euler"),
    )

    def initialize(self, vehicle, ctx) -> None:
        store = vehicle.store
        ppx = store.get("ppx")
        qqx = store.get("qqx")
        rrx = store.get("rrx")
        store.set("WBEB", np.array([ppx * RAD, qqx * RAD, rrx * RAD], dtype=float))

    def execute(self, vehicle, ctx) -> None:
        store = vehicle.store
        fmb = store.get("FMB")
        ibbb = store.get("IBBB")
        eng_ang_mom = store.get("eng_ang_mom")
        wbeb = store.get("WBEB")
        wbebd = store.get("WBEBD")
        int_step = ctx.int_step
        l_engine = np.array([eng_ang_mom, 0.0, 0.0], dtype=float)
        # Flat6 goldens use np.linalg.inv; Round6/ROCKET6 use cadac_inverse
        # (UPDATES 0.168.12). Do not unify.
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


class Flat6Newton(ModuleBase):
    """Zipfel 6-DOF flat Earth translational Newton (CADAC ``flat6_newton``)."""

    name = "newton"
    fields = (
        Field("time", 0.0, "real", "exec", "newton", ("scrn", "plot", "com")),
        Field("halt", 0, "int", "exec", "newton"),
        Field("VBEBD", (0.0, 0.0, 0.0), "vec", "state", "newton"),
        Field("VBEB", (0.0, 0.0, 0.0), "vec", "state", "newton", ("plot",)),
        Field("SBELD", (0.0, 0.0, 0.0), "vec", "state", "newton"),
        Field("SBEL", (0.0, 0.0, 0.0), "vec", "state", "newton", ("plot", "com")),
        Field("sbel1", 0.0, "real", "data", "newton"),
        Field("sbel2", 0.0, "real", "data", "newton"),
        Field("sbel3", 0.0, "real", "data", "newton"),
        Field("SBELM", (0.0, 0.0, 0.0), "vec", "save", "newton"),
        Field("groundrange", 0.0, "real", "diag", "newton"),
        Field("FSPB", (0.0, 0.0, 0.0), "vec", "out", "newton"),
        Field("VBEL", (0.0, 0.0, 0.0), "vec", "out", "newton", ("com", "scrn", "plot")),
        Field("dvbe", 0.0, "real", "in/out", "newton", ("plot",)),
        Field("alpha0x", 0.0, "real", "data", "newton"),
        Field("beta0x", 0.0, "real", "data", "newton"),
        Field("hbe", 0.0, "real", "out", "newton", ("scrn", "plot")),
        Field("psivlx", 0.0, "real", "diag", "newton", ("scrn", "plot")),
        Field("thtvlx", 0.0, "real", "diag", "newton", ("scrn", "plot")),
        Field("alx", 0.0, "real", "diag", "newton", ("plot",)),
        Field("anx", 0.0, "real", "diag", "newton", ("scrn", "plot")),
        Field("ayx", 0.0, "real", "diag", "newton", ("plot",)),
        Field("ATB", (0.0, 0.0, 0.0), "vec", "diag", "newton"),
        Field("mfreeze_newt", 0, "int", "save", "newton"),
        Field("dvbef", 0.0, "real", "save", "newton"),
    )

    def initialize(self, vehicle, ctx) -> None:
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

    def execute(self, vehicle, ctx) -> None:
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
