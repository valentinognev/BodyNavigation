from math import acos, asin, cos, exp, hypot, log, sin, sqrt, tan

import numpy as np

from cadac.constants import DEG, EPS, RAD
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.earth import cadine
from cadac.math.frames import cadtbv, mat2tr, polar_from_cart, skew

# C++ cruise5 modes kept verbatim (Tasks 21+); Fortran MGUIDP=1/2 are additive.
_CPP_MGUIDANCE = frozenset({3, 6, 30, 33, 40, 43, 60, 66, 70})
_TERRAIN_STACK = 100


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _angle(vec1, vec2):
    scalar = float(vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2])
    abs1 = sqrt(float(vec1[0] ** 2 + vec1[1] ** 2 + vec1[2] ** 2))
    abs2 = sqrt(float(vec2[0] ** 2 + vec2[1] ** 2 + vec2[2] ** 2))
    dum = abs1 * abs2
    if dum > EPS:
        argument = scalar / dum
    else:
        argument = 1.0
    if argument > 1.0:
        argument = 1.0
    if argument < -1.0:
        argument = -1.0
    return acos(argument)


def _ran_ms(iseed):
    """Microsoft FORTRAN RAN(ISEED): uniform (0,1) and updated seed."""
    iseed = int(iseed) * 65539
    if iseed < 0:
        iseed = (iseed + 2147483647) + 1
    iseed = ((iseed + 2**31) % 2**32) - 2**31
    if iseed < 0:
        iseed = (iseed + 2147483647) + 1
    return float(iseed) * 0.4656613e-9, iseed


def _nint(value):
    """Fortran NINT — nearest integer, halves away from zero."""
    if value >= 0:
        return int(value + 0.5)
    return int(value - 0.5)


class Cruise5Guidance:
    name = "guidance"

    def __init__(self):
        # Fortran COMMON/HOI/HO(100), /HI/H(100) — 1-based cells
        self._ho = [0.0] * (_TERRAIN_STACK + 1)
        self._h = [0.0] * (_TERRAIN_STACK + 1)
        self._iset_c1 = 1
        self._ncell = 0
        self._sig2 = 0.0
        self._iset_d1 = 1
        self._tstart = 0.0
        self._sbel_horiz_prev = None
        self._terrain_planted = False
        self._nl = 1  # Fortran C1 local NL (persists across calls)

    def define(self, vehicle):
        store = vehicle.store
        plot = ("scrn", "plot")
        for field in (
            Field("mguidance", 0, "int", "data", "guidance", ("scrn",)),
            Field("pronav_gain", 0.0, "real", "data", "guidance"),
            Field("line_gain", 0.0, "real", "data", "guidance"),
            Field("nl_gain_fact", 1.0, "real", "data", "guidance"),
            Field("decrement", 0.0, "real", "data", "guidance"),
            Field("wp_lonx", 0.0, "real", "data", "guidance"),
            Field("wp_latx", 0.0, "real", "data", "guidance"),
            Field("wp_alt", 0.0, "real", "data", "guidance"),
            Field("psifgx", 0.0, "real", "data", "guidance"),
            Field("thtfgx", 0.0, "real", "data", "guidance"),
            Field("point_gain", 0.0, "real", "data", "guidance"),
            Field("wp_sltrange", 999999.0, "real", "diag", "guidance"),
            Field("nl_gain", 0.0, "real", "diag", "guidance"),
            Field("VBEO", (0.0, 0.0, 0.0), "vec", "diag", "guidance"),
            Field("VBEF", (0.0, 0.0, 0.0), "vec", "diag", "guidance"),
            Field("wp_grdrange", 999999.0, "real", "diag", "guidance", plot),
            Field("SWBG", (0.0, 0.0, 0.0), "vec", "out", "guidance"),
            Field("rad_min", 0.0, "real", "diag", "guidance"),
            Field("rad_geometric", 0.0, "real", "diag", "guidance"),
            Field("wp_flag", 0, "int", "diag", "guidance", ("plot",)),
            # Fortran C1 / D1TER TF/OA (MODULE.FOR)
            Field("dcell", 100.0, "real", "data", "guidance"),
            Field("rahead", 0.0, "real", "data", "guidance"),
            Field("dhtrc", 0.0, "real", "data", "guidance"),
            Field("slope", 0.0, "real", "data", "guidance"),
            Field("occden", 0.0, "real", "data", "guidance"),
            Field("sigobs", 0.0, "real", "data", "guidance"),
            Field("iseed2", 12345, "int", "data", "guidance"),
            Field("tlead", 0.0, "real", "data", "guidance"),
            Field("gelev", 0.0, "real", "data", "guidance"),
            Field("racq", 0.0, "real", "data", "guidance"),
            Field("hgmean", 0.0, "real", "data", "guidance"),
            Field("hgbias", 0.0, "real", "data", "guidance"),
            Field("facth", 0.444, "real", "data", "guidance"),
            Field("rcor", 609.6, "real", "data", "guidance"),
            Field("sigmah", 30.48, "real", "data", "guidance"),
            Field("iseed", 12345, "int", "data", "guidance"),
            Field("j_ter", 0, "int", "diag", "guidance"),
            Field("gndpt", 0.0, "real", "diag", "guidance"),
            Field("gndtck", 0.0, "real", "diag", "guidance"),
            Field("gndocc", 0.0, "real", "diag", "guidance"),
            Field("hge", 0.0, "real", "out", "guidance", plot),
            Field("hges", 0.0, "real", "diag", "guidance", plot),
            Field("hbe", 0.0, "real", "diag", "guidance"),
            Field("hbg", 0.0, "real", "out", "guidance", plot),
            Field("hbgs", 0.0, "real", "out", "guidance", plot),
            Field("elmax", 0.0, "real", "diag", "guidance", plot),
            Field("hclear", 0.0, "real", "diag", "guidance"),
            Field("dhobst", 0.0, "real", "diag", "guidance", plot),
            Field("clmean", 0.0, "real", "diag", "guidance"),
            Field("clsigm", 0.0, "real", "diag", "guidance"),
            Field("tfoa", 0.0, "real", "diag", "guidance", plot),
            Field("clobn", 0.0, "real", "diag", "guidance"),
            Field("thtvl", 0.0, "real", "diag", "guidance"),
            Field("hr", 0.0, "real", "state", "guidance"),
            Field("hrd", 0.0, "real", "state", "guidance"),
            Field("hg", 0.0, "real", "state", "guidance"),
            Field("hgd", 0.0, "real", "state", "guidance"),
            Field("hgslop", 0.0, "real", "diag", "guidance"),
            Field("mguidp", 0, "int", "diag", "guidance"),
            Field("mguidl", 0, "int", "diag", "guidance"),
            Field("mroll", 0, "int", "diag", "guidance"),
            Field("maut", 0, "int", "data", "control"),
            # Fortran C1 bank-to-turn roll command (MODULE.FOR MGUIDL=3)
            Field("ggp", 3.0, "real", "data", "guidance"),
            Field("bgp", 1.0, "real", "data", "guidance"),
            Field("aldead", 0.01740, "real", "data", "guidance"),
            Field("phibvc", 0.0, "real", "out", "guidance", plot),
        ):
            if field.name not in store:
                store.define(field)

    def initialize(self, vehicle, ctx):
        """Fortran C1I — reset terrain stack pointers."""
        store = vehicle.store
        hge = store.get("hge")
        store.set("j_ter", 0)
        store.set("gndpt", 0.0)
        self._ho[1] = hge
        self._h[1] = hge
        store.set("hges", hge)
        self._iset_c1 = 1
        self._ncell = 0
        self._sig2 = 0.0
        self._iset_d1 = 1
        self._tstart = 0.0
        self._sbel_horiz_prev = None
        self._terrain_planted = False
        self._nl = 1

    def plant_terrain_stack(self, ho, h):
        """Unit-test hook: plant 1-based HO/H terrain databases."""
        n = max(len(ho), len(h)) - 1
        for i in range(1, min(n, _TERRAIN_STACK) + 1):
            if i < len(ho):
                self._ho[i] = float(ho[i])
            if i < len(h):
                self._h[i] = float(h[i])
        self._terrain_planted = True

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mguidance = store.get("mguidance")
        if mguidance == 0:
            return
        grav = store.get("grav")
        phicx = store.get("phicx")
        alcomx = 0.0
        ancomx = 0.0

        mroll = int(mguidance / 100)
        mguidl = int((mguidance - mroll * 100) / 10)
        mguidp = mguidance - mguidl * 10 - mroll * 100
        store.set("mroll", mroll)
        store.set("mguidl", mguidl)
        store.set("mguidp", mguidp)
        # Strip MROLL hundreds digit so C++ modes keep working with inverted attitude
        mode = mguidance - mroll * 100
        acv2 = None

        if mode in _CPP_MGUIDANCE:
            if mode == 30:
                algv = self.guidance_line(vehicle)
                alcomx = float(algv[1] / grav)
                acv2 = float(algv[1])
            elif mode == 3:
                algv = self.guidance_line(vehicle)
                alcomx = 0.0
                ancomx = float(-algv[2] / grav)
            elif mode == 33:
                algv = self.guidance_line(vehicle)
                alcomx = float(algv[1] / grav)
                ancomx = float(-algv[2] / grav)
                acv2 = float(algv[1])
            elif mode == 60:
                apnb = self.guidance_pronav(vehicle)
                alcomx = float(apnb[1] / grav)
                ancomx = 0.0
            elif mode == 6:
                apnb = self.guidance_pronav(vehicle)
                alcomx = 0.0
                ancomx = float(-apnb[2] / grav)
            elif mode == 66:
                apnb = self.guidance_pronav(vehicle)
                alcomx = float(apnb[1] / grav)
                ancomx = float(-apnb[2] / grav)
            elif mode == 43:
                algv = self.guidance_line(vehicle)
                apgv = self.guidance_point(vehicle)
                alcomx = float(apgv[1] / grav)
                ancomx = float(-algv[2] / grav)
            elif mode == 40:
                apgv = self.guidance_point(vehicle)
                alcomx = float(apgv[1] / grav)
            elif mode == 70:
                phicx = self.guidance_arc(vehicle)
        elif mguidp in (1, 2):
            ancomx = self._guidance_tfoa(vehicle, ctx, mguidp)
            if mguidl == 3:
                algv = self.guidance_line(vehicle)
                alcomx = float(algv[1] / grav)
                acv2 = float(algv[1])
        else:
            raise ValueError(f"unknown mguidance {mguidance}")

        # Fortran C1 bank-to-turn: PHIBVC from ACV2; MROLL=1 inverts
        mturn = int(store.get("mturn")) if "mturn" in store else 1
        if mguidl == 3 and mturn != 0 and acv2 is not None:
            phibvc = self._phibvc_bank_to_turn(vehicle, acv2, grav)
            if mroll == 1:
                phibvc = 3.1412 - phibvc
            store.set("phibvc", phibvc)

        anposlimx = store.get("anposlimx")
        anneglimx = store.get("anneglimx")
        allimx = store.get("allimx")
        if ancomx > anposlimx:
            ancomx = anposlimx
        if ancomx < anneglimx:
            ancomx = anneglimx
        if alcomx > allimx:
            alcomx = allimx
        if alcomx < -allimx:
            alcomx = -allimx
        store.set("phicx", phicx)
        store.set("ancomx", ancomx)
        store.set("alcomx", alcomx)

    def terminate(self, vehicle, ctx):
        pass

    def _phibvc_bank_to_turn(self, vehicle, acv2, grav):
        """Fortran C1 bank-to-turn PHIBVC from ACV2 (before MROLL invert)."""
        store = vehicle.store
        ggp = float(store.get("ggp"))
        bgp = float(store.get("bgp"))
        aldead = float(store.get("aldead"))
        philim = float(store.get("philim")) if "philim" in store else 0.0
        if philim <= 0.0 and "philimx" in store:
            philim = float(store.get("philimx")) * RAD
        if "FSPCB" in store:
            fspcb3 = float(np.asarray(store.get("FSPCB"), dtype=float).reshape(3)[2])
        elif "FSPV" in store:
            fspcb3 = float(np.asarray(store.get("FSPV"), dtype=float).reshape(3)[2])
        else:
            # No INS/forces yet — nominal 1-g level (FSPCB3≈-AGRAV)
            fspcb3 = -grav
        dum = acv2 / grav
        if abs(dum) < aldead:
            dum = 0.0
        pc = ggp / (0.01 + bgp + abs(fspcb3 / grav))
        phibvc = -_sign(fspcb3) * pc * dum
        if phibvc > philim:
            phibvc = philim
        if phibvc < -philim:
            phibvc = -philim
        return phibvc

    def _guidance_tfoa(self, vehicle, ctx, mguidp):
        """Fortran D1TER + C1 look-down (1) / look-fwd (2) TF/OA."""
        store = vehicle.store
        int_step = ctx.int_step
        hbe = float(store.get("hbe"))
        if abs(hbe) < EPS and "alt" in store:
            hbe = float(store.get("alt"))
            store.set("hbe", hbe)

        vbeg = store.get("vbeg")
        vhor = hypot(float(vbeg[0]), float(vbeg[1]))
        if vhor < EPS:
            vhor = max(float(store.get("dvbe")), EPS)

        # Groundtrack: Fortran D1 updates before C1. Unit tests plant GNDTCK.
        if not self._terrain_planted:
            gndtck = float(store.get("gndtck")) + vhor * int_step
            store.set("gndtck", gndtck)

        if self._terrain_planted:
            hge = float(store.get("hge"))
        else:
            hge = self._d1ter(vehicle, ctx, vhor)
            store.set("hge", hge)

        ancomx = 0.0
        if mguidp == 1:
            hbg = hbe - hge
            store.set("hbg", hbg)
            clobn = float(store.get("clobn"))
            if hbg <= 0.0:
                if self._iset_d1 == 1:
                    self._iset_d1 = 0
                    self._tstart = float(store.get("time"))
                    clobn = clobn + 1.0
                    store.set("clobn", clobn)
            else:
                dtim = float(store.get("time")) - self._tstart
                rcor = float(store.get("rcor"))
                if vhor > EPS and dtim > rcor / vhor:
                    self._iset_d1 = 1
            return ancomx

        return self._c1_look_fwd(vehicle, hbe, hge)

    def _d1ter(self, vehicle, ctx, vhor):
        """Fortran D1TER — 2nd-order autocorrelated stochastic terrain."""
        store = vehicle.store
        int_step = ctx.int_step
        hgmean = float(store.get("hgmean"))
        hgbias = float(store.get("hgbias"))
        facth = float(store.get("facth"))
        rcor = float(store.get("rcor"))
        sigmah = float(store.get("sigmah"))
        iseed = int(store.get("iseed"))
        hr = float(store.get("hr"))
        hrd = float(store.get("hrd"))
        hg = float(store.get("hg"))
        hgd = float(store.get("hgd"))

        if vhor < EPS:
            hge = hg + hgbias + hgmean
            store.set("hge", hge)
            return hge

        sigma = 1.0 / sqrt(int_step)
        v1, iseed = _ran_ms(iseed)
        while v1 == 0.0:
            v1, iseed = _ran_ms(iseed)
        v2, iseed = _ran_ms(iseed)
        gauss = sigma * sqrt(2.0 * log(1.0 / v1)) * cos(6.2831853072 * v2)

        tau = rcor / vhor
        dum = sigmah * sqrt(2.0 * (1.0 + facth)) / (facth * (tau**1.5))
        hrd_new = -hr / tau + dum * gauss
        hgd_new = -hg / (facth * tau) + hr
        hgslop = -hg / (facth * rcor) + hr / vhor
        hge = hg + hgbias + hgmean

        hr = integrate(hrd_new, hrd, hr, int_step)
        hg = integrate(hgd_new, hgd, hg, int_step)

        store.set("iseed", iseed)
        store.set("hr", hr)
        store.set("hrd", hrd_new)
        store.set("hg", hg)
        store.set("hgd", hgd_new)
        store.set("hgslop", hgslop)
        store.set("hge", hge)
        return hge

    def _c1_look_fwd(self, vehicle, hbe, hge):
        """Fortran C1 look-fwd terrain stack + TF/OA pitch command."""
        store = vehicle.store
        dcell = float(store.get("dcell"))
        rahead = float(store.get("rahead"))
        dhtrc = float(store.get("dhtrc"))
        slope = float(store.get("slope"))
        occden = float(store.get("occden"))
        sigobs = float(store.get("sigobs"))
        iseed2 = int(store.get("iseed2"))
        tlead = float(store.get("tlead"))
        gelev = float(store.get("gelev"))
        racq = float(store.get("racq"))
        gndtck = float(store.get("gndtck"))
        gndpt = float(store.get("gndpt"))
        j = int(store.get("j_ter"))
        gndocc = float(store.get("gndocc"))
        thtvl = float(store.get("thtvl"))
        if abs(thtvl) < EPS and "thtvgx" in store:
            thtvl = float(store.get("thtvgx")) * RAD
            store.set("thtvl", thtvl)
        dvbe = float(store.get("dvbe"))
        clmean = float(store.get("clmean"))
        clobn = float(store.get("clobn"))

        n = _nint(rahead / dcell) if dcell > 0 else 0
        k = _nint(racq / dcell) + 1 if dcell > 0 else 1
        if k < 1:
            k = 1
        if k > _TERRAIN_STACK:
            k = _TERRAIN_STACK
        if n < 1:
            n = 1
        if n > _TERRAIN_STACK:
            n = _TERRAIN_STACK

        dgnd = gndtck - gndpt
        # Fortran NL is a static local: do not reset to 1 every call
        dhobst = 0.0
        elmax = float(store.get("elmax"))

        if dgnd >= dcell and dcell > 0:
            if j < n:
                j = j + 1
                gndpt = gndtck
                self._h[j] = hge
                self._ho[j] = hge
                self._nl = 1
                self._ncell = 0
                self._iset_c1 = 1
                self._sig2 = 0.0
                clmean = 0.0
                gndocc = 0.0
            else:
                gndpt = gndtck
                for i in range(1, n):
                    self._h[i] = self._h[i + 1]
                    self._ho[i] = self._ho[i + 1]
                self._h[n] = hge
                self._ho[n] = hge

                dhobst = 0.0
                if gndtck >= gndocc and occden > 0.0:
                    u1, iseed2 = _ran_ms(iseed2)
                    docc = -log(1.0 - u1) / occden
                    gndocc = gndtck + docc
                    u2, iseed2 = _ran_ms(iseed2)
                    rayl = sqrt(-2.0 * log(1.0 - u2))
                    dhobst = rayl * sigobs
                    self._ho[k] = self._ho[k] + dhobst

                elm = -99999.0
                for i in range(2, n + 1):
                    el_i = (self._ho[i] + dhtrc - hbe) / ((i - 1) * dcell)
                    if el_i > elm:
                        elm = el_i
                    elmax = elm - thtvl
                    rlead = dvbe * tlead
                    nl = _nint(rlead / dcell) + 1
                    if nl < 1:
                        nl = 1
                    if nl > n:
                        nl = n
                    self._nl = nl

                    clear = hbe - self._ho[1]
                    if clear <= 0.0:
                        if self._iset_c1 == 1:
                            self._iset_c1 = 0
                            clobn = clobn + 1.0
                    else:
                        self._iset_c1 = 1

                    self._ncell = self._ncell + 1
                    clmm2 = clmean * clmean
                    clmean = clmean + (clear - clmean) / (self._ncell + 1)
                    clm2 = clmean * clmean
                    self._sig2 = (
                        self._sig2
                        + (clear * clear - self._sig2 - clmm2) / (self._ncell + 1)
                        + clmm2
                        - clm2
                    )
                    clsigm = sqrt(self._sig2) if self._sig2 > 0.0 else 0.0
                    store.set("clsigm", clsigm)

        nl = self._nl
        if nl < 1:
            nl = 1
        if nl > n:
            nl = n

        hges = self._ho[1]
        hclear = hbe - self._ho[1]
        ancomx = 0.0
        if elmax < slope:
            tfoa = 10.0
            # Fortran MAUT=13 (alt hold). Do not write maut — Cruise5Control
            # only accepts maut digits 0/6 today (Task 77); TF uses hbgs/tfoa.
            hbgs = hbe - self._h[nl]
            store.set("hbgs", hbgs)
        else:
            tfoa = 9.0
            # Fortran MAUT=14 (accel hold) → ANCOM
            ancomx = 1.0 + gelev * elmax * dvbe

        store.set("j_ter", j)
        store.set("gndpt", gndpt)
        store.set("gndocc", gndocc)
        store.set("iseed2", iseed2)
        store.set("dhobst", dhobst)
        store.set("elmax", elmax)
        store.set("hges", hges)
        store.set("hclear", hclear)
        store.set("clmean", clmean)
        store.set("clobn", clobn)
        store.set("tfoa", tfoa)
        return ancomx

    def guidance_pronav(self, vehicle):
        store = vehicle.store
        pronav_gain = store.get("pronav_gain")
        grav = store.get("grav")
        tbg = store.get("TBG")
        woeb = store.get("WOEB")
        closing_speed = store.get("closing_speed")
        utbb = store.get("UTBB")
        grav_g = np.array([0.0, 0.0, grav])
        return skew(woeb) @ utbb * (pronav_gain * closing_speed) - tbg @ grav_g

    def guidance_line(self, vehicle):
        store = vehicle.store
        line_gain = store.get("line_gain")
        nl_gain_fact = store.get("nl_gain_fact")
        decrement = store.get("decrement")
        wp_lonx = store.get("wp_lonx")
        wp_latx = store.get("wp_latx")
        wp_alt = store.get("wp_alt")
        psifgx = store.get("psifgx")
        thtfgx = store.get("thtfgx")
        time = store.get("time")
        grav = store.get("grav")
        tig = store.get("tig")
        thtvgx = store.get("thtvgx")
        vbeg = store.get("vbeg")
        sbii = store.get("sbii")
        philimx = store.get("philimx")

        tfg = mat2tr(psifgx * RAD, thtfgx * RAD)
        swii = cadine(wp_lonx * RAD, wp_latx * RAD, wp_alt, time)
        swbg = tig.T @ (swii - sbii)
        polar = polar_from_cart(swbg)
        wp_sltrange = float(polar[0])
        tog = mat2tr(float(polar[1]), float(polar[2]))
        wp_grdrange = hypot(float(swbg[0]), float(swbg[1]))
        vbeo = tog @ vbeg
        vbef = tfg @ vbeg
        nl_gain = nl_gain_fact * (1 - exp(-wp_sltrange / decrement))
        algv = np.array(
            [
                grav * sin(thtvgx * RAD),
                line_gain * (-vbeo[1] + nl_gain * vbef[1]),
                line_gain * (-vbeo[2] + nl_gain * vbef[2])
                - grav * cos(thtvgx * RAD),
            ]
        )
        dvbe = sqrt(float(vbeg @ vbeg))
        rad_min = dvbe * dvbe / (grav * tan(philimx * RAD))
        if wp_grdrange < 2 * rad_min:
            sh = np.array([swbg[0], swbg[1], 0.0])
            vh = np.array([vbeg[0], vbeg[1], 0.0])
            wp_flag = _sign(float(vh @ sh))
        else:
            wp_flag = 0

        store.set("wp_sltrange", wp_sltrange)
        store.set("nl_gain", nl_gain)
        store.set("VBEO", vbeo)
        store.set("VBEF", vbef)
        store.set("wp_grdrange", wp_grdrange)
        store.set("SWBG", swbg)
        store.set("rad_min", rad_min)
        store.set("wp_flag", wp_flag)
        return algv

    def guidance_point(self, vehicle):
        store = vehicle.store
        wp_lonx = store.get("wp_lonx")
        wp_latx = store.get("wp_latx")
        wp_alt = store.get("wp_alt")
        point_gain = store.get("point_gain")
        time = store.get("time")
        grav = store.get("grav")
        tig = store.get("tig")
        thtvgx = store.get("thtvgx")
        vbeg = store.get("vbeg")
        sbii = store.get("sbii")
        philimx = store.get("philimx")

        swii = cadine(wp_lonx * RAD, wp_latx * RAD, wp_alt, time)
        swbg = tig.T @ (swii - sbii)
        polar = polar_from_cart(swbg)
        wp_sltrange = float(polar[0])
        tog = mat2tr(float(polar[1]), float(polar[2]))
        vbeo = tog @ vbeg
        apgv = np.array(
            [
                grav * sin(thtvgx * RAD),
                point_gain * (-vbeo[1]),
                point_gain * (-vbeo[2]) - grav * cos(thtvgx * RAD),
            ]
        )
        wp_grdrange = hypot(float(swbg[0]), float(swbg[1]))
        dvbe = sqrt(float(vbeg @ vbeg))
        rad_min = dvbe * dvbe / (grav * tan(philimx * RAD))
        if wp_grdrange < 2 * rad_min:
            sh = np.array([swbg[0], swbg[1], 0.0])
            vh = np.array([vbeg[0], vbeg[1], 0.0])
            wp_flag = _sign(float(vh @ sh))
        else:
            wp_flag = 0

        store.set("wp_sltrange", wp_sltrange)
        store.set("VBEO", vbeo)
        store.set("wp_grdrange", wp_grdrange)
        store.set("SWBG", swbg)
        store.set("rad_min", rad_min)
        store.set("wp_flag", wp_flag)
        return apgv

    def guidance_arc(self, vehicle):
        store = vehicle.store
        wp_lonx = store.get("wp_lonx")
        wp_latx = store.get("wp_latx")
        wp_alt = store.get("wp_alt")
        time = store.get("time")
        fspv = store.get("FSPV")
        grav = store.get("grav")
        tig = store.get("tig")
        dvbe = store.get("dvbe")
        vbeg = store.get("vbeg")
        sbii = store.get("sbii")
        alphax = store.get("alphax")
        phimvx = store.get("phimvx")
        philimx = store.get("philimx")

        swii = cadine(wp_lonx * RAD, wp_latx * RAD, wp_alt, time)
        swbg = tig.T @ (swii - sbii)
        swbg1 = float(swbg[0])
        swbg2 = float(swbg[1])
        sh = np.array([swbg1, swbg2, 0.0])
        dwbh = sqrt(swbg1 * swbg1 + swbg2 * swbg2)
        vbeg1 = float(vbeg[0])
        vbeg2 = float(vbeg[1])
        vh = np.array([vbeg1, vbeg2, 0.0])
        uv = skew(vh) @ sh
        psiwvx = DEG * _angle(vh, sh)
        zz = np.array([0.0, 0.0, 1.0])
        psiwvx = psiwvx * _sign(float(uv[0] * zz[0] + uv[1] * zz[1] + uv[2] * zz[2]))
        alpha = alphax * RAD
        phimv = phimvx * RAD
        tbv = cadtbv(phimv, alpha)
        fspb = tbv @ fspv
        fspb3 = float(fspb[2])
        argument = 0.0
        if abs(psiwvx) < 90:
            num = -2 * dvbe * dvbe * sin(psiwvx * RAD)
            denom = fspb3 * dwbh
            if denom != 0:
                argument = num / denom
            if abs(argument) <= 1.0 and abs(asin(argument)) < philimx * RAD:
                phicx = DEG * asin(argument)
            else:
                phicx = philimx * _sign(argument)
        else:
            phicx = philimx * _sign(psiwvx)
        rad_geometric = 0.0
        if psiwvx != 0:
            rad_geometric = abs(dwbh / (2 * sin(psiwvx * RAD)))
        rad_min = dvbe * dvbe / (grav * tan(philimx * RAD))
        if dwbh < 2 * rad_min:
            wp_flag = _sign(float(vh[0] * sh[0] + vh[1] * sh[1] + vh[2] * sh[2]))
        else:
            wp_flag = 0
        wp_grdrange = dwbh

        store.set("SWBG", swbg)
        store.set("wp_grdrange", wp_grdrange)
        store.set("rad_min", rad_min)
        store.set("rad_geometric", rad_geometric)
        store.set("wp_flag", wp_flag)
        return phicx
