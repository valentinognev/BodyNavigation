from math import cos, sin

import numpy as np

from cadac.constants import DEG, RAD
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import cadtbv

_ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def _a3tra_stt(alpha: float, beta: float) -> np.ndarray:
    """Fortran A3TRA yaw-to-turn TBV from ALPHA/BETA (MTURN=0)."""
    calp = cos(alpha)
    salp = sin(alpha)
    cbet = cos(beta)
    sbet = sin(beta)
    return np.array(
        [
            [calp * cbet, -calp * sbet, -salp],
            [sbet, cbet, 0.0],
            [salp * cbet, -salp * sbet, calp],
        ],
        dtype=float,
    )


class Cruise5Control:
    name = "control"

    def define(self, vehicle):
        store = vehicle.store
        plot = ("scrn", "plot")
        for field in (
            Field("mcontrol", 0, "int", "data", "control", ("scrn",)),
            Field("psivgcx", 0.0, "real", "data", "control", ("plot",)),
            Field("thtvgcx", 0.0, "real", "data", "control", ("plot",)),
            Field("alphacx", 0.0, "real", "data", "control"),
            Field("phimvcx", 0.0, "real", "data", "control"),
            Field("TBV", _ZEROS33, "mat", "out", "control"),
            Field("TBG", _ZEROS33, "mat", "out", "control"),
            Field("gain_thtvg", 0.0, "real", "data", "control"),
            Field("gain_psivg", 0.0, "real", "data", "control"),
            Field("anx", 0.0, "real", "diag", "control", plot),
            Field("avx", 0.0, "real", "diag", "control", plot),
            Field("alphax", 0.0, "real", "out", "control", plot),
            Field("phimvx", 0.0, "real", "out", "control", plot),
            Field("phicx", 0.0, "real", "data", "control", plot),
            Field("phix", 0.0, "real", "state", "control", ("plot",)),
            Field("phixd", 0.0, "real", "state", "control"),
            Field("philimx", 0.0, "real", "data", "control"),
            Field("tphi", 0.0, "real", "data", "control"),
            Field("anposlimx", 0.0, "real", "data", "control"),
            Field("anneglimx", 0.0, "real", "data", "control"),
            Field("gacp", 0.0, "real", "data", "control"),
            Field("ta", 0.0, "real", "data", "control"),
            Field("xi", 0.0, "real", "state", "control"),
            Field("xid", 0.0, "real", "state", "control"),
            Field("qq", 0.0, "real", "diag", "control", ("plot",)),
            Field("tip", 0.0, "real", "diag", "control", ("plot",)),
            Field("alp", 0.0, "real", "state", "control", ("plot",)),
            Field("alpd", 0.0, "real", "state", "control"),
            Field("alpposlimx", 0.0, "real", "data", "control"),
            Field("alpneglimx", 0.0, "real", "data", "control"),
            Field("ancomx", 0.0, "real", "data", "control", plot),
            Field("alcomx", 0.0, "real", "data", "control", plot),
            Field("allimx", 0.0, "real", "data", "control"),
            Field("gcp", 0.0, "real", "data", "control"),
            Field("alx", 0.0, "real", "diag", "control", ("plot",)),
            Field("altdlim", 0.0, "real", "data", "control"),
            Field("gh", 0.0, "real", "data", "control"),
            Field("gv", 0.0, "real", "data", "control"),
            Field("altd", 0.0, "real", "diag", "control", ("plot",)),
            Field("altcom", 0.0, "real", "data", "control", ("plot",)),
            # Fortran C2 MAUT / APTVC (MODULE.FOR) — dual path with mcontrol
            Field("maut", 0, "int", "data", "control", ("scrn",)),
            Field("mturn", 1, "int", "data", "control"),
            Field("alphac", 0.0, "real", "data", "control"),
            Field("betac", 0.0, "real", "data", "control"),
            Field("wqc", 0.0, "real", "data", "control"),
            Field("wpc", 0.0, "real", "data", "control"),
            Field("aptvc", 0.0, "real", "data", "control"),
            Field("ga", 0.0, "real", "data", "control"),
            Field("gp", 0.0, "real", "data", "control"),
            Field("aiz", 0.0, "real", "data", "control"),
            Field("cmdel", 0.0, "real", "data", "control"),
            Field("rleng", 0.0, "real", "data", "control"),
            Field("parm", 0.0, "real", "data", "control"),
            Field("flplim", 0.0, "real", "data", "control"),
            Field("tvclim", 0.0, "real", "data", "control"),
            Field("alplim", 0.0, "real", "data", "control"),
            Field("cnalp", 0.0, "real", "data", "control"),
            Field("tr", 0.0, "real", "data", "control"),
            Field("philim", 0.0, "real", "data", "control"),
            Field("ratep", 0.0, "real", "state", "control"),
            Field("ratepd", 0.0, "real", "state", "control"),
            Field("ph", 0.0, "real", "state", "control"),
            Field("phd", 0.0, "real", "state", "control"),
            Field("xphi", 0.0, "real", "state", "control"),
            Field("xphid", 0.0, "real", "state", "control"),
            Field("delq", 0.0, "real", "diag", "control", ("plot",)),
            Field("eta", 0.0, "real", "diag", "control", ("plot",)),
            Field("polea", 0.0, "real", "diag", "control"),
            Field("polep", 0.0, "real", "diag", "control"),
            Field("trcalc", 0.0, "real", "diag", "control"),
            Field("aermp", 0.0, "real", "diag", "control"),
            Field("tvcmp", 0.0, "real", "diag", "control"),
            Field("betax", 0.0, "real", "out", "control", plot),
        ):
            if field.name not in store:
                store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        if store.get("maut") != 0:
            self._execute_maut(vehicle, ctx)
            return
        mcontrol = store.get("mcontrol")
        if mcontrol not in (0, 1, 3, 4, 6, 10, 11, 16, 36, 40, 44, 46):
            raise ValueError(f"unknown mcontrol {mcontrol}")
        int_step = ctx.int_step
        ancomx = store.get("ancomx")
        alcomx = store.get("alcomx")
        altcom = store.get("altcom")
        phicx = store.get("phicx")
        tgv = store.get("tgv")
        thtvgcx = store.get("thtvgcx")
        psivgcx = store.get("psivgcx")
        alphacx = store.get("alphacx")
        phimvx = 0.0
        alphax = 0.0
        if mcontrol == 0:
            phimvx = 0.0
            alphax = 0.0
        if mcontrol == 1:
            phimvx = 0.0
            alphax = self.control_flightpath(vehicle, thtvgcx, phimvx)
        if mcontrol == 10:
            phicx = self.control_heading(vehicle, psivgcx)
            phimvx = self.control_bank(vehicle, phicx, int_step)
            alphax = alphacx
        if mcontrol == 11:
            phicx = self.control_heading(vehicle, psivgcx)
            phimvx = self.control_bank(vehicle, phicx, int_step)
            alphax = self.control_flightpath(vehicle, thtvgcx, phimvx)
        if mcontrol == 3:
            phimvx = self.control_bank(vehicle, phicx, int_step)
            alphax = alphacx
        if mcontrol == 4:
            alphax = self.control_load(vehicle, ancomx, int_step)
        if mcontrol == 40:
            phicx = self.control_lateral(vehicle, alcomx)
            phimvx = self.control_bank(vehicle, phicx, int_step)
        if mcontrol == 44:
            phicx = self.control_lateral(vehicle, alcomx)
            phimvx = self.control_bank(vehicle, phicx, int_step)
            alphax = self.control_load(vehicle, ancomx, int_step)
        if mcontrol == 6:
            ancomx = self.control_altitude(vehicle, altcom, phimvx)
            alphax = self.control_load(vehicle, ancomx, int_step)
        if mcontrol == 16:
            phicx = self.control_heading(vehicle, psivgcx)
            phimvx = self.control_bank(vehicle, phicx, int_step)
            ancomx = self.control_altitude(vehicle, altcom, phimvx)
            alphax = self.control_load(vehicle, ancomx, int_step)
        if mcontrol == 46:
            phicx = self.control_lateral(vehicle, alcomx)
            phimvx = self.control_bank(vehicle, phicx, int_step)
            ancomx = self.control_altitude(vehicle, altcom, phimvx)
            alphax = self.control_load(vehicle, ancomx, int_step)
        if mcontrol == 36:
            phimvx = self.control_bank(vehicle, phicx, int_step)
            ancomx = self.control_altitude(vehicle, altcom, phimvx)
            alphax = self.control_load(vehicle, ancomx, int_step)
        tbv = cadtbv(phimvx * RAD, alphax * RAD)
        tbg = tbv @ tgv.T
        store.set("phicx", phicx)
        store.set("TBV", tbv)
        store.set("TBG", tbg)
        store.set("alphax", alphax)
        store.set("phimvx", phimvx)
        store.set("ancomx", ancomx)

    def terminate(self, vehicle, ctx):
        pass

    def _execute_maut(self, vehicle, ctx):
        """Fortran C2 MAUT=|MAUTL|MAUTP| (digits 1,6 live; others later)."""
        store = vehicle.store
        maut = store.get("maut")
        mautl = int(maut / 10)
        mautp = maut - mautl * 10
        mturn = store.get("mturn")
        int_step = ctx.int_step
        tgv = store.get("tgv")
        alpha = 0.0
        beta = 0.0
        phibv = 0.0

        if mautp == 1:
            # Fortran: ALPHA=ALPHAC
            alpha = store.get("alphac")
        elif mautp == 6:
            alpha = self.c2_pitch(vehicle, store.get("wqc"), int_step)
        elif mautp not in (0,):
            raise ValueError(f"unknown mautp {mautp}")

        if mturn == 0:
            phibv = 0.0
            if mautl == 1:
                # Fortran: BETA=BETAC (sideslip angle hold)
                beta = store.get("betac")
            elif mautl == 6:
                # Live Fortran comments out STT yaw-rate; BTT path only.
                raise ValueError("mautl=6 requires mturn=1 (BTT)")
            elif mautl not in (0,):
                raise ValueError(f"unknown mautl {mautl} (mturn=0)")
        else:
            beta = 0.0
            if mautl == 6:
                # Fortran: PHIC=WPC; CALL C2PHI; XPHID=PHI; PHIBV=XPHI
                phi, _phid = self.c2_phi(vehicle, store.get("wpc"), int_step)
                xphi = store.get("xphi")
                xphid_old = store.get("xphid")
                phibv = xphi
                xphi = integrate(phi, xphid_old, xphi, int_step)
                store.set("xphid", phi)
                store.set("xphi", xphi)
            elif mautl not in (0,):
                raise ValueError(f"unknown mautl {mautl} (mturn=1)")

        alphax = alpha * DEG
        betax = beta * DEG
        phimvx = phibv * DEG
        if mturn == 0:
            tbv = _a3tra_stt(alpha, beta)
        else:
            tbv = cadtbv(phimvx * RAD, alphax * RAD)
        tbg = tbv @ tgv.T
        store.set("TBV", tbv)
        store.set("TBG", tbg)
        store.set("alphax", alphax)
        store.set("betax", betax)
        store.set("phimvx", phimvx)

    def c2_pitch(self, vehicle, pitch, int_step):
        """Fortran C2PITCH — aero/TVC rate loop (TR=0 detailed; TR>0 lag)."""
        store = vehicle.store
        aptvc = store.get("aptvc")
        ga = store.get("ga")
        gp = store.get("gp")
        aiz = store.get("aiz")
        cmdel = store.get("cmdel")
        rleng = store.get("rleng")
        parm = store.get("parm")
        flplim = store.get("flplim")
        tvclim = store.get("tvclim")
        alplim = store.get("alplim")
        cnalp = store.get("cnalp")
        tr = store.get("tr")
        pdynmc = store.get("pdynmc")
        area = store.get("area")
        fthalt = store.get("thrust")
        amass = store.get("mass")
        dvba = store.get("dvbe")
        ratep = store.get("ratep")
        ratepd = store.get("ratepd")
        alp = store.get("alp")
        alpd = store.get("alpd")

        ratepc = pitch
        polea = 0.0
        polep = 0.0
        delq = 0.0
        eta = 0.0
        aermp = 0.0
        tvcmp = 0.0
        trcalc = store.get("trcalc")

        if tr > 0.0:
            ratepd_new = (ratepc - ratep) / tr
        else:
            cmom = pdynmc * area * rleng * cmdel
            polea = (1.0 - aptvc) * ga * cmom / aiz
            polep = aptvc * gp * fthalt * parm / aiz
            pole = polea + polep
            if pole != 0.0:
                trcalc = 1.0 / pole
            eratep = ratepc - ratep
            delq = -eratep * (1.0 - aptvc) * ga
            if delq > flplim:
                delq = flplim
            if delq < -flplim:
                delq = -flplim
            aermp = -delq * cmom
            if fthalt > 0:
                eta = -eratep * aptvc * gp
            else:
                eta = 0.0
            if eta > tvclim:
                eta = tvclim
            if eta < -tvclim:
                eta = -tvclim
            tvcmp = -eta * fthalt * parm
            amp = aermp + tvcmp
            ratepd_new = amp / aiz

        tip = dvba * amass / (pdynmc * area * cnalp + fthalt)
        alpd_new = (tip * ratep - alp) / tip
        alpha = alp
        if alpha > alplim:
            alpha = alplim
        if alpha < -alplim:
            alpha = -alplim

        ratep = integrate(ratepd_new, ratepd, ratep, int_step)
        alp = integrate(alpd_new, alpd, alp, int_step)

        store.set("ratep", ratep)
        store.set("ratepd", ratepd_new)
        store.set("alp", alp)
        store.set("alpd", alpd_new)
        store.set("delq", delq)
        store.set("eta", eta)
        store.set("polea", polea)
        store.set("polep", polep)
        store.set("trcalc", trcalc)
        store.set("aermp", aermp)
        store.set("tvcmp", tvcmp)
        store.set("tip", tip)
        return alpha

    def c2_phi(self, vehicle, phic, int_step):
        """Fortran C2PHI — first-order roll lag (radians). Returns PHI=PH pre-integrate."""
        store = vehicle.store
        philim = store.get("philim")
        tphi = store.get("tphi")
        ph = store.get("ph")
        phd = store.get("phd")
        # MROLL gating is Task 78; default MROLL=0 always limits.
        if phic > philim:
            phic = philim
        if phic < -philim:
            phic = -philim
        phd_new = (phic - ph) / tphi
        phi_out = ph
        ph = integrate(phd_new, phd, ph, int_step)
        store.set("ph", ph)
        store.set("phd", phd_new)
        return phi_out, phd_new

    def control_bank(self, vehicle, phicx, int_step):
        store = vehicle.store
        philimx = store.get("philimx")
        tphi = store.get("tphi")
        phix = store.get("phix")
        phixd = store.get("phixd")
        if phicx > philimx:
            phicx = philimx
        if phicx < -philimx:
            phicx = -philimx
        phixd_new = (phicx - phix) / tphi
        phix = integrate(phixd_new, phixd, phix, int_step)
        store.set("phix", phix)
        store.set("phixd", phixd_new)
        return phix

    def control_load(self, vehicle, ancomx, int_step):
        store = vehicle.store
        anposlimx = store.get("anposlimx")
        anneglimx = store.get("anneglimx")
        phimvx = store.get("phimvx")
        gacp = store.get("gacp")
        ta = store.get("ta")
        alphax = store.get("alphax")
        alpposlimx = store.get("alpposlimx")
        alpneglimx = store.get("alpneglimx")
        fspv = store.get("FSPV")
        grav = store.get("grav")
        xi = store.get("xi")
        xid = store.get("xid")
        alp = store.get("alp")
        alpd = store.get("alpd")
        mass = store.get("mass")
        dvbe = store.get("dvbe")
        pdynmc = store.get("pdynmc")
        thrust = store.get("thrust")
        area = store.get("area")
        cla = store.get("cla")

        alpha = alphax * RAD
        phimv = phimvx * RAD
        tbv = cadtbv(phimv, alpha)
        fspb = tbv @ fspv

        if ancomx > anposlimx:
            ancomx = anposlimx
        if ancomx < anneglimx:
            ancomx = anneglimx

        fspb3 = fspb[2]
        anx = -fspb3 / grav
        eanx = ancomx - anx
        tip = dvbe * mass / (pdynmc * area * cla / RAD + thrust)

        gr = 0.0
        if ta > 0:
            gr = gacp * tip / dvbe
            gi = gr / ta
            xid_new = gi * eanx
            xi = integrate(xid_new, xid, xi, int_step)
            xid = xid_new
        else:
            xi = 0.0

        qq = gr * eanx + xi
        alpd_new = qq - alp / tip
        alp = integrate(alpd_new, alpd, alp, int_step)
        alpd = alpd_new

        alpx = alp * DEG
        if alpx > alpposlimx:
            alpx = alpposlimx
        if alpx < alpneglimx:
            alpx = alpneglimx

        store.set("xi", xi)
        store.set("xid", xid)
        store.set("alp", alp)
        store.set("alpd", alpd)
        store.set("anx", anx)
        store.set("qq", qq)
        store.set("tip", tip)
        return alpx

    def control_heading(self, vehicle, psivgcx):
        store = vehicle.store
        gain_psivg = store.get("gain_psivg")
        psivgx = store.get("psivgx")
        if abs(psivgcx) <= 135:
            psivgx_comp = psivgx
        else:
            if psivgx * psivgcx >= 0:
                psivgx_comp = psivgx
            else:
                if psivgx >= 0:
                    sign_psivgx = 1
                else:
                    sign_psivgx = -1
                psivgx_comp = 360 - psivgx * sign_psivgx
        return gain_psivg * (psivgcx - psivgx_comp)

    def control_flightpath(self, vehicle, thtvgcx, phimvx):
        store = vehicle.store
        gain_thtvg = store.get("gain_thtvg")
        alpposlimx = store.get("alpposlimx")
        alpneglimx = store.get("alpneglimx")
        pdynmc = store.get("pdynmc")
        thtvg = store.get("thtvg")
        grav = store.get("grav")
        mass = store.get("mass")
        area = store.get("area")
        cla = store.get("cla")
        avx = gain_thtvg * (thtvgcx * RAD - thtvg)
        anx = avx / cos(phimvx * RAD)
        alphax = (anx * mass * grav) / (pdynmc * area * cla)
        if alphax > alpposlimx:
            alphax = alpposlimx
        if alphax < alpneglimx:
            alphax = alpneglimx
        store.set("anx", anx)
        store.set("avx", avx)
        return alphax

    def control_altitude(self, vehicle, altcom, phimvx):
        store = vehicle.store
        anposlimx = store.get("anposlimx")
        anneglimx = store.get("anneglimx")
        altdlim = store.get("altdlim")
        gh = store.get("gh")
        gv = store.get("gv")
        alt = store.get("alt")
        grav = store.get("grav")
        vbeg = store.get("vbeg")

        ealt = gh * (altcom - alt)
        if ealt > altdlim:
            ealt = altdlim
        if ealt < -altdlim:
            ealt = -altdlim
        altd = -vbeg[2]
        ancomx = (gv * (ealt - altd) / grav + 1) * (1 / cos(phimvx * RAD))
        if ancomx > anposlimx:
            ancomx = anposlimx
        if ancomx < anneglimx:
            ancomx = anneglimx

        store.set("altd", altd)
        return ancomx

    def control_lateral(self, vehicle, alcomx):
        store = vehicle.store
        allimx = store.get("allimx")
        phimvx = store.get("phimvx")
        alphax = store.get("alphax")
        gcp = store.get("gcp")
        fspv = store.get("FSPV")
        grav = store.get("grav")

        alpha = alphax * RAD
        phimv = phimvx * RAD
        tbv = cadtbv(phimv, alpha)
        fspb = tbv @ fspv
        fspb3 = fspb[2]
        anx = -fspb3 / grav

        if alcomx > allimx:
            alcomx = allimx
        if alcomx < -allimx:
            alcomx = -allimx

        sign = 1 if anx >= 0 else -1
        phic = gcp * sign / (abs(anx) + 0.001) * alcomx
        phicx = phic * DEG

        alx = fspv[1] / grav
        store.set("alx", alx)
        return phicx
