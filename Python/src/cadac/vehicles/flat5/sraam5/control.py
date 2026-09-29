"""SRAAM5 autopilot — Fortran MODULE.FOR subroutines C2 / C2PTCH / C2YAW."""

from math import acos, asin, atan, atan2, cos, sin, tan

from cadac.constants import AGRAV, DEG
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field


class Sraam5Control:
    """Autopilot Module (C2). MAUT=44 accel; MAUT=11 α/β hold; MTURN=1 BTT."""

    name = "control"

    def define(self, vehicle):
        store = vehicle.store
        plot = ("scrn", "plot")
        for field in (
            Field("maut", 0, "int", "data", "control"),
            Field("anplim", 0.0, "real", "data", "control"),
            Field("annlim", 0.0, "real", "data", "control"),
            Field("allim", 0.0, "real", "data", "control"),
            Field("betlim", 0.0, "real", "data", "control"),
            Field("alpplim", 0.0, "real", "data", "control"),
            Field("alplim", 0.0, "real", "data", "control"),
            Field("alnlim", 0.0, "real", "data", "control"),
            Field("factgacp", 0.0, "real", "data", "control"),
            Field("facttr", 0.0, "real", "data", "control"),
            Field("ta", 0.0, "real", "data", "control"),
            Field("gacp", 0.0, "real", "data", "control"),
            Field("gr", 0.0, "real", "data", "control"),
            Field("tr", 0.0, "real", "data", "control"),
            Field("tvclim", 0.0, "real", "data", "control"),
            Field("rleng", 0.0, "real", "data", "control"),
            Field("aptvc", 0.0, "real", "data", "control"),
            Field("ga", 0.0, "real", "data", "control"),
            Field("gp", 0.0, "real", "data", "control"),
            Field("parm", 0.0, "real", "data", "control"),
            Field("aiz", 0.0, "real", "data", "control"),
            Field("flplim", 0.0, "real", "data", "control"),
            # MAUTP=1 / MAUTL=1 (STT) commanded incidence
            Field("alphac", 0.0, "real", "data", "control"),
            Field("betac", 0.0, "real", "data", "control"),
            # Bank-to-turn roll lag (CADAC C2PHI; SRAAM5 C2 stripped, restored for MTURN=1)
            Field("phibvc", 0.0, "real", "data", "control"),
            Field("tphi", 0.0, "real", "data", "control"),
            Field("philim", 0.0, "real", "data", "control"),
            # Outputs
            Field("alpha", 0.0, "real", "out", "control"),
            Field("beta", 0.0, "real", "out", "control"),
            Field("alphap", 0.0, "real", "out", "control"),
            Field("phip", 0.0, "real", "out", "control"),
            Field("phibv", 0.0, "real", "out", "control"),
            Field("alphax", 0.0, "real", "out", "control", plot),
            Field("betax", 0.0, "real", "out", "control", plot),
            Field("alphapx", 0.0, "real", "diag", "control"),
            # States (C2I: 942-949 rate/incidence; 950-953 integral feedback)
            Field("alp", 0.0, "real", "state", "control"),
            Field("alpd", 0.0, "real", "state", "control"),
            Field("ratep", 0.0, "real", "state", "control"),
            Field("ratepd", 0.0, "real", "state", "control"),
            Field("bet", 0.0, "real", "state", "control"),
            Field("betd", 0.0, "real", "state", "control"),
            Field("ratey", 0.0, "real", "state", "control"),
            Field("rateyd", 0.0, "real", "state", "control"),
            Field("xi", 0.0, "real", "state", "control"),
            Field("xid", 0.0, "real", "state", "control"),
            Field("yi", 0.0, "real", "state", "control"),
            Field("yid", 0.0, "real", "state", "control"),
            # Roll state for MTURN=1 (D2 consumes PHD; HEAD C(0958))
            Field("ph", 0.0, "real", "state", "control"),
            Field("phd", 0.0, "real", "state", "control"),
            # Diagnostics from rate loops
            Field("tip", 0.0, "real", "diag", "control"),
            Field("trcalc", 0.0, "real", "diag", "control"),
            Field("aermp", 0.0, "real", "diag", "control"),
            Field("tvcmp", 0.0, "real", "diag", "control"),
            Field("eta", 0.0, "real", "diag", "control"),
            Field("delq", 0.0, "real", "diag", "control"),
            Field("aermy", 0.0, "real", "diag", "control"),
            Field("tvcmy", 0.0, "real", "diag", "control"),
            Field("zeta", 0.0, "real", "diag", "control"),
            Field("delr", 0.0, "real", "diag", "control"),
        ):
            if field.name not in store:
                store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        maut = int(store.get("maut"))
        mauta = int(maut / 100)
        mautl = int((maut - mauta * 100) / 10)
        mautp = maut - mauta * 100 - mautl * 10
        mturn = int(store.get("mturn")) if "mturn" in store else 0
        int_step = ctx.int_step

        pdynmc = store.get("pdynmc")
        facttr = store.get("facttr")
        factgacp = store.get("factgacp")
        # SRAAM5-specific autopilot parameters (overwrite deck TR/GACP)
        tr = (-2.0e-7 * pdynmc + 0.22) * (1.0 + facttr)
        gacp = (2.0e-3 * pdynmc) ** 0.575 * (1.0 + factgacp)
        store.set("tr", tr)
        store.set("gacp", gacp)

        alph = 0.0
        bett = 0.0

        # Angle of attack hold (MAUTP=1)
        if mautp == 1:
            alph = store.get("alphac")

        # Pitch acceleration hold (MAUTP=4)
        if mautp == 4 and mauta == 0:
            alph = self._pitch_accel(vehicle, int_step)

        if mturn == 0:
            # Yaw-to-turn (skid-to-turn)
            store.set("phibv", 0.0)
            # Sideslip angle hold (MAUTL=1)
            if mautl == 1:
                bett = store.get("betac")
            # Yaw acceleration hold (MAUTL=4)
            if mautl == 4 and mauta == 0:
                bett = self._yaw_accel(vehicle, int_step)
        else:
            # Bank-to-turn (α+φ): no sideslip; lateral accel → bank (C2ACCL+C2PHI)
            # MAUTL=1 under BTT is bank-command hold via PHIBVC (not BETAC)
            bett = 0.0
            if mautl == 4 and mauta == 0:
                phic = self._c2_accl(vehicle)
                phibv, phd = self._c2_phi(vehicle, phic, int_step)
                store.set("phibv", phibv)
                store.set("phd", phd)
            elif mautl == 1:
                phic = store.get("phibvc")
                phibv, phd = self._c2_phi(vehicle, phic, int_step)
                store.set("phibv", phibv)
                store.set("phd", phd)

        if mturn == 0:
            # Total angle of attack limiter (STT polar α/β)
            alpplim = store.get("alpplim")
            alphap = acos(cos(alph) * cos(bett))
            if alphap > alpplim:
                alphap = alpplim
            if alphap < 1.0e-10:
                phip = 0.0
            else:
                phip = atan2(tan(bett), sin(alph))
            alpha = atan(cos(phip) * tan(alphap))
            beta = asin(sin(phip) * sin(alphap))
        else:
            # BTT: α in body plane of symmetry; β=0 (A3TRA uses ALPHA, PHIBV)
            alpha = alph
            beta = 0.0
            alphap = abs(alph)
            phip = 0.0

        store.set("alphap", alphap)
        store.set("phip", phip)
        store.set("alpha", alpha)
        store.set("beta", beta)
        store.set("alphapx", DEG * alphap)
        store.set("alphax", DEG * alpha)
        store.set("betax", DEG * beta)

    def terminate(self, vehicle, ctx):
        pass

    def _pitch_accel(self, vehicle, int_step):
        """Fortran C2 MAUTP=4 + C2PTCH — returns ALPH (rad)."""
        store = vehicle.store
        ancom = store.get("ancom")
        anplim = store.get("anplim")
        annlim = store.get("annlim")
        if ancom > anplim:
            ancom = anplim
        if ancom < annlim:
            ancom = annlim

        fspcb = store.get("FSPCB")
        abecz = -ancom * AGRAV
        ep = abecz - fspcb[2]

        dvbe = store.get("dvbe")
        amass = store.get("amass")
        pdynmc = store.get("pdynmc")
        area = store.get("area")
        cnalp = store.get("cnalp")
        tip = dvbe * amass / (pdynmc * area * cnalp)

        ta = store.get("ta")
        gacp = store.get("gacp")
        tr = store.get("tr")
        xi = store.get("xi")
        xid = store.get("xid")
        gr = store.get("gr")

        if ta > 0.0:
            gr = gacp * tip * tr / dvbe
            gi = gr / ta
            xid_new = gi * ep
            pitch = -(ep * gr + xi)
            xi = integrate(xid_new, xid, xi, int_step)
            store.set("xid", xid_new)
            store.set("xi", xi)
        else:
            pitch = -ep * gr

        store.set("gr", gr)
        return self._c2_ptch(vehicle, pitch, int_step)

    def _c2_accl(self, vehicle):
        """Lateral accel → roll command (CADAC C2ACCL; SRAAM5 C2 stripped BTT)."""
        store = vehicle.store
        alcom = store.get("alcom")
        allim = store.get("allim")
        if alcom > allim:
            alcom = allim
        if alcom < -allim:
            alcom = -allim
        fspcb = store.get("FSPCB")
        pc = (alcom * AGRAV) / (abs(fspcb[2]) + 0.001)
        # SIGN(1., FSPCB(3)) in Fortran
        if fspcb[2] >= 0.0:
            return -pc
        return pc

    def _c2_phi(self, vehicle, phic, int_step):
        """Roll lag (CADAC C2PHI). Returns (PHI=PH pre-integrate, PHD)."""
        store = vehicle.store
        philim = store.get("philim")
        tphi = store.get("tphi")
        ph = store.get("ph")
        phd = store.get("phd")
        if philim > 0.0:
            if phic > philim:
                phic = philim
            if phic < -philim:
                phic = -philim
        if tphi > 0.0:
            phd_new = (phic - ph) / tphi
            phi_out = ph
            ph = integrate(phd_new, phd, ph, int_step)
            store.set("ph", ph)
            store.set("phd", phd_new)
            return phi_out, phd_new
        # No lag: instantaneous bank; PHD=0 for D2
        store.set("ph", phic)
        store.set("phd", 0.0)
        return phic, 0.0

    def _yaw_accel(self, vehicle, int_step):
        """Fortran C2 MAUTL=4 + C2YAW — returns BETT (rad)."""
        store = vehicle.store
        alcom = store.get("alcom")
        allim = store.get("allim")
        if alcom > allim:
            alcom = allim
        if alcom < -allim:
            alcom = -allim

        fspcb = store.get("FSPCB")
        abecy = alcom * AGRAV
        ey = abecy - fspcb[1]

        dvbe = store.get("dvbe")
        amass = store.get("amass")
        pdynmc = store.get("pdynmc")
        area = store.get("area")
        cybet = store.get("cybet")
        tiy = dvbe * amass / (-pdynmc * area * cybet)

        ta = store.get("ta")
        gacp = store.get("gacp")
        tr = store.get("tr")
        yi = store.get("yi")
        yid = store.get("yid")
        gr = store.get("gr")

        if ta > 0.0:
            gr = gacp * tiy * tr / dvbe
            gi = gr / ta
            yid_new = gi * ey
            yaw = ey * gr + yi
            yi = integrate(yid_new, yid, yi, int_step)
            store.set("yid", yid_new)
            store.set("yi", yi)
        else:
            yaw = ey * gr

        store.set("gr", gr)
        return self._c2_yaw(vehicle, yaw, int_step)

    def _c2_ptch(self, vehicle, pitch, int_step):
        """Fortran C2PTCH — pitch rate loop; returns ALPH (pre-integrate ALP)."""
        store = vehicle.store
        tr = store.get("tr")
        ratep = store.get("ratep")
        ratepd = store.get("ratepd")
        alp = store.get("alp")
        alpd = store.get("alpd")
        pdynmc = store.get("pdynmc")
        area = store.get("area")
        cnalp = store.get("cnalp")
        fthalt = store.get("fthalt")
        amass = store.get("amass")
        dvbe = store.get("dvbe")
        alplim = store.get("alplim")
        alnlim = store.get("alnlim")

        ratepc = pitch
        aermp = 0.0
        tvcmp = 0.0
        eta = 0.0
        delq = 0.0
        trcalc = store.get("trcalc")

        if tr > 0.0:
            ratepd_new = (ratepc - ratep) / tr
        else:
            aptvc = store.get("aptvc")
            ga = store.get("ga")
            gp = store.get("gp")
            aiz = store.get("aiz")
            cmdel = store.get("cmdel")
            rleng = store.get("rleng")
            parm = store.get("parm")
            flplim = store.get("flplim")
            tvclim = store.get("tvclim")
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
            ratepd_new = (aermp + tvcmp) / aiz

        tip = dvbe * amass / (pdynmc * area * cnalp + fthalt)
        alpd_new = (tip * ratep - alp) / tip
        alph = alp
        if alph > alplim:
            alph = alplim
        if alph < -alnlim:
            alph = -alnlim

        ratep = integrate(ratepd_new, ratepd, ratep, int_step)
        alp = integrate(alpd_new, alpd, alp, int_step)

        store.set("ratep", ratep)
        store.set("ratepd", ratepd_new)
        store.set("alp", alp)
        store.set("alpd", alpd_new)
        store.set("tip", tip)
        store.set("trcalc", trcalc)
        store.set("aermp", aermp)
        store.set("tvcmp", tvcmp)
        store.set("eta", eta)
        store.set("delq", delq)
        return alph

    def _c2_yaw(self, vehicle, yaw, int_step):
        """Fortran C2YAW — yaw rate loop; returns BETT (pre-integrate BET)."""
        store = vehicle.store
        tr = store.get("tr")
        ratey = store.get("ratey")
        rateyd = store.get("rateyd")
        bet = store.get("bet")
        betd = store.get("betd")
        pdynmc = store.get("pdynmc")
        area = store.get("area")
        cybet = store.get("cybet")
        fthalt = store.get("fthalt")
        amass = store.get("amass")
        dvbe = store.get("dvbe")
        betlim = store.get("betlim")

        rateyc = yaw
        aermy = 0.0
        tvcmy = 0.0
        zeta = 0.0
        delr = 0.0

        if tr > 0.0:
            rateyd_new = (rateyc - ratey) / tr
        else:
            aptvc = store.get("aptvc")
            ga = store.get("ga")
            gp = store.get("gp")
            aiz = store.get("aiz")
            cmdel = store.get("cmdel")
            rleng = store.get("rleng")
            parm = store.get("parm")
            flplim = store.get("flplim")
            tvclim = store.get("tvclim")
            cmom = pdynmc * area * rleng * cmdel
            eratey = rateyc - ratey
            delr = -eratey * (1.0 - aptvc) * ga
            if delr > flplim:
                delr = flplim
            if delr < -flplim:
                delr = -flplim
            aermy = -delr * cmom
            if fthalt > 0:
                zeta = -eratey * aptvc * gp
            else:
                zeta = 0.0
            if zeta > tvclim:
                zeta = tvclim
            if zeta < -tvclim:
                zeta = -tvclim
            tvcmy = -zeta * fthalt * parm
            rateyd_new = (aermy + tvcmy) / aiz

        tiy = dvbe * amass / (-pdynmc * area * cybet + fthalt)
        betd_new = -(tiy * ratey + bet) / tiy
        bett = bet
        if bett > betlim:
            bett = betlim
        if bett < -betlim:
            bett = -betlim

        ratey = integrate(rateyd_new, rateyd, ratey, int_step)
        bet = integrate(betd_new, betd, bet, int_step)

        store.set("ratey", ratey)
        store.set("rateyd", rateyd_new)
        store.set("bet", bet)
        store.set("betd", betd_new)
        store.set("aermy", aermy)
        store.set("tvcmy", tvcmy)
        store.set("zeta", zeta)
        store.set("delr", delr)
        return bett
