"""SRAAM5 guidance — Fortran MODULE.FOR C1 / C1MID / C1TERM."""

from math import atan2, cos, fabs, sin, sqrt

import numpy as np

from cadac.constants import AGRAV, DEG
from cadac.kernel.state import Field
from cadac.math.frames import cart_from_pol, polar_from_cart, skew

_ZEROS3 = (0.0, 0.0, 0.0)


class Sraam5Guidance:
    name = "guidance"

    def define(self, vehicle):
        store = vehicle.store
        plot = ("plot",)
        for field in (
            Field("mguid", 0, "int", "data", "guidance"),
            Field("gnav", 0.0, "real", "data", "guidance"),
            Field("ancomx", 0.0, "real", "out", "guidance"),
            Field("alcomx", 0.0, "real", "out", "guidance"),
            Field("epchta", 0.0, "real", "save", "guidance"),
            Field("ST1ELM", _ZEROS3, "vec", "save", "guidance"),
            Field("VT1ELC", _ZEROS3, "vec", "save", "guidance"),
            Field("WOELC", _ZEROS3, "vec", "out", "guidance"),
            Field("UT1BLC", _ZEROS3, "vec", "out", "guidance"),
            Field("tgoc", 0.0, "real", "diag", "guidance"),
            Field("dt1bc", 0.0, "real", "diag", "guidance", plot),
            Field("dvt1bc", 0.0, "real", "diag", "guidance", plot),
            Field("psiobcx", 0.0, "real", "diag", "guidance", plot),
            Field("thtobcx", 0.0, "real", "diag", "guidance", plot),
            Field("ST1ELC", _ZEROS3, "vec", "diag", "guidance"),
            Field("ST1BLC", _ZEROS3, "vec", "diag", "guidance"),
            Field("gn", 0.0, "real", "diag", "guidance"),
            Field("apny", 0.0, "real", "diag", "guidance"),
            Field("apnz", 0.0, "real", "diag", "guidance"),
            Field("adely", 0.0, "real", "diag", "guidance"),
            Field("adelz", 0.0, "real", "diag", "guidance"),
            Field("all", 0.0, "real", "diag", "guidance", plot),
            Field("ann", 0.0, "real", "diag", "guidance", plot),
        ):
            if field.name not in store:
                store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mguid = int(store.get("mguid"))
        mnav = int(store.get("mnav"))
        epchta = float(store.get("epchta"))
        st1elm = np.asarray(store.get("ST1ELM"), dtype=float).copy()
        vt1elc = np.asarray(store.get("VT1ELC"), dtype=float).copy()
        time = float(ctx.sim_time)

        # Target data receipt (MNAV=3) and extrapolation — Fortran C1
        if mnav == 3:
            mnav = 0
            epchta = time
            st1elm = np.asarray(store.get("ST1CEL"), dtype=float).copy()
            vt1elc = np.asarray(store.get("VT1CEL"), dtype=float).copy()

        dtimex = time - epchta
        st1elc = st1elm + vt1elc * dtimex
        sbelc = np.asarray(store.get("SBELC"), dtype=float)
        st1blc = st1elc - sbelc

        store.set("mnav", mnav)
        store.set("epchta", epchta)
        store.set("ST1ELM", st1elm)
        store.set("VT1ELC", vt1elc)
        store.set("ST1ELC", st1elc)
        store.set("ST1BLC", st1blc)

        if mguid == 0:
            return
        if mguid == 3:
            self._c1_mid(vehicle, st1blc, vt1elc)
            return
        if mguid == 6:
            self._c1_term(vehicle)
            return
        raise ValueError(f"unknown mguid {mguid}")

    def _c1_mid(self, vehicle, st1blc, vt1elc):
        """Fortran C1 midcourse kinematics + C1MID pro-nav."""
        store = vehicle.store
        gnav = float(store.get("gnav"))
        tblc = np.asarray(store.get("TBLC"), dtype=float)
        vbelc = np.asarray(store.get("VBELC"), dtype=float)
        st1blc = np.asarray(st1blc, dtype=float)
        vt1elc = np.asarray(vt1elc, dtype=float)

        dt1bc = float(np.linalg.norm(st1blc))
        ut1blc = st1blc * (1.0 / dt1bc)
        ut1bbc = tblc @ ut1blc
        polar = polar_from_cart(ut1bbc)
        psiobc = float(polar[1])
        thtobc = float(polar[2])
        psiobcx = psiobc * DEG
        thtobcx = thtobc * DEG

        vt1blc = vt1elc - vbelc
        dvt1bc = fabs(float(ut1blc @ vt1blc))
        tgoc = dt1bc / dvt1bc
        woelc = skew(ut1blc) @ vt1blc * (1.0 / dt1bc)

        # C1MID: reconstruct LOS unit from polar angles, then pro-nav
        uobb = cart_from_pol(1.0, psiobc, thtobc)
        uobl = tblc.T @ uobb
        apnl = skew(woelc) @ uobl * (gnav * dvt1bc)
        aapnb = tblc @ apnl
        ancomx = -float(aapnb[2]) / AGRAV
        alcomx = float(aapnb[1]) / AGRAV

        store.set("ancomx", ancomx)
        store.set("alcomx", alcomx)
        store.set("WOELC", woelc)
        store.set("UT1BLC", ut1blc)
        store.set("tgoc", tgoc)
        store.set("dt1bc", dt1bc)
        store.set("dvt1bc", dvt1bc)
        store.set("psiobcx", psiobcx)
        store.set("thtobcx", thtobcx)

    def _c1_term(self, vehicle):
        """Fortran C1TERM — LOS-rate pro-nav + circular g-limiter."""
        store = vehicle.store
        gnav = float(store.get("gnav"))
        st1el = np.asarray(store.get("ST1EL"), dtype=float)
        vt1el = np.asarray(store.get("VT1EL"), dtype=float)
        thtpb = float(store.get("thtpb"))
        psipb = float(store.get("psipb"))
        sigdpy = float(store.get("sigdpy"))
        sigdpz = float(store.get("sigdpz"))
        fspcb = np.asarray(store.get("FSPCB"), dtype=float)
        gmax = float(store.get("gmax"))
        trcode = float(store.get("trcode"))
        trcvel = float(store.get("trcvel"))
        sbel = np.asarray(store.get("SBEL"), dtype=float)
        vbel = np.asarray(store.get("VBEL"), dtype=float)

        sbt1l = sbel - st1el
        dbt1 = float(np.linalg.norm(sbt1l))
        vbt1l = vbel - vt1el
        dum = float(sbt1l @ vbt1l)
        dcvel = fabs(dum / dbt1)

        if dbt1 <= 1000.0 and dcvel < trcvel:
            trcode = 1.0

        fspcb1 = float(fspcb[0])
        adely = sin(psipb) * fspcb1
        adelz = sin(thtpb) * cos(psipb) * fspcb1
        gn = gnav * dcvel
        apny = gn * sigdpz
        apnz = gn * sigdpy
        cththb = fabs(cos(thtpb) * cos(psipb))
        all_ = (apny + adely) / (cththb * AGRAV)
        ann = (apnz + adelz) / (cththb * AGRAV)

        aa = sqrt(all_ * all_ + ann * ann)
        if aa > gmax:
            aa = gmax
        if max(fabs(ann), fabs(all_)) < 1.0e-10:
            phi = 0.0
        else:
            phi = atan2(ann, all_)
        alcomx = aa * cos(phi)
        ancomx = aa * sin(phi)

        store.set("ancomx", ancomx)
        store.set("alcomx", alcomx)
        store.set("trcode", trcode)
        store.set("gn", gn)
        store.set("apny", apny)
        store.set("apnz", apnz)
        store.set("adely", adely)
        store.set("adelz", adelz)
        store.set("all", all_)
        store.set("ann", ann)

    def terminate(self, vehicle, ctx):
        pass
