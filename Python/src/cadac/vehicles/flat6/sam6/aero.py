from math import fabs, sqrt

from cadac.constants import AGRAV, DEG, RAD
from cadac.kernel.state import Field

SMALL = 1e-7


class Sam6Aero:
    name = "aerodynamics"

    def __init__(self, deck):
        self.deck = deck

    def define(self, vehicle):
        store = vehicle.store
        plot = ("plot",)
        scrn = ("scrn",)
        for field in (
            Field("refl", 0.25, "real", "init", "aerodynamics"),
            Field("refa", 0.0491, "real", "init", "aerodynamics"),
            Field("ca", 0.0, "real", "out", "aerodynamics"),
            Field("cy", 0.0, "real", "out", "aerodynamics"),
            Field("cn", 0.0, "real", "out", "aerodynamics"),
            Field("cll", 0.0, "real", "out", "aerodynamics"),
            Field("clm", 0.0, "real", "out", "aerodynamics"),
            Field("cln", 0.0, "real", "out", "aerodynamics"),
            Field("cyb", 0.0, "real", "save", "aerodynamics", plot),
            Field("clnb", 0.0, "real", "save", "aerodynamics", plot),
            Field("clnr", 0.0, "real", "diag", "aerodynamics"),
            Field("clndr", 0.0, "real", "diag", "aerodynamics"),
            Field("ca0", 0.0, "real", "diag", "aerodynamics"),
            Field("cad", 0.0, "real", "diag", "aerodynamics"),
            Field("cndq", 0.0, "real", "diag", "aerodynamics"),
            Field("clmdq", 0.0, "real", "diag", "aerodynamics"),
            Field("clmq", 0.0, "real", "diag", "aerodynamics"),
            Field("clldp", 0.0, "real", "diag", "aerodynamics"),
            Field("cllp", 0.0, "real", "diag", "aerodynamics"),
            Field("dna", 0.0, "real", "out", "aerodynamics"),
            Field("dnd", 0.0, "real", "out", "aerodynamics"),
            Field("dma", 0.0, "real", "out", "aerodynamics"),
            Field("dmq", 0.0, "real", "out", "aerodynamics"),
            Field("dmd", 0.0, "real", "out", "aerodynamics"),
            Field("dlp", 0.0, "real", "out", "aerodynamics"),
            Field("dld", 0.0, "real", "out", "aerodynamics"),
            Field("cna", 0.0, "real", "save", "aerodynamics", plot),
            Field("clma", 0.0, "real", "save", "aerodynamics", plot),
            Field("dlnd", 0.0, "real", "out", "aerodynamics"),
            Field("stmarg_pitch", 0.0, "real", "save", "aerodynamics", plot),
            Field("realq1", 0.0, "real", "diag", "aerodynamics", scrn),
            Field("realq2", 0.0, "real", "diag", "aerodynamics", scrn),
            Field("wnq", 0.0, "real", "diag", "aerodynamics"),
            Field("zetq", 0.0, "real", "diag", "aerodynamics"),
            Field("realp", 0.0, "real", "diag", "aerodynamics"),
            Field("alplimx", 40.0, "real", "data", "aerodynamics"),
            Field("gavail", 0.0, "real", "diag", "aerodynamics", plot),
            Field("gmax", 0.0, "real", "diag", "aerodynamics", plot),
            Field("pqreal", 0.0, "real", "diag", "aerodynamics"),
            Field("dnr", 0.0, "real", "out", "aerodynamics"),
            Field("dyb", 0.0, "real", "out", "aerodynamics"),
            Field("dnb", 0.0, "real", "out", "aerodynamics"),
            Field("wnr", 0.0, "real", "diag", "aerodynamics"),
            Field("zetr", 0.0, "real", "diag", "aerodynamics"),
            Field("stmarg_yaw", 0.0, "real", "save", "aerodynamics", plot),
            Field("realr1", 0.0, "real", "diag", "aerodynamics"),
            Field("realr2", 0.0, "real", "diag", "aerodynamics"),
            Field("prreal", 0.0, "real", "diag", "aerodynamics"),
            Field("trcond", 0, "int", "diag", "aerodynamics"),
            Field("trortho", 0.0, "real", "data", "aerodynamics"),
            Field("tralp", 0.0, "real", "data", "aerodynamics"),
            Field("trdynm", 0.0, "real", "data", "aerodynamics"),
            Field("trload", 0.0, "real", "data", "aerodynamics"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        store.set("trcond", 0)
        store.set("trortho", 1e-4)
        store.set("tralp", 1.047)
        store.set("trdynm", 1e4)
        store.set("trload", 0.001)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        look_up = self.deck.look_up
        alplimx = store.get("alplimx")
        refl = store.get("refl")
        refa = store.get("refa")
        trcond = store.get("trcond")
        trload = store.get("trload")
        mach = store.get("vmach")
        pdynmc = store.get("pdynmc")
        ppx = store.get("ppx")
        qqx = store.get("qqx")
        rrx = store.get("rrx")
        dvbe = store.get("dvbe")
        mprop = store.get("mprop")
        mass = store.get("mass")
        xcgref = store.get("xcgref")
        xcg = store.get("xcg")
        alimitx = store.get("alimitx")
        dpx = store.get("dpx")
        dqx = store.get("dqx")
        drx = store.get("drx")
        delx1 = store.get("delx1")
        delx2 = store.get("delx2")
        delx3 = store.get("delx3")
        delx4 = store.get("delx4")
        alphax = store.get("alphax")
        betax = store.get("betax")

        deffx = (fabs(delx1) + fabs(delx2) + fabs(delx3) + fabs(delx4)) / 4
        ca0 = look_up("ca0_vs_mach,betax,alphax", mach, betax, alphax)
        cad = look_up("cad_vs_mach", mach)
        ca = ca0 + cad * deffx
        if mprop == 0:
            cab = look_up("cab_vs_mach", mach)
            ca += cab

        cy0 = look_up("cy0_vs_mach,betax,alphax", mach, betax, alphax)
        cydr = look_up("cydr_vs_mach,betax,alphax", mach, betax, alphax)
        cy = cy0 + cydr * drx

        cn0 = look_up("cn0_vs_mach,betax,alphax", mach, betax, alphax)
        cndq = look_up("cndq_vs_mach,betax,alphax", mach, betax, alphax)
        cn = cn0 + cndq * dqx

        cll0 = look_up("cll0_vs_mach,betax,alphax", mach, betax, alphax)
        cllp = look_up("cllp_vs_mach", mach)
        clldp = look_up("clldp_vs_mach,betax,alphax", mach, betax, alphax)
        cll = cll0 + cllp * ppx * RAD * refl / (2 * dvbe) + clldp * dpx

        clm0 = look_up("clm0_vs_mach,betax,alphax", mach, betax, alphax)
        clmq = look_up("clmq_vs_mach", mach)
        clmdq = look_up("clmdq_vs_mach,betax,alphax", mach, betax, alphax)
        clm = clm0 + clmq * qqx * RAD * refl / (2 * dvbe) + clmdq * dqx - cn / refl * (xcgref - xcg)

        cln0 = look_up("cln0_vs_mach,betax,alphax", mach, betax, alphax)
        clnr = look_up("clnr_vs_mach", mach)
        clndr = look_up("clndr_vs_mach,betax,alphax", mach, betax, alphax)
        cln = cln0 + clnr * rrx * RAD * refl / (2 * dvbe) + clndr * drx - cy / refl * (xcgref - xcg)

        cn0mx = look_up("cn0_vs_mach,betax,alphax", mach, 0, alplimx)
        almx = cn0mx * pdynmc * refa
        weight = mass * AGRAV
        gmax = almx / weight
        if gmax >= alimitx:
            gmax = alimitx
        aload = sqrt(cn0 * cn0 + cy0 * cy0) * pdynmc * refa
        gg = aload / weight
        gavail = gmax - gg
        if gavail > alimitx:
            gavail = alimitx
        if gavail < 0:
            gavail = 0
        if gmax < trload:
            trcond = 4

        store.set("ca", ca)
        store.set("cy", cy)
        store.set("cn", cn)
        store.set("cll", cll)
        store.set("clm", clm)
        store.set("cln", cln)
        store.set("gmax", gmax)
        store.set("clnr", clnr)
        store.set("clndr", clndr)
        store.set("ca0", ca0)
        store.set("cad", cad)
        store.set("cndq", cndq)
        store.set("clmdq", clmdq)
        store.set("clmq", clmq)
        store.set("clldp", clldp)
        store.set("cllp", cllp)
        store.set("gavail", gavail)
        store.set("trcond", trcond)
        self.aerodynamics_der(vehicle)

    def aerodynamics_der(self, vehicle):
        store = vehicle.store
        look_up = self.deck.look_up
        alplimx = store.get("alplimx")
        refl = store.get("refl")
        refa = store.get("refa")
        cyb = store.get("cyb")
        clnb = store.get("clnb")
        cna = store.get("cna")
        clma = store.get("clma")
        stmarg_pitch = store.get("stmarg_pitch")
        stmarg_yaw = store.get("stmarg_yaw")
        mach = store.get("vmach")
        pdynmc = store.get("pdynmc")
        alppx = store.get("alppx")
        alphax = store.get("alphax")
        betax = store.get("betax")
        dvbe = store.get("dvbe")
        mass = store.get("mass")
        xcgref = store.get("xcgref")
        xcg = store.get("xcg")
        ai11 = store.get("ai11")
        ai33 = store.get("ai33")
        clnr = store.get("clnr")
        clndr = store.get("clndr")
        cndq = store.get("cndq")
        clmdq = store.get("clmdq")
        clmq = store.get("clmq")
        cllp = store.get("cllp")
        clldp = store.get("clldp")

        if alppx < (alplimx - 3):
            alphapx = fabs(alphax) + 3
            if alphapx < 3:
                alphapx = 3
            alphamx = fabs(alphax) - 3
            if alphamx < 0:
                alphamx = 0
            cn0p = look_up("cn0_vs_mach,betax,alphax", mach, betax, alphapx)
            cn0m = look_up("cn0_vs_mach,betax,alphax", mach, betax, alphamx)
            cna = (cn0p - cn0m) / ((alphapx - alphamx) * RAD)
            clm0p = look_up("clm0_vs_mach,betax,alphax", mach, betax, alphapx)
            clm0m = look_up("clm0_vs_mach,betax,alphax", mach, betax, alphamx)
            clma = (clm0p - clm0m) / ((alphapx - alphamx) * RAD) - cna / refl * (xcgref - xcg)

            betapx = fabs(betax) + 3
            if betapx < 3:
                betapx = 3
            betamx = fabs(betax) - 3
            if betamx < 0:
                betamx = 0
            cy0p = look_up("cy0_vs_mach,betax,alphax", mach, betapx, alphax)
            cy0m = look_up("cy0_vs_mach,betax,alphax", mach, betamx, alphax)
            cyb = (cy0p - cy0m) / ((betapx - betamx) * RAD)
            cln0p = look_up("cln0_vs_mach,betax,alphax", mach, betapx, alphax)
            cln0m = look_up("cln0_vs_mach,betax,alphax", mach, betamx, alphax)
            clnb = (cln0p - cln0m) / ((betapx - betamx) * RAD) - cyb / refl * (xcgref - xcg)

        dna = (pdynmc * refa / mass) * cna
        dma = (pdynmc * refa * refl / ai33) * clma
        dmq = DEG * (pdynmc * refa * refl / ai33) * (refl / (2 * dvbe)) * clmq
        dmd = DEG * (pdynmc * refa * refl / ai33) * clmdq
        dyb = (pdynmc * refa / mass) * cyb
        dnb = (pdynmc * refa * refl / ai33) * clnb
        dnr = DEG * (pdynmc * refa * refl / ai33) * (refl / (2 * dvbe)) * clnr
        dnd = (pdynmc * refa / mass) * cndq
        dlnd = DEG * (pdynmc * refa * refl / ai33) * clndr
        dlp = DEG * (pdynmc * refa * refl / ai11) * (refl / (2 * dvbe)) * cllp
        dld = DEG * (pdynmc * refa * refl / ai11) * clldp

        wnq = 0.0
        zetq = 0.0
        realq1 = 0.0
        realq2 = 0.0
        pqreal = 0.0
        if fabs(dna) >= SMALL:
            stmarg_pitch = -(dma / dna) * (ai33 / (refl * mass))
            a11 = dmq
            a12 = dma / dna
            a21 = dna
            a22 = -dna / dvbe
            arg = pow((a11 + a22), 2) - 4 * (a11 * a22 - a12 * a21)
            if arg >= 0:
                wnq = 0
                zetq = 0
                dum = a11 + a22
                realq1 = (dum + sqrt(arg)) / 2
                realq2 = (dum - sqrt(arg)) / 2
                pqreal = (realq1 + realq2) / 2
            else:
                realq1 = 0
                realq2 = 0
                wnq = sqrt(a11 * a22 - a12 * a21)
                zetq = -(a11 + a22) / (2 * wnq)
                pqreal = -zetq * wnq

        wnr = 0.0
        zetr = 0.0
        realr1 = 0.0
        realr2 = 0.0
        prreal = 0.0
        if fabs(dyb) >= SMALL:
            stmarg_yaw = -(dnb / dyb) * (ai33 / (refl * mass))
            a11 = dnr
            a12 = dnb / dyb
            a21 = -dyb
            a22 = dyb / dvbe
            arg = pow((a11 + a22), 2) - 4 * (a11 * a22 - a12 * a21)
            if arg >= 0:
                wnr = 0
                zetr = 0
                dum = a11 + a22
                realr1 = (dum + sqrt(arg)) / 2
                realr2 = (dum - sqrt(arg)) / 2
                prreal = (realr1 + realr2) / 2
            else:
                realr1 = 0
                realr2 = 0
                wnr = sqrt(a11 * a22 - a12 * a21)
                zetr = -(a11 + a22) / (2 * wnr)
                prreal = -zetr * wnr

        realp = dlp
        store.set("dna", dna)
        store.set("dnd", dnd)
        store.set("dma", dma)
        store.set("dmq", dmq)
        store.set("dmd", dmd)
        store.set("dlp", dlp)
        store.set("dld", dld)
        store.set("dlnd", dlnd)
        store.set("dnr", dnr)
        store.set("dyb", dyb)
        store.set("dnb", dnb)
        store.set("cyb", cyb)
        store.set("clnb", clnb)
        store.set("cna", cna)
        store.set("clma", clma)
        store.set("stmarg_pitch", stmarg_pitch)
        store.set("stmarg_yaw", stmarg_yaw)
        store.set("realq1", realq1)
        store.set("realq2", realq2)
        store.set("wnq", wnq)
        store.set("zetq", zetq)
        store.set("realp", realp)
        store.set("pqreal", pqreal)
        store.set("wnr", wnr)
        store.set("zetr", zetr)
        store.set("realr1", realr1)
        store.set("realr2", realr2)
        store.set("prreal", prreal)

    def terminate(self, vehicle, ctx):
        pass
