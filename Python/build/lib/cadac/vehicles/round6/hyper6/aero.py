from math import cos, fabs, sin, sqrt

from cadac.constants import AGRAV, RAD
from cadac.kernel.state import Field


class Hyper6Aero:
    name = "aerodynamics"

    def __init__(self, deck):
        self.deck = deck

    def define(self, vehicle):
        store = vehicle.store
        plot = ("plot",)
        for field in (
            Field("maero", 0, "int", "data", "aerodynamics"),
            Field("refa", 0.0, "real", "init", "aerodynamics"),
            Field("refb", 0.0, "real", "init", "aerodynamics"),
            Field("refc", 0.0, "real", "init", "aerodynamics"),
            Field("cd", 0.0, "real", "dia", "aerodynamics"),
            Field("cl", 0.0, "real", "dia", "aerodynamics"),
            Field("cy", 0.0, "real", "out", "aerodynamics"),
            Field("cll", 0.0, "real", "out", "aerodynamics"),
            Field("clm", 0.0, "real", "out", "aerodynamics"),
            Field("cln", 0.0, "real", "out", "aerodynamics"),
            Field("cx", 0.0, "real", "out", "aerodynamics"),
            Field("cz", 0.0, "real", "out", "aerodynamics"),
            Field("cd0", 0.0, "real", "diag", "aerodynamics"),
            Field("cda", 0.0, "real", "diag", "aerodynamics"),
            Field("cl0", 0.0, "real", "diag", "aerodynamics"),
            Field("cla", 0.0, "real", "diag", "aerodynamics"),
            Field("clde", 0.0, "real", "diag", "aerodynamics"),
            Field("cyb", 0.0, "real", "diag", "aerodynamics"),
            Field("cyda", 0.0, "real", "diag", "aerodynamics"),
            Field("cydr", 0.0, "real", "diag", "aerodynamics"),
            Field("cllb", 0.0, "real", "diag", "aerodynamics"),
            Field("cllda", 0.0, "real", "diag", "aerodynamics"),
            Field("cllp", 0.0, "real", "diag", "aerodynamics"),
            Field("cllr", 0.0, "real", "diag", "aerodynamics"),
            Field("cm0", 0.0, "real", "diag", "aerodynamics"),
            Field("cma", 0.0, "real", "diag", "aerodynamics"),
            Field("cmde", 0.0, "real", "diag", "aerodynamics"),
            Field("cmq", 0.0, "real", "diag", "aerodynamics"),
            Field("clnb", 0.0, "real", "diag", "aerodynamics"),
            Field("clnda", 0.0, "real", "diag", "aerodynamics"),
            Field("clndr", 0.0, "real", "diag", "aerodynamics"),
            Field("clnp", 0.0, "real", "diag", "aerodynamics"),
            Field("clnr", 0.0, "real", "diag", "aerodynamics"),
            Field("clldr", 0.0, "real", "diag", "aerodynamics"),
            Field("clovercd", 0.0, "real", "diag", "aerodynamics"),
            Field("stmarg", 0.0, "real", "diag", "aerodynamics", plot),
            Field("dla", 0.0, "real", "out", "aerodynamics"),
            Field("dlde", 0.0, "real", "out", "aerodynamics"),
            Field("dma", 0.0, "real", "out", "aerodynamics"),
            Field("dmq", 0.0, "real", "out", "aerodynamics"),
            Field("dmde", 0.0, "real", "out", "aerodynamics"),
            Field("dyb", 0.0, "real", "out", "aerodynamics"),
            Field("dydr", 0.0, "real", "out", "aerodynamics"),
            Field("dnb", 0.0, "real", "out", "aerodynamics"),
            Field("dnr", 0.0, "real", "out", "aerodynamics"),
            Field("dndr", 0.0, "real", "out", "aerodynamics"),
            Field("dllp", 0.0, "real", "out", "aerodynamics"),
            Field("dllda", 0.0, "real", "out", "aerodynamics"),
            Field("alpplimx", 0.0, "real", "data", "aerodynamics"),
            Field("strct_pos_limitx", 0.0, "real", "data", "aerodynamics"),
            Field("gavail_pos", 0.0, "real", "diag", "aerodynamics", plot),
            Field("gmax", 0.0, "real", "out", "aerodynamics", plot),
            Field("alpnlimx", 0.0, "real", "data", "aerodynamics"),
            Field("strct_neg_limitx", 0.0, "real", "data", "aerodynamics"),
            Field("gavail_neg", 0.0, "real", "diag", "aerodynamics", plot),
            Field("gminx", 0.0, "real", "out", "aerodynamics", plot),
            Field("realp1", 0.0, "real", "diag", "aerodynamics"),
            Field("realp2", 0.0, "real", "diag", "aerodynamics"),
            Field("wnp", 0.0, "real", "diag", "aerodynamics", plot),
            Field("zetp", 0.0, "real", "diag", "aerodynamics"),
            Field("rpreal", 0.0, "real", "diag", "aerodynamics"),
            Field("realy1", 0.0, "real", "diag", "aerodynamics"),
            Field("realy2", 0.0, "real", "diag", "aerodynamics"),
            Field("wny", 0.0, "real", "diag", "aerodynamics"),
            Field("zety", 0.0, "real", "diag", "aerodynamics"),
            Field("ryreal", 0.0, "real", "diag", "aerodynamics"),
            Field("trcode", 0.0, "real", "init", "aerodynamics"),
            Field("tmcode", 0.0, "real", "data", "aerodynamics"),
            Field("trmach", 0.0, "real", "data", "aerodynamics"),
            Field("trdynm", 0.0, "real", "data", "aerodynamics"),
            Field("trload", 0.0, "real", "data", "aerodynamics"),
            Field("tralp", 0.0, "real", "data", "aerodynamics"),
            Field("refa_st", 0.0, "real", "init", "aerodynamics"),
            Field("caa", 0.0, "real", "init", "aerodynamics"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        store.set("refa", 557.42)
        store.set("refb", 24.38)
        store.set("refc", 22.86)
        store.set("trmach", 0.8)
        store.set("trdynm", 10.0e3)
        store.set("trload", 3.0)
        store.set("tralp", 21.0)
        store.set("trcode", 0.0)
        store.set("tmcode", 0.0)
        store.set("refa_st", 7.0)
        store.set("caa", 0.4)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        maero = store.get("maero")
        if maero != 1:
            raise ValueError(f"unknown maero {maero}")

        look_up = self.deck.look_up
        alpplimx = store.get("alpplimx")
        strct_pos_limitx = store.get("strct_pos_limitx")
        alpnlimx = store.get("alpnlimx")
        strct_neg_limitx = store.get("strct_neg_limitx")
        refa = store.get("refa")
        refb = store.get("refb")
        refc = store.get("refc")
        alphax = store.get("alphax")
        betax = store.get("betax")
        vmach = store.get("vmach")
        pdynmc = store.get("pdynmc")
        dvba = store.get("dvba")
        ppx = store.get("ppx")
        qqx = store.get("qqx")
        rrx = store.get("rrx")
        vmass = store.get("vmass")
        delax = store.get("delax")
        delex = store.get("delex")
        delrx = store.get("delrx")

        cd0 = look_up("cd0_vs_alpha_mach", alphax, vmach)
        cda = look_up("cda_vs_alpha_mach", alphax, vmach)
        cd = cd0 + cda * alphax

        cl0 = look_up("cl0_vs_alpha_mach", alphax, vmach)
        cla = look_up("cla_vs_alpha_mach", alphax, vmach)
        clde = look_up("clde_vs_alpha_mach", alphax, vmach)
        cl = cl0 + cla * alphax + clde * delex
        clovercd = fabs(cl / cd)

        cyb = look_up("cyb_vs_alpha_mach", alphax, vmach)
        cyda = look_up("cyda_vs_alpha_mach", alphax, vmach)
        cydr = look_up("cydr_vs_alpha_mach", alphax, vmach)
        cy = cyb * betax + cyda * delax + cydr * delrx

        cllb = look_up("cllb_vs_alpha_mach", alphax, vmach)
        cllda = look_up("cllda_vs_alpha_mach", alphax, vmach)
        clldr = look_up("clldr_vs_alpha_mach", alphax, vmach)
        cllp = look_up("cllp_vs_alpha_mach", alphax, vmach)
        cllr = look_up("cllr_vs_alpha_mach", alphax, vmach)
        cll = (
            cllb * betax
            + cllda * delax
            + clldr * delrx
            + cllp * ppx * RAD * refb / (2 * dvba)
            + cllr * rrx * RAD * refb / (2 * dvba)
        )

        cm0 = look_up("cm0_vs_alpha_mach", alphax, vmach)
        cma = look_up("cma_vs_alpha_mach", alphax, vmach)
        cmde = look_up("cmde_vs_alpha_mach", alphax, vmach)
        cmq = look_up("cmq_vs_alpha_mach", alphax, vmach)
        clm = cm0 + cma * alphax + cmde * delex + cmq * qqx * RAD * refc / (2.0 * dvba)

        clnb = look_up("clnb_vs_alpha_mach", alphax, vmach)
        clnda = look_up("clnda_vs_alpha_mach", alphax, vmach)
        clndr = look_up("clndr_vs_alpha_mach", alphax, vmach)
        clnp = look_up("clnp_vs_alpha_mach", alphax, vmach)
        clnr = look_up("clnr_vs_alpha_mach", alphax, vmach)
        cln = (
            clnb * betax
            + clnda * delax
            + clndr * delrx
            + clnp * ppx * RAD * refb / (2.0 * dvba)
            + clnr * rrx * RAD * refb / (2.0 * dvba)
        )

        cosa = cos(alphax * RAD)
        sina = sin(alphax * RAD)
        cx = -cd * cosa + cl * sina
        cz = -cd * sina - cl * cosa

        cl0max = look_up("cl0_vs_alpha_mach", alpplimx, vmach)
        clamax = look_up("cla_vs_alpha_mach", alpplimx, vmach)
        clmax = cl0max + clamax * alpplimx
        almax = clmax * pdynmc * refa
        weight = vmass * AGRAV
        gmax = almax / weight
        if gmax >= strct_pos_limitx:
            gmax = strct_pos_limitx
        aload = cl * pdynmc * refa
        gg = aload / weight
        gavail_pos = gmax - gg

        cl0min = look_up("cl0_vs_alpha_mach", alpnlimx, vmach)
        clamin = look_up("cla_vs_alpha_mach", alpnlimx, vmach)
        clmin = cl0min + clamin * alpnlimx
        almin = clmin * pdynmc * refa
        gminx = almin / weight
        if gminx <= strct_neg_limitx:
            gminx = strct_neg_limitx
        gavail_neg = gminx - gg

        store.set("refa", refa)
        store.set("cy", cy)
        store.set("cll", cll)
        store.set("clm", clm)
        store.set("cln", cln)
        store.set("cx", cx)
        store.set("cz", cz)
        store.set("gmax", gmax)
        store.set("gminx", gminx)
        store.set("cd", cd)
        store.set("cl", cl)
        store.set("cd0", cd0)
        store.set("cda", cda)
        store.set("cl0", cl0)
        store.set("cla", cla)
        store.set("clde", clde)
        store.set("cyb", cyb)
        store.set("cyda", cyda)
        store.set("cydr", cydr)
        store.set("cllb", cllb)
        store.set("cllda", cllda)
        store.set("cllp", cllp)
        store.set("cllr", cllr)
        store.set("cm0", cm0)
        store.set("cma", cma)
        store.set("cmde", cmde)
        store.set("cmq", cmq)
        store.set("clnb", clnb)
        store.set("clnda", clnda)
        store.set("clndr", clndr)
        store.set("clnp", clnp)
        store.set("clnr", clnr)
        store.set("clldr", clldr)
        store.set("clovercd", clovercd)
        store.set("gavail_pos", gavail_pos)
        store.set("gavail_neg", gavail_neg)

        self.aerodynamics_der(vehicle)

    def aerodynamics_der(self, vehicle):
        store = vehicle.store
        refa = store.get("refa")
        refb = store.get("refb")
        refc = store.get("refc")
        pdynmc = store.get("pdynmc")
        dvba = store.get("dvba")
        vmass = store.get("vmass")
        ibbb = store.get("IBBB")
        cla = store.get("cla")
        clde = store.get("clde")
        cyb = store.get("cyb")
        cydr = store.get("cydr")
        cllda = store.get("cllda")
        cllp = store.get("cllp")
        cma = store.get("cma")
        cmde = store.get("cmde")
        cmq = store.get("cmq")
        clnb = store.get("clnb")
        clndr = store.get("clndr")
        clnr = store.get("clnr")

        ibbb11 = ibbb[0, 0]
        ibbb22 = ibbb[1, 1]
        ibbb33 = ibbb[2, 2]
        duml = (pdynmc * refa / vmass) / RAD
        dla = duml * cla
        dlde = duml * clde
        dumm = pdynmc * refa * refc / ibbb22
        dma = dumm * cma / RAD
        dmq = dumm * (refc / (2.0 * dvba)) * cmq
        dmde = dumm * cmde / RAD

        dumy = pdynmc * refa / vmass
        dyb = dumy * cyb / RAD
        dydr = dumy * cydr / RAD
        dumn = pdynmc * refa * refb / ibbb33
        dnb = dumn * clnb / RAD
        dnr = dumn * (refb / (2.0 * dvba)) * clnr
        dndr = dumn * clndr / RAD

        dumll = pdynmc * refa * refb / ibbb11
        dllp = dumll * (refb / (2.0 * dvba)) * cllp
        dllda = dumll * cllda / RAD

        stmarg = 0.0
        if cla:
            stmarg = -cma / cla

        a11 = dmq
        a12 = dma / dla
        a21 = dla
        a22 = -dla / dvba
        arg = (a11 + a22) ** 2 - 4.0 * (a11 * a22 - a12 * a21)
        if arg >= 0.0:
            wnp = 0.0
            zetp = 0.0
            dum = a11 + a22
            realp1 = (dum + sqrt(arg)) / 2.0
            realp2 = (dum - sqrt(arg)) / 2.0
            rpreal = (realp1 + realp2) / 2.0
        else:
            realp1 = 0.0
            realp2 = 0.0
            wnp = sqrt(a11 * a22 - a12 * a21)
            zetp = -(a11 + a22) / (2.0 * wnp)
            rpreal = -zetp * wnp

        a11 = dnr
        a12 = dnb / dyb
        a21 = -dyb
        a22 = dyb / dvba
        arg = (a11 + a22) ** 2 - 4.0 * (a11 * a22 - a12 * a21)
        if arg >= 0.0:
            wny = 0.0
            zety = 0.0
            dum = a11 + a22
            realy1 = (dum + sqrt(arg)) / 2.0
            realy2 = (dum - sqrt(arg)) / 2.0
            ryreal = (realy1 + realy2) / 2.0
        else:
            realy1 = 0.0
            realy2 = 0.0
            wny = sqrt(a11 * a22 - a12 * a21)
            zety = -(a11 + a22) / (2.0 * wny)
            ryreal = -zety * wny

        store.set("dla", dla)
        store.set("dlde", dlde)
        store.set("dma", dma)
        store.set("dmq", dmq)
        store.set("dmde", dmde)
        store.set("dyb", dyb)
        store.set("dydr", dydr)
        store.set("dnb", dnb)
        store.set("dnr", dnr)
        store.set("dndr", dndr)
        store.set("dllp", dllp)
        store.set("dllda", dllda)
        store.set("stmarg", stmarg)
        store.set("realp1", realp1)
        store.set("realp2", realp2)
        store.set("wnp", wnp)
        store.set("zetp", zetp)
        store.set("rpreal", rpreal)
        store.set("realy1", realy1)
        store.set("realy2", realy2)
        store.set("wny", wny)
        store.set("zety", zety)
        store.set("ryreal", ryreal)

    def terminate(self, vehicle, ctx):
        pass
