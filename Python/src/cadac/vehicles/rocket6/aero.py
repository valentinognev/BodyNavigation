from math import cos, sin, sqrt

from cadac.constants import AGRAV, RAD
from cadac.kernel.state import Field


def _cadac_sign(variable):
    if variable < 0:
        return -1
    return 1


class Rocket6Aero:
    name = "aerodynamics"

    def __init__(self, aero_deck):
        self.deck = aero_deck

    def define(self, vehicle):
        store = vehicle.store
        plot = ("plot",)
        scrn = ("scrn",)
        for field in (
            Field("maero", 0, "int", "data", "aerodynamics", scrn),
            Field("refa", 0.0, "real", "init", "aerodynamics"),
            Field("refd", 0.0, "real", "init", "aerodynamics"),
            Field("xcg_ref", 0.0, "real", "init", "aerodynamics"),
            Field("cy", 0.0, "real", "out", "aerodynamics"),
            Field("cll", 0.0, "real", "out", "aerodynamics"),
            Field("clm", 0.0, "real", "out", "aerodynamics"),
            Field("cln", 0.0, "real", "out", "aerodynamics"),
            Field("cx", 0.0, "real", "out", "aerodynamics"),
            Field("cz", 0.0, "real", "out", "aerodynamics"),
            Field("ca0", 0.0, "real", "diag", "aerodynamics"),
            Field("caa", 0.0, "real", "diag", "aerodynamics"),
            Field("cn0", 0.0, "real", "diag", "aerodynamics"),
            Field("clm0", 0.0, "real", "diag", "aerodynamics"),
            Field("clmq", 0.0, "real", "diag", "aerodynamics"),
            Field("cla", 0.0, "real", "out", "aerodynamics"),
            Field("clde", 0.0, "real", "diag", "aerodynamics"),
            Field("cyb", 0.0, "real", "diag", "aerodynamics"),
            Field("cydr", 0.0, "real", "diag", "aerodynamics"),
            Field("cllda", 0.0, "real", "diag", "aerodynamics"),
            Field("cllp", 0.0, "real", "diag", "aerodynamics"),
            Field("cma", 0.0, "real", "diag", "aerodynamics"),
            Field("cmde", 0.0, "real", "diag", "aerodynamics"),
            Field("cmq", 0.0, "real", "diag", "aerodynamics"),
            Field("cnb", 0.0, "real", "diag", "aerodynamics"),
            Field("cndr", 0.0, "real", "diag", "aerodynamics"),
            Field("cnr", 0.0, "real", "diag", "aerodynamics"),
            Field("stmarg_yaw", 0.0, "real", "diag", "aerodynamics", plot),
            Field("stmarg_pitch", 0.0, "real", "diag", "aerodynamics", plot),
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
            Field("realp1", 0.0, "real", "diag", "aerodynamics", plot),
            Field("realp2", 0.0, "real", "diag", "aerodynamics"),
            Field("wnp", 0.0, "real", "diag", "aerodynamics", plot),
            Field("zetp", 0.0, "real", "diag", "aerodynamics", plot),
            Field("rpreal", 0.0, "real", "diag", "aerodynamics"),
            Field("realy1", 0.0, "real", "diag", "aerodynamics", plot),
            Field("realy2", 0.0, "real", "diag", "aerodynamics"),
            Field("wny", 0.0, "real", "diag", "aerodynamics", plot),
            Field("zety", 0.0, "real", "diag", "aerodynamics", plot),
            Field("ryreal", 0.0, "real", "diag", "aerodynamics"),
            Field("trcode", 0.0, "real", "init", "aerodynamics"),
            Field("tmcode", 0.0, "real", "data", "aerodynamics"),
            Field("trmach", 0.0, "real", "data", "aerodynamics"),
            Field("trdynm", 0.0, "real", "data", "aerodynamics"),
            Field("trload", 0.0, "real", "data", "aerodynamics"),
            Field("tralp", 0.0, "real", "data", "aerodynamics"),
            Field("alplimx", 0.0, "real", "data", "aerodynamics"),
            Field("alimitx", 0.0, "real", "data", "aerodynamics"),
            Field("gnavail", 0.0, "real", "diag", "aerodynamics"),
            Field("gyavail", 0.0, "real", "diag", "aerodynamics"),
            Field("gnmax", 0.0, "real", "out", "aerodynamics", plot),
            Field("gymax", 0.0, "real", "out", "aerodynamics", plot),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        store.set("trmach", 0.8)
        store.set("trdynm", 10.0e3)
        store.set("trload", 3.0)
        store.set("tralp", 21.0)
        store.set("trcode", 0.0)
        store.set("tmcode", 0.0)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        maero = store.get("maero")
        if maero == 13:
            slv = 3
        elif maero == 12:
            slv = 2
        elif maero == 11:
            slv = 1
        else:
            raise ValueError(f"unknown maero {maero}")

        look_up = self.deck.look_up
        alplimx = store.get("alplimx")
        alimitx = store.get("alimitx")
        refa = store.get("refa")
        refd = store.get("refd")
        xcg_ref = store.get("xcg_ref")
        cla = store.get("cla")
        cma = store.get("cma")
        alppx = store.get("alppx")
        phipx = store.get("phipx")
        vmach = store.get("vmach")
        pdynmc = store.get("pdynmc")
        dvba = store.get("dvba")
        qqx = store.get("qqx")
        rrx = store.get("rrx")
        mprop = store.get("mprop")
        vmass = store.get("vmass")
        xcg = store.get("xcg")

        phip = phipx * RAD
        cphip = cos(phip)
        sphip = sin(phip)
        qqax = qqx * cphip - rrx * sphip

        ca0 = look_up(f"ca0slv{slv}_vs_mach", vmach)
        caa = look_up(f"caaslv{slv}_vs_mach", vmach)
        ca0b = look_up(f"ca0bslv{slv}_vs_mach", vmach)
        thrust_on = 1.0 if mprop else 0.0
        ca = ca0 + caa * alppx + thrust_on * ca0b

        cn0 = look_up(f"cn0slv{slv}_vs_mach_alpha", vmach, alppx)
        cna = cn0
        clm0 = look_up(f"clm0slv{slv}_vs_mach_alpha", vmach, alppx)
        clmq = look_up(f"clmqslv{slv}_vs_mach", vmach)
        clmaref = clm0 + clmq * qqax * refd / (2.0 * dvba)
        clma = clmaref - cna * (xcg_ref - xcg) / refd

        alplx = alppx + 3.0
        alpmx = alppx - 3.0
        if alpmx < 0.0:
            alpmx = 0.0
        cn0p = look_up(f"cn0slv{slv}_vs_mach_alpha", vmach, alplx)
        cn0m = look_up(f"cn0slv{slv}_vs_mach_alpha", vmach, alpmx)
        if alplx < alplimx:
            cla = (cn0p - cn0m) / (alplx - alpmx)
        clm0p = look_up(f"clm0slv{slv}_vs_mach_alpha", vmach, alplx)
        clm0m = look_up(f"clm0slv{slv}_vs_mach_alpha", vmach, alpmx)
        if alppx < alplimx:
            cma = (clm0p - clm0m) / (alplx - alpmx) - cla * (xcg_ref - xcg) / refd

        cx = -ca
        cy = -cna * sphip
        cz = -cna * cphip
        cll = 0.0
        clm = clma * cphip
        cln = -clma * sphip

        cn0mx = look_up(f"cn0slv{slv}_vs_mach_alpha", vmach, alplimx)
        anlmx = cn0mx * pdynmc * refa
        weight = vmass * AGRAV
        gnmax = anlmx / weight
        if gnmax >= alimitx:
            gnmax = alimitx
        cn = 0.0
        aloadn = cn * pdynmc * refa
        gng = aloadn / weight
        gnavail = gnmax - gng
        gymax = gnmax
        gyavail = gnavail

        clde = 0.0
        cyb = -cla
        cydr = 0.0
        cllda = 0.0
        cllp = 0.0
        cmde = 0.0
        cmq = clmq
        cnb = -cma
        cndr = 0.0
        cnr = clmq

        store.set("refa", refa)
        store.set("cy", cy)
        store.set("cll", cll)
        store.set("clm", clm)
        store.set("cln", cln)
        store.set("cx", cx)
        store.set("cz", cz)
        store.set("gnmax", gnmax)
        store.set("gymax", gymax)
        store.set("cla", cla)
        store.set("clde", clde)
        store.set("cyb", cyb)
        store.set("cydr", cydr)
        store.set("cllda", cllda)
        store.set("cllp", cllp)
        store.set("cma", cma)
        store.set("cmde", cmde)
        store.set("cmq", cmq)
        store.set("cnb", cnb)
        store.set("cndr", cndr)
        store.set("cnr", cnr)
        store.set("ca0", ca0)
        store.set("caa", caa)
        store.set("cn0", cn0)
        store.set("clm0", clm0)
        store.set("clmq", clmq)
        store.set("gnavail", gnavail)
        store.set("gyavail", gyavail)

        self.aerodynamics_der(vehicle)

    def aerodynamics_der(self, vehicle):
        store = vehicle.store
        refa = store.get("refa")
        refd = store.get("refd")
        pdynmc = store.get("pdynmc")
        dvba = store.get("dvba")
        vmass = store.get("vmass")
        xcg = store.get("xcg")
        ibbb = store.get("IBBB")
        names = store.names()
        mtvc = store.get("mtvc") if "mtvc" in names else 0
        gtvc = store.get("gtvc") if "gtvc" in names else 0.0
        parm = store.get("parm") if "parm" in names else 0.0
        thrust = store.get("thrust") if "thrust" in names else 0.0
        cla = store.get("cla")
        clde = store.get("clde")
        cyb = store.get("cyb")
        cydr = store.get("cydr")
        cllda = store.get("cllda")
        cllp = store.get("cllp")
        cma = store.get("cma")
        cmde = store.get("cmde")
        cmq = store.get("cmq")
        cnb = store.get("cnb")
        cndr = store.get("cndr")
        cnr = store.get("cnr")

        ibbb11 = ibbb[0, 0]
        ibbb22 = ibbb[1, 1]
        ibbb33 = ibbb[2, 2]
        duml = (pdynmc * refa / vmass) / RAD
        dla = duml * cla
        dlde = duml * clde
        dumm = pdynmc * refa * refd / ibbb22
        dma = dumm * cma / RAD
        dmq = dumm * (refd / (2 * dvba)) * cmq
        dmde = dumm * cmde / RAD

        dumy = pdynmc * refa / vmass
        dyb = dumy * cyb / RAD
        dydr = dumy * cydr / RAD
        dumn = pdynmc * refa * refd / ibbb33
        dnb = dumn * cnb / RAD
        dnr = dumn * (refd / (2 * dvba)) * cnr
        dndr = dumn * cndr / RAD

        dumll = pdynmc * refa * refd / ibbb11
        dllp = dumll * (refd / (2 * dvba)) * cllp
        dllda = dumll * cllda / RAD

        if mtvc == 1 or mtvc == 2 or mtvc == 3:
            dlde = gtvc * thrust / vmass
            dmde = -(parm - xcg) * gtvc * thrust / ibbb[2, 2]
            dydr = dlde
            dndr = dmde

        stmarg_pitch = 0.0
        if cla:
            stmarg_pitch = -cma / cla
        stmarg_yaw = 0.0
        if cyb:
            stmarg_yaw = -cnb / cyb

        a11 = dmq
        a12 = 0.0
        if dla:
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
        if dyb:
            a12 = dnb / dyb
        else:
            a12 = 0.0
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
        store.set("stmarg_yaw", stmarg_yaw)
        store.set("stmarg_pitch", stmarg_pitch)
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
