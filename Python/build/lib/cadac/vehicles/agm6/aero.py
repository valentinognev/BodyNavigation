from math import cos, fabs, sin, sqrt

from cadac.constants import AGRAV, DEG
from cadac.kernel.state import Field


def _optional(store, name, default=0.0):
    if name in store:
        return store.get(name)
    return default


class Agm6Aero:
    name = "aerodynamics"

    def __init__(self, deck):
        self.deck = deck

    def define(self, vehicle):
        store = vehicle.store
        plot = ("plot",)
        for field in (
            Field("refl", 0.0, "real", "init", "aerodynamics"),
            Field("refa", 0.0, "real", "init", "aerodynamics"),
            Field("ca", 0.0, "real", "out", "aerodynamics"),
            Field("cy", 0.0, "real", "out", "aerodynamics"),
            Field("cn", 0.0, "real", "out", "aerodynamics"),
            Field("cll", 0.0, "real", "out", "aerodynamics"),
            Field("clm", 0.0, "real", "out", "aerodynamics"),
            Field("cln", 0.0, "real", "out", "aerodynamics"),
            Field("cn0", 0.0, "real", "diag", "aerodynamics"),
            Field("cnp", 0.0, "real", "diag", "aerodynamics"),
            Field("clm0", 0.0, "real", "diag", "aerodynamics"),
            Field("clmp", 0.0, "real", "diag", "aerodynamics"),
            Field("cyp", 0.0, "real", "diag", "aerodynamics"),
            Field("clnp", 0.0, "real", "diag", "aerodynamics"),
            Field("ca0", 0.0, "real", "diag", "aerodynamics"),
            Field("caa", 0.0, "real", "diag", "aerodynamics"),
            Field("cad", 0.0, "real", "diag", "aerodynamics"),
            Field("cndq", 0.0, "real", "diag", "aerodynamics"),
            Field("clmdq", 0.0, "real", "diag", "aerodynamics"),
            Field("clmq", 0.0, "real", "diag", "aerodynamics"),
            Field("cllap", 0.0, "real", "diag", "aerodynamics"),
            Field("clldp", 0.0, "real", "diag", "aerodynamics"),
            Field("cllp", 0.0, "real", "diag", "aerodynamics"),
            Field("dna", 0.0, "real", "out", "aerodynamics"),
            Field("dnd", 0.0, "real", "out", "aerodynamics"),
            Field("dma", 0.0, "real", "out", "aerodynamics"),
            Field("dmq", 0.0, "real", "out", "aerodynamics"),
            Field("dmd", 0.0, "real", "out", "aerodynamics"),
            Field("dlp", 0.0, "real", "out", "aerodynamics"),
            Field("dld", 0.0, "real", "out", "aerodynamics"),
            Field("cna", 0.0, "real", "diag", "aerodynamics"),
            Field("cya", 0.0, "real", "diag", "aerodynamics"),
            Field("clma", 0.0, "real", "diag", "aerodynamics"),
            Field("clna", 0.0, "real", "diag", "aerodynamics"),
            Field("stmarg", 0.0, "real", "diag", "aerodynamics", plot),
            Field("realq1", 0.0, "real", "diag", "aerodynamics", plot),
            Field("realq2", 0.0, "real", "diag", "aerodynamics", plot),
            Field("wnq", 0.0, "real", "diag", "aerodynamics", plot),
            Field("zetq", 0.0, "real", "diag", "aerodynamics", plot),
            Field("realp", 0.0, "real", "diag", "aerodynamics"),
            Field("alplimx", 0.0, "real", "data", "aerodynamics"),
            Field("gavail", 0.0, "real", "diag", "aerodynamics", plot),
            Field("gmax", 0.0, "real", "diag", "aerodynamics", plot),
            Field("pqreal", 0.0, "real", "diag", "aerodynamics"),
            Field("trcond", 0, "int", "init", "aerodynamics", plot),
            Field("tmcode", 0.0, "real", "data", "aerodynamics"),
            Field("trcvel", 0.0, "real", "data", "aerodynamics"),
            Field("trmach", 0.0, "real", "data", "aerodynamics"),
            Field("trdynm", 0.0, "real", "data", "aerodynamics"),
            Field("trload", 0.0, "real", "data", "aerodynamics"),
            Field("tralp", 0.0, "real", "data", "aerodynamics"),
            Field("trtht", 0.0, "real", "data", "aerodynamics"),
            Field("trthtd", 0.0, "real", "data", "aerodynamics"),
            Field("trphid", 0.0, "real", "data", "aerodynamics"),
            Field("trate", 0.0, "real", "data", "aerodynamics"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        store.set("refl", 0.5)
        store.set("refa", 0.196)
        store.set("trcvel", 10e-5)
        store.set("trmach", 0.4)
        store.set("trdynm", 10e3)
        store.set("trload", 0.5)
        store.set("tralp", 1)
        store.set("trcond", 0)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        look_up = self.deck.look_up
        alplimx = store.get("alplimx")
        refl = store.get("refl")
        refa = store.get("refa")
        trcond = store.get("trcond")
        trload = store.get("trload")
        vmach = store.get("vmach")
        pdynmc = store.get("pdynmc")
        alppx = store.get("alppx")
        phip = store.get("phip")
        ppx = store.get("ppx")
        qqx = store.get("qqx")
        rrx = store.get("rrx")
        dvba = store.get("dvba")
        vmass = store.get("vmass")
        alimit = _optional(store, "alimit")
        dpx = _optional(store, "dpx")
        dqx = _optional(store, "dqx")
        drx = _optional(store, "drx")

        cphip = cos(phip)
        sphip = sin(phip)
        dqax = dqx * cphip - drx * sphip
        drax = dqx * sphip + drx * cphip
        qqax = qqx * cphip - rrx * sphip
        rrax = qqx * sphip + rrx * cphip

        ca0 = look_up("ca0_vs_mach", vmach)
        caa = look_up("caa_vs_mach", vmach)
        cad = look_up("cad_vs_mach", vmach)
        deff = (fabs(dqax) + fabs(drax)) / 2
        ca = ca0 + caa * alppx + cad * deff * deff

        cyp = look_up("cyp_vs_mach_alpha", vmach, alppx)
        cydr = look_up("cndq_vs_mach", vmach)
        s4phi = sin(4 * phip)
        cya = cyp * s4phi + cydr * drax

        cn0 = look_up("cn0_vs_mach_alpha", vmach, alppx)
        cnp = look_up("cnp_vs_mach_alpha", vmach, alppx)
        cndq = look_up("cndq_vs_mach", vmach)
        s2phi = sin(2 * phip) ** 2
        cna = cn0 + cnp * s2phi + cndq * dqax

        cllap = look_up("cllap_vs_mach", vmach)
        cllp = look_up("cllp_vs_mach", vmach)
        clldp = look_up("clldp_vs_mach", vmach)
        cll = cllap * alppx * alppx * s4phi + cllp * ppx * refl / (2 * dvba) + clldp * dpx

        clm0 = look_up("clm0_vs_mach_alpha", vmach, alppx)
        clmp = look_up("clmp_vs_mach_alpha", vmach, alppx)
        clmq = look_up("clmq_vs_mach", vmach)
        clmdq = look_up("clmdq_vs_mach", vmach)
        clma = clm0 + clmp * s2phi + clmq * qqax * refl / (2 * dvba) + clmdq * dqax

        clnp = look_up("clnp_vs_mach_alpha", vmach, alppx)
        clnr = clmq
        clndr = clmdq
        clna = clnp * s4phi + clnr * rrax * refl / (2 * dvba) + clndr * drax

        cy = cya * cphip - cna * sphip
        cn = cya * sphip + cna * cphip
        clm = clma * cphip + clna * sphip
        cln = clna * cphip - clma * sphip

        cn0mx = look_up("cn0_vs_mach_alpha", vmach, alplimx)
        almx = cn0mx * pdynmc * refa
        weight = vmass * AGRAV
        gmax = almx / weight
        if gmax >= alimit:
            gmax = alimit
        aload = cn0 * pdynmc * refa
        gg = aload / weight
        gavail = gmax - gg
        if gmax < trload:
            trcond = 4

        store.set("ca", ca)
        store.set("cy", cy)
        store.set("cn", cn)
        store.set("cll", cll)
        store.set("clm", clm)
        store.set("cln", cln)
        store.set("gmax", gmax)
        store.set("trcond", trcond)
        store.set("cn0", cn0)
        store.set("cnp", cnp)
        store.set("clm0", clm0)
        store.set("clmp", clmp)
        store.set("cyp", cyp)
        store.set("clnp", clnp)
        store.set("ca0", ca0)
        store.set("caa", caa)
        store.set("cad", cad)
        store.set("cndq", cndq)
        store.set("clmdq", clmdq)
        store.set("clmq", clmq)
        store.set("cllap", cllap)
        store.set("clldp", clldp)
        store.set("cllp", cllp)
        store.set("cna", cna)
        store.set("cya", cya)
        store.set("clma", clma)
        store.set("clna", clna)
        store.set("gavail", gavail)

        self.aerodynamics_der(vehicle)

    def aerodynamics_der(self, vehicle):
        store = vehicle.store
        look_up = self.deck.look_up
        alplimx = store.get("alplimx")
        refl = store.get("refl")
        refa = store.get("refa")
        dna = store.get("dna")
        dnd = store.get("dnd")
        dma = store.get("dma")
        dmq = store.get("dmq")
        dmd = store.get("dmd")
        dlp = store.get("dlp")
        dld = store.get("dld")
        stmarg = store.get("stmarg")
        vmach = store.get("vmach")
        pdynmc = store.get("pdynmc")
        alppx = store.get("alppx")
        dvba = store.get("dvba")
        vmass = store.get("vmass")
        ai11 = store.get("ai11")
        ai33 = store.get("ai33")
        cndq = store.get("cndq")
        clmdq = store.get("clmdq")
        clmq = store.get("clmq")
        cllp = store.get("cllp")
        clldp = store.get("clldp")

        if alppx < (alplimx - 3.0):
            alpp = alppx + 3
            if alpp < 3:
                alpp = 3
            alpm = alppx - 3
            if alpm < 0:
                alpm = 0
            cn0p = look_up("cn0_vs_mach_alpha", vmach, alpp)
            cn0m = look_up("cn0_vs_mach_alpha", vmach, alpm)
            dum = cn0p - cn0m
            cna = DEG * dum / (alpp - alpm)
            cnd = DEG * cndq
            clm0p = look_up("clm0_vs_mach_alpha", vmach, alpp)
            clm0m = look_up("clm0_vs_mach_alpha", vmach, alpm)
            cma = DEG * (clm0p - clm0m) / (alpp - alpm)
            cmq = DEG * clmq
            cmd = DEG * clmdq
            clp = DEG * cllp
            cld = DEG * clldp
            dumn = pdynmc * refa / vmass
            dna = dumn * cna
            dnd = dumn * cnd
            dumm = pdynmc * refa * refl / ai33
            dma = dumm * cma
            dmq = dumm * (refl / (2 * dvba)) * cmq
            dmd = dumm * cmd
            duml = pdynmc * refa * refl / ai11
            dlp = duml * (refl / (2.0 * dvba)) * clp
            dld = duml * cld
            stmarg = -cma / cna

        a11 = dmq
        a12 = dma / dna
        a21 = dna
        a22 = -dna / dvba
        arg = (a11 + a22) ** 2 - 4.0 * (a11 * a22 - a12 * a21)
        if arg >= 0.0:
            wnq = 0.0
            zetq = 0.0
            dum = a11 + a22
            realq1 = (dum + sqrt(arg)) / 2.0
            realq2 = (dum - sqrt(arg)) / 2.0
            pqreal = (realq1 + realq2) / 2.0
        else:
            realq1 = 0.0
            realq2 = 0.0
            wnq = sqrt(a11 * a22 - a12 * a21)
            zetq = -(a11 + a22) / (2.0 * wnq)
            pqreal = -zetq * wnq
        realp = dlp

        store.set("dna", dna)
        store.set("dnd", dnd)
        store.set("dma", dma)
        store.set("dmq", dmq)
        store.set("dmd", dmd)
        store.set("dlp", dlp)
        store.set("dld", dld)
        store.set("stmarg", stmarg)
        store.set("realq1", realq1)
        store.set("realq2", realq2)
        store.set("wnq", wnq)
        store.set("zetq", zetq)
        store.set("realp", realp)
        store.set("pqreal", pqreal)

    def terminate(self, vehicle, ctx):
        pass
