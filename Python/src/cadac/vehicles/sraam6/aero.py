import math

from cadac.constants import AGRAV
from cadac.kernel.state import Field


class Sraam6Aero:
    name = "aerodynamics"

    def __init__(self, deck):
        self.deck = deck

    def define(self, vehicle):
        store = vehicle.store
        plot = ("plot",)
        scrn_plot = ("scrn", "plot")
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
            Field("stmarg", 0.0, "real", "diag", "aerodynamics"),
            Field("realq1", 0.0, "real", "diag", "aerodynamics"),
            Field("realq2", 0.0, "real", "diag", "aerodynamics"),
            Field("wnq", 0.0, "real", "diag", "aerodynamics"),
            Field("zetq", 0.0, "real", "diag", "aerodynamics"),
            Field("realp", 0.0, "real", "diag", "aerodynamics"),
            Field("alplimx", 0.0, "real", "data", "aerodynamics"),
            Field("gavail", 0.0, "real", "diag", "aerodynamics", plot),
            Field("gmax", 0.0, "real", "diag", "aerodynamics", plot),
            Field("pqreal", 0.0, "real", "diag", "aerodynamics"),
            Field("trcond", 0, "int", "diag", "aerodynamics", scrn_plot),
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
        store.set("refl", 0.1524)
        store.set("refa", 0.01824)
        store.set("trcvel", 10e-4)
        store.set("trmach", 0.5)
        store.set("trdynm", 10e3)
        store.set("trload", 3)
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
        dvbe = store.get("dvbe")
        mprop = store.get("mprop")
        vmass = store.get("vmass")
        xcgref = store.get("xcgref")
        xcg = store.get("xcg")
        alimit = store.get("alimit")
        dpx = store.get("dpx")
        dqx = store.get("dqx")
        drx = store.get("drx")

        cphip = math.cos(phip)
        sphip = math.sin(phip)
        dqax = dqx * cphip - drx * sphip
        drax = dqx * sphip + drx * cphip
        qqax = qqx * cphip - rrx * sphip
        rrax = qqx * sphip + rrx * cphip

        ca0 = look_up("ca0_vs_mach", vmach)
        caa = look_up("caa_vs_mach", vmach)
        cad = look_up("cad_vs_mach", vmach)
        caoff = look_up("caoff_vs_mach", vmach)
        deff = (math.fabs(dqax) + math.fabs(drax)) / 2.0
        ca = ca0 + caa * alppx + cad * deff * deff + float(1 - mprop) * caoff

        cyp = look_up("cyp_vs_mach_alpha", vmach, alppx)
        cydr = look_up("cndq_vs_mach_alpha", vmach, alppx)
        s4phi = math.sin(4.0 * phip)
        cya = cyp * s4phi + cydr * drax

        cn0 = look_up("cn0_vs_mach_alpha", vmach, alppx)
        cnp = look_up("cnp_vs_mach_alpha", vmach, alppx)
        cndq = look_up("cndq_vs_mach_alpha", vmach, alppx)
        s2phi = math.sin(2.0 * phip) ** 2
        cna = cn0 + cnp * s2phi + cndq * dqax

        cllap = look_up("cllap_vs_mach_alpha", vmach, alppx)
        cllp = look_up("cllp_vs_mach_alpha", vmach, alppx)
        clldp = look_up("clldp_vs_mach_alpha", vmach, alppx)
        cll = (
            cllap * alppx * alppx * s4phi
            + cllp * ppx * refl / (2.0 * dvbe)
            + clldp * dpx
        )

        clm0 = look_up("clm0_vs_mach_alpha", vmach, alppx)
        clmp = look_up("clmp_vs_mach_alpha", vmach, alppx)
        clmq = look_up("clmq_vs_mach", vmach)
        clmdq = look_up("clmdq_vs_mach_alpha", vmach, alppx)
        clmaref = (
            clm0
            + clmp * s2phi
            + clmq * qqax * refl / (2.0 * dvbe)
            + clmdq * dqax
        )
        clma = clmaref - cna * (xcgref - xcg) / refl

        clnp = look_up("clnp_vs_mach_alpha", vmach, alppx)
        clnr = clmq
        clndr = clmdq
        clnaref = clnp * s4phi + clnr * rrax * refl / (2.0 * dvbe) + clndr * drax
        clna = clnaref - cya * (xcgref - xcg) / refl

        cy = cya * cphip - cna * sphip
        cn = cya * sphip + cna * cphip
        clm = clma * cphip + clna * sphip
        cln = -clma * sphip + clna * cphip

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
        store.set("cn0", cn0)
        store.set("cnp", cnp)
        store.set("clm0", clm0)
        store.set("clmp", clmp)
        store.set("cyp", cyp)
        store.set("ca0", ca0)
        store.set("clnp", clnp)
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
        store.set("trcond", trcond)

    def terminate(self, vehicle, ctx):
        pass
