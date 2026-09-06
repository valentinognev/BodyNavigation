from math import cos, sin

import numpy as np

from cadac.constants import AGRAV, RAD
from cadac.kernel.state import Field


class Plane6Aero:
    name = "aerodynamics"

    def __init__(self, deck):
        self.deck = deck

    def define(self, vehicle):
        store = vehicle.store
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for field in (
            Field("refa", 0.0, "real", "init", "aerodynamics"),
            Field("refb", 0.0, "real", "init", "aerodynamics"),
            Field("refc", 0.0, "real", "init", "aerodynamics"),
            Field("cd", 0.0, "real", "diag", "aerodynamics"),
            Field("cl", 0.0, "real", "diag", "aerodynamics"),
            Field("cxt", 0.0, "real", "out", "aerodynamics"),
            Field("cyt", 0.0, "real", "out", "aerodynamics"),
            Field("czt", 0.0, "real", "out", "aerodynamics"),
            Field("clt", 0.0, "real", "out", "aerodynamics"),
            Field("cmt", 0.0, "real", "out", "aerodynamics"),
            Field("cnt", 0.0, "real", "out", "aerodynamics"),
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
            Field("stmarg", 0.0, "real", "diag", "aerodynamics", ("plot",)),
            Field("dla", 0.0, "real", "out", "aerodynamics"),
            Field("dlde", 0.0, "real", "out", "aerodynamics"),
            Field("dma", 0.0, "real", "out", "aerodynamics", ("plot",)),
            Field("dmq", 0.0, "real", "out", "aerodynamics"),
            Field("dmde", 0.0, "real", "out", "aerodynamics", ("plot",)),
            Field("dyb", 0.0, "real", "out", "aerodynamics"),
            Field("dydr", 0.0, "real", "out", "aerodynamics"),
            Field("dnb", 0.0, "real", "out", "aerodynamics"),
            Field("dnr", 0.0, "real", "out", "aerodynamics"),
            Field("dndr", 0.0, "real", "out", "aerodynamics"),
            Field("dllp", 0.0, "real", "out", "aerodynamics"),
            Field("dllda", 0.0, "real", "out", "aerodynamics"),
            Field("cdrag", 0.0, "real", "out", "aerodynamics"),
            Field("clift", 0.0, "real", "diag", "aerodynamics"),
            Field("alplimpx", 0.0, "real", "data", "aerodynamics"),
            Field("gmax", 0.0, "real", "out", "aerodynamics", ("plot",)),
            Field("alplimnx", 0.0, "real", "data", "aerodynamics"),
            Field("gminx", 0.0, "real", "out", "aerodynamics", ("plot",)),
            Field("realp1", 0.0, "real", "diag", "aerodynamics", ("plot",)),
            Field("realp2", 0.0, "real", "diag", "aerodynamics", ("plot",)),
            Field("wnp", 0.0, "real", "diag", "aerodynamics", ("plot",)),
            Field("zetp", 0.0, "real", "diag", "aerodynamics", ("plot",)),
            Field("rpreal", 0.0, "real", "diag", "aerodynamics"),
            Field("realy1", 0.0, "real", "diag", "aerodynamics", ("plot",)),
            Field("realy2", 0.0, "real", "diag", "aerodynamics", ("plot",)),
            Field("wny", 0.0, "real", "diag", "aerodynamics", ("plot",)),
            Field("zety", 0.0, "real", "diag", "aerodynamics", ("plot",)),
            Field("ryreal", 0.0, "real", "diag", "aerodynamics"),
            Field("trcode", 0.0, "real", "init", "aerodynamics"),
            Field("tmcode", 0.0, "real", "data", "aerodynamics"),
            Field("trmach", 0.0, "real", "data", "aerodynamics"),
            Field("trdynm", 0.0, "real", "data", "aerodynamics"),
            Field("trload", 0.0, "real", "data", "aerodynamics"),
            Field("tralppx", 0.0, "real", "data", "aerodynamics"),
            Field("tralpnx", 0.0, "real", "data", "aerodynamics"),
            Field("trbetx", 0.0, "real", "data", "aerodynamics"),
            Field("vmass", 0.0, "real", "init", "aerodynamics"),
            Field("IBBB", zeros33, "mat", "init", "aerodynamics"),
            Field("eng_ang_mom", 0.0, "real", "init", "aerodynamics"),
            Field("xcg", 0.0, "real", "data", "aerodynamics"),
            Field("xcgr", 0.0, "real", "data", "aerodynamics"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        store.set("refa", 27.87)
        store.set("refb", 9.14)
        store.set("refc", 3.45)
        vmass = 9496.0
        ibbb = np.zeros((3, 3))
        ibbb[0, 0] = 12875.0
        ibbb[2, 0] = -1331.4
        ibbb[1, 1] = 75673.0
        ibbb[0, 2] = -1331.4
        ibbb[2, 2] = 85551.0
        store.set("vmass", vmass)
        store.set("IBBB", ibbb)
        store.set("eng_ang_mom", 70000.0)
        store.set("trmach", 0.8)
        store.set("trdynm", 10.0e3)
        store.set("trload", 3.0)
        store.set("tralppx", 21.0)
        store.set("tralpnx", -6.0)
        store.set("trbetx", 5.0)
        store.set("trcode", 0.0)
        store.set("tmcode", 0.0)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        look_up = self.deck.look_up

        cd = 0.0
        cd0 = 0.0
        cda = 0.0
        cl0 = 0.0
        cla = 0.0
        cyda = 0.0
        cllb = 0.0
        cllr = 0.0
        cm0 = 0.0
        cmde = 0.0
        clnda = 0.0
        clnp = 0.0
        clldr = 0.0

        alplimpx = store.get("alplimpx")
        alplimnx = store.get("alplimnx")
        xcg = store.get("xcg")
        xcgr = store.get("xcgr")
        refa = store.get("refa")
        refb = store.get("refb")
        refc = store.get("refc")
        trcode = store.get("trcode")
        trload = store.get("trload")
        vmass = store.get("vmass")
        alphax = store.get("alphax")
        betax = store.get("betax")
        pdynmc = store.get("pdynmc")
        dvba = store.get("dvba")
        ppx = store.get("ppx")
        qqx = store.get("qqx")
        rrx = store.get("rrx")
        delax = store.get("delax")
        delex = store.get("delex")
        delrx = store.get("delrx")

        c2v = refc / (2 * dvba)
        b2v = refb / (2 * dvba)

        cx = look_up("cx_vs_elev_alpha", delex, alphax)
        cxq = look_up("cxq_vs_alpha", alphax)
        cxt = cx + c2v * cxq * qqx * RAD

        cyr = look_up("cyr_vs_alpha", alphax)
        cyp = look_up("cyp_vs_alpha", alphax)
        cyt = (
            -0.02 * betax
            + 0.021 * delax / 20
            + 0.086 * delrx / 30
            + b2v * (cyr * rrx * RAD + cyp * ppx * RAD)
        )

        cz = look_up("cz_vs_alpha", alphax)
        czq = look_up("czq_vs_alpha", alphax)
        czt = cz * (1 - (betax * RAD) ** 2) - 0.19 * delex / 25 + c2v * czq * qqx * RAD

        cl = look_up("cl_vs_beta_alpha", betax, alphax)
        cldr = look_up("cldr_vs_beta_alpha", betax, alphax)
        clda = look_up("clda_vs_beta_alpha", betax, alphax)
        clda = -clda
        look_up("clr_vs_alpha", alphax)
        clp = look_up("clp_vs_alpha", alphax)
        clt = cl + clda * delax / 20 + cldr * delrx / 30 + b2v * (cllr * rrx * RAD + clp * ppx * RAD)

        cm = look_up("cm_vs_elev_alpha", delex, alphax)
        cmq = look_up("cmq_vs_alpha", alphax)
        cmt = cm + c2v * cmq * qqx * RAD + czt * (xcgr - xcg) / refc

        cn = look_up("cn_vs_beta_alpha", betax, alphax)
        cnda = look_up("cnda_vs_beta_alpha", betax, alphax)
        cndr = look_up("cndr_vs_beta_alpha", betax, alphax)
        cnr = look_up("cnr_vs_alpha", alphax)
        cnp = look_up("cnp_vs_alpha", alphax)
        cnt = (
            cn
            + cnda * delax / 20
            + cndr * delrx / 30
            - cyt * (xcgr - xcg) / refb
            + b2v * (cnr * rrx * RAD + cnp * ppx * RAD)
        )

        czp = look_up("cz_vs_alpha", alplimpx)
        czn = look_up("cz_vs_alpha", alplimnx)
        alpx = -czp * pdynmc * refa
        alnx = -czn * pdynmc * refa
        weight = vmass * AGRAV
        gmax = alpx / weight
        gminx = alnx / weight

        cosa = cos(alphax * RAD)
        sina = sin(alphax * RAD)
        cdrag = -cxt * cosa - czt * sina
        clift = cxt * sina - czt * cosa
        clovercd = clift / cdrag

        clde = 0.19 / 25
        cyb = -0.02 * RAD
        cydr = -0.086 / 30
        clnr = b2v * cnr
        clndr = cndr / 30
        cllp = b2v * clp
        cllda = clda / 20

        if gmax < trload:
            trcode = 4

        store.set("cxt", cxt)
        store.set("cyt", cyt)
        store.set("czt", czt)
        store.set("clt", clt)
        store.set("cmt", cmt)
        store.set("cnt", cnt)
        store.set("cdrag", cdrag)
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
        store.set("cmde", cmde)
        store.set("cmq", cmq)
        store.set("clnda", clnda)
        store.set("clndr", clndr)
        store.set("clnp", clnp)
        store.set("clnr", clnr)
        store.set("clldr", clldr)
        store.set("clovercd", clovercd)
        store.set("clift", clift)
        store.set("trcode", trcode)

    def terminate(self, vehicle, ctx):
        pass
