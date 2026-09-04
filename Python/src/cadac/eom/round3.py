import numpy as np

from cadac.constants import RAD, REARTH, WEII3
from cadac.env.gravity import gravity
from cadac.env.iso62 import iso62
from cadac.kernel.state import Field
from cadac.math.earth import cadtei, cadtge
from cadac.math.frames import mat2tr


class Round3Environment:
    name = "environment"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("time", 0.0, "real", "exec", "environment", ("scrn", "plot", "com")),
            Field("event_time", 0.0, "real", "exec", "environment", ("scrn",)),
            Field("int_step_new", 0.0, "real", "data", "environment"),
            Field("out_step_fact", 0.0, "real", "data", "environment"),
            Field("grav", 0.0, "real", "out", "environment"),
            Field("rho", 0.0, "real", "out", "environment"),
            Field("pdynmc", 0.0, "real", "out", "environment", ("scrn", "plot")),
            Field("mach", 0.0, "real", "out", "environment", ("scrn", "plot", "com")),
            Field("vsound", 0.0, "real", "diag", "environment"),
            Field("press", 0.0, "real", "diag", "environment"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        vehicle.store.set("time", ctx.sim_time)
        vehicle.store.set("int_step_new", ctx.int_step)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        ctx.int_step = store.get("int_step_new")
        ctx.out_fact = store.get("out_step_fact")
        alt = store.get("alt")
        dvbe = store.get("dvbe")
        atm = iso62(alt, dvbe)
        store.set("time", ctx.sim_time)
        store.set("event_time", ctx.event_time)
        store.set("grav", gravity(alt))
        store.set("rho", atm["rho"])
        store.set("pdynmc", atm["pdynmc"])
        store.set("mach", atm["mach"])
        store.set("vsound", atm["vsound"])
        store.set("press", atm["press"])

    def terminate(self, vehicle, ctx):
        pass


class Round3Newton:
    name = "newton"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for field in (
            Field("psivg", 0.0, "real", "out", "newton"),
            Field("thtvg", 0.0, "real", "out", "newton"),
            Field("lonx", 0.0, "real", "init/diag", "newton", ("scrn", "plot", "com")),
            Field("latx", 0.0, "real", "init/diag", "newton", ("scrn", "plot", "com")),
            Field("alt", 0.0, "real", "init/out", "newton", ("scrn", "plot", "com")),
            Field("tgv", zeros33, "mat", "init", "newton"),
            Field("tig", zeros33, "mat", "init/out", "newton"),
            Field("dvbe", 0.0, "real", "init/out", "newton", ("scrn", "plot", "com")),
            Field("weii", zeros33, "mat", "init", "newton"),
            Field("psivgx", 0.0, "real", "init/out", "newton", ("scrn", "plot", "com")),
            Field("thtvgx", 0.0, "real", "init/out", "newton", ("scrn", "plot", "com")),
            Field("sb0ii", zeros3, "vec", "init", "newton"),
            Field("sbeg", zeros3, "vec", "state", "newton", ("scrn", "plot", "com")),
            Field("vbeg", zeros3, "vec", "state", "newton", ("scrn", "plot", "com")),
            Field("tge", zeros33, "mat", "out", "newton"),
            Field("altx", 0.0, "real", "diag", "newton"),
            Field("sbii", zeros3, "vec", "state", "newton", ("com",)),
            Field("vbii", zeros3, "vec", "state", "newton"),
            Field("abii", zeros3, "vec", "state", "newton"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        dvbe = store.get("dvbe")
        psivgx = store.get("psivgx")
        thtvgx = store.get("thtvgx")
        lonx = store.get("lonx")
        latx = store.get("latx")
        alt = store.get("alt")

        sbig = np.array([0.0, 0.0, -(alt + REARTH)])
        tge = cadtge(lonx * RAD, latx * RAD)
        teg = tge.T
        sbie = teg @ sbig
        tei = cadtei(ctx.sim_time)
        sbii = tei.T @ sbie
        sb0ii = sbii.copy()

        psivg = psivgx * RAD
        thtvg = thtvgx * RAD
        vbeg = np.array(
            [
                dvbe * np.cos(thtvg) * np.cos(psivg),
                dvbe * np.cos(thtvg) * np.sin(psivg),
                dvbe * (-np.sin(thtvg)),
            ]
        )

        weii = np.zeros((3, 3))
        weii[0, 1] = -WEII3
        weii[1, 0] = WEII3

        tig = tei.T @ teg
        vbii = tig @ vbeg + weii @ sbii
        tgv = mat2tr(psivg, thtvg).T

        store.set("tgv", tgv)
        store.set("tig", tig)
        store.set("weii", weii)
        store.set("sb0ii", sb0ii)
        store.set("vbeg", vbeg)
        store.set("tge", tge)
        store.set("sbii", sbii)
        store.set("vbii", vbii)

    def execute(self, vehicle, ctx):
        pass

    def terminate(self, vehicle, ctx):
        pass
