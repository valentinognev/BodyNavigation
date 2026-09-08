import numpy as np

from cadac.constants import DEG, RAD, REARTH, WEII3
from cadac.env.gravity import gravity
from cadac.env.iso62 import iso62
from cadac.kernel.integrate import integrate
from cadac.kernel.module import ModuleBase
from cadac.kernel.state import Field
from cadac.math.earth import cadsph, cadtei, cadtge
from cadac.math.frames import mat2tr, polar_from_cart


class Round3Environment(ModuleBase):
    name = "environment"
    fields = (
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
    )

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


class Round3Newton(ModuleBase):
    name = "newton"
    fields = (
        Field("psivg", 0.0, "real", "out", "newton"),
        Field("thtvg", 0.0, "real", "out", "newton"),
        Field("lonx", 0.0, "real", "init/diag", "newton", ("scrn", "plot", "com")),
        Field("latx", 0.0, "real", "init/diag", "newton", ("scrn", "plot", "com")),
        Field("alt", 0.0, "real", "init/out", "newton", ("scrn", "plot", "com")),
        Field("tgv", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "init", "newton"),
        Field("tig", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "init/out", "newton"),
        Field("dvbe", 0.0, "real", "init/out", "newton", ("scrn", "plot", "com")),
        Field("weii", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "init", "newton"),
        Field("psivgx", 0.0, "real", "init/out", "newton", ("scrn", "plot", "com")),
        Field("thtvgx", 0.0, "real", "init/out", "newton", ("scrn", "plot", "com")),
        Field("sb0ii", (0.0, 0.0, 0.0), "vec", "init", "newton"),
        Field("sbeg", (0.0, 0.0, 0.0), "vec", "state", "newton", ("scrn", "plot", "com")),
        Field("vbeg", (0.0, 0.0, 0.0), "vec", "state", "newton", ("scrn", "plot", "com")),
        Field("tge", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "out", "newton"),
        Field("altx", 0.0, "real", "diag", "newton"),
        Field("sbii", (0.0, 0.0, 0.0), "vec", "state", "newton", ("com",)),
        Field("vbii", (0.0, 0.0, 0.0), "vec", "state", "newton"),
        Field("abii", (0.0, 0.0, 0.0), "vec", "state", "newton"),
    )

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
        store = vehicle.store
        weii = store.get("weii")
        sbeg = store.get("sbeg")
        vbeg = store.get("vbeg")
        sbii = store.get("sbii")
        vbii = store.get("vbii")
        abii = store.get("abii")
        tgv = store.get("tgv")
        tig = store.get("tig")
        fspv = store.get("FSPV")
        grav = store.get("grav")
        int_step = ctx.int_step

        grav_vec = np.zeros(3)
        grav_vec[2] = grav

        abii_new = tig @ ((tgv @ fspv) + grav_vec)
        vbii_new = integrate(abii_new, abii, vbii, int_step)
        sbii = integrate(vbii_new, vbii, sbii, int_step)
        abii = abii_new
        vbii = vbii_new

        tei = cadtei(ctx.sim_time)
        sbie = tei @ sbii
        lon, lat, alt = cadsph(sbie)
        lonx = lon * DEG
        latx = lat * DEG
        altx = alt / 1000.0

        tge = cadtge(lon, lat)
        tgi = tge @ tei
        vbeg_new = tgi @ (vbii - weii @ sbii)
        sbeg = integrate(vbeg_new, vbeg, sbeg, int_step)
        vbeg = vbeg_new

        polar = polar_from_cart(vbeg)
        dvbe = float(polar[0])
        psivg = float(polar[1])
        thtvg = float(polar[2])
        psivgx = psivg * DEG
        thtvgx = thtvg * DEG

        tig = tgi.T
        tvg = mat2tr(psivg, thtvg)
        tgv = tvg.T

        store.set("sbeg", sbeg)
        store.set("vbeg", vbeg)
        store.set("sbii", sbii)
        store.set("vbii", vbii)
        store.set("abii", abii)
        store.set("tgv", tgv)
        store.set("tig", tig)
        store.set("dvbe", dvbe)
        store.set("psivg", psivg)
        store.set("thtvg", thtvg)
        store.set("alt", alt)
        store.set("psivgx", psivgx)
        store.set("thtvgx", thtvgx)
        store.set("lonx", lonx)
        store.set("latx", latx)
        store.set("tge", tge)
        store.set("altx", altx)
