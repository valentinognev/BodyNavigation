"""Zipfel 3-DOF round-Earth equations of motion (CADAC Round3)."""

import math

import numpy as np

from cadac.constants import DEG, R, RAD, REARTH, WEII3
from cadac.env.gravity import gravity
from cadac.env.iso62 import iso62
from cadac.kernel.integrate import integrate
from cadac.kernel.module import ModuleBase
from cadac.kernel.state import Field
from cadac.math.earth import cadsph, cadtei, cadtge
from cadac.math.frames import hypot3, mat2tr, polar_from_cart

# CRUISE5 Fortran G2 wind smoother time constant (s).
_CRUISE5_TWIND = 1.0


class Round3Environment(ModuleBase):
    """Zipfel 3-DOF round Earth atmosphere (CADAC ``round3_environment``).

    ``mair_pack`` selects digit meaning of ``mair``:

    * ``\"ghame3\"`` (default, HYPER3): MAIR=0 ISO-62; MAIR=1 weather deck
      atmosphere only (density / pressure / temperature °C).
    * ``\"cruise5\"`` (CRUISE5 G2): MAIR=|MATM|MWIND| with
      MATM=INT(MAIR/10), MWIND=MAIR-MATM*10 — MATM 0 ISO / 1 tabular;
      MWIND 0 none / 1 constant wind from ``dvael``/``psiwlx``.
    """

    name = "environment"
    fields = (
        Field("time", 0.0, "real", "exec", "environment", ("scrn", "plot", "com")),
        Field("event_time", 0.0, "real", "exec", "environment", ("scrn",)),
        Field("int_step_new", 0.0, "real", "data", "environment"),
        Field("out_step_fact", 0.0, "real", "data", "environment"),
        Field("mair", 0, "int", "data", "environment"),
        Field("grav", 0.0, "real", "out", "environment"),
        Field("rho", 0.0, "real", "out", "environment"),
        Field("pdynmc", 0.0, "real", "out", "environment", ("scrn", "plot")),
        Field("mach", 0.0, "real", "out", "environment", ("scrn", "plot", "com")),
        Field("vsound", 0.0, "real", "diag", "environment"),
        Field("press", 0.0, "real", "diag", "environment"),
    )

    def __init__(self, weather_deck=None, mair_pack: str = "ghame3") -> None:
        self.weather_deck = weather_deck
        if mair_pack not in ("ghame3", "cruise5"):
            raise ValueError(f"unknown mair_pack {mair_pack}")
        self.mair_pack = mair_pack

    def initialize(self, vehicle, ctx) -> None:
        vehicle.store.set("time", ctx.sim_time)
        vehicle.store.set("int_step_new", ctx.int_step)

    def _tabular_atmosphere(self, alt, dvba):
        if self.weather_deck is None:
            raise ValueError("mair tabular atmosphere requires a weather Datadeck")
        rho = self.weather_deck.look_up("density", alt)
        tempc = self.weather_deck.look_up("temperature", alt)
        press = self.weather_deck.look_up("pressure", alt)
        tempk = tempc + 273.16
        vsound = math.sqrt(1.4 * R * tempk)
        mach = abs(dvba / vsound) if vsound > 1.0e-10 else 0.0
        pdynmc = 0.5 * rho * dvba**2
        return rho, press, vsound, mach, pdynmc

    def execute(self, vehicle, ctx) -> None:
        store = vehicle.store
        ctx.int_step = store.get("int_step_new")
        ctx.out_fact = store.get("out_step_fact")
        alt = store.get("alt")
        dvbe = store.get("dvbe")
        mair = store.get("mair")

        if self.mair_pack == "cruise5":
            matm = int(mair // 10)
            mwind = int(mair - matm * 10)
            if matm not in (0, 1) or mwind not in (0, 1):
                raise ValueError(f"unknown mair {mair}")

            vael = np.zeros(3)
            if mwind != 0:
                dvael = store.get("dvael")
                psiwlx = store.get("psiwlx")
                dvae3 = store.get("dvae3")
                dvw = dvael  # MWIND=1 constant
                vael_raw = np.array(
                    [
                        -dvw * math.cos(psiwlx * RAD),
                        -dvw * math.sin(psiwlx * RAD),
                        dvae3,
                    ],
                    dtype=float,
                )
                vael_s = np.array(store.get("VAEL"), dtype=float, copy=True)
                vaeld = np.array(store.get("VAELD"), dtype=float, copy=True)
                vaeld_new = (vael_raw - vael_s) * (1.0 / _CRUISE5_TWIND)
                vael_s = integrate(vaeld_new, vaeld, vael_s, ctx.int_step)
                store.set("VAEL", vael_s)
                store.set("VAELD", vaeld_new)
                store.set("dvw", float(dvw))
                vael = vael_s
            else:
                store.set("VAEL", vael)
                store.set("VAELD", np.zeros(3))
                store.set("dvw", 0.0)

            vbeg = np.asarray(store.get("vbeg"), dtype=float)
            dvba = float(hypot3(vbeg - vael))
            store.set("dvba", dvba)

            if matm == 0:
                atm = iso62(alt, dvba)
                rho = atm["rho"]
                press = atm["press"]
                vsound = atm["vsound"]
                mach = atm["mach"]
                pdynmc = atm["pdynmc"]
            else:
                rho, press, vsound, mach, pdynmc = self._tabular_atmosphere(alt, dvba)
        else:
            # GHAME3 / default: MAIR=0 ISO, MAIR=1 weather deck (no wind).
            if mair == 0:
                atm = iso62(alt, dvbe)
                rho = atm["rho"]
                press = atm["press"]
                vsound = atm["vsound"]
                mach = atm["mach"]
                pdynmc = atm["pdynmc"]
            elif mair == 1:
                if self.weather_deck is None:
                    raise ValueError("mair 1 requires a weather Datadeck")
                rho, press, vsound, mach, pdynmc = self._tabular_atmosphere(alt, dvbe)
            else:
                raise ValueError(f"unknown mair {mair}")

        store.set("time", ctx.sim_time)
        store.set("event_time", ctx.event_time)
        store.set("grav", gravity(alt))
        store.set("rho", rho)
        store.set("pdynmc", pdynmc)
        store.set("mach", mach)
        store.set("vsound", vsound)
        store.set("press", press)


class Round3Newton(ModuleBase):
    """Zipfel 3-DOF round Earth translational Newton (CADAC ``round3_newton``)."""

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

    def initialize(self, vehicle, ctx) -> None:
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

    def execute(self, vehicle, ctx) -> None:
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
