from math import cos

from cadac.constants import RAD
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field


class Cruise5Propulsion:
    name = "propulsion"

    def __init__(self, deck):
        self.deck = deck

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("mprop", 0, "int", "data/diag", "propulsion", ("scrn", "plot")),
            Field("cg", 0.0, "real", "diag", "propulsion", ("plot",)),
            Field("fidle", 0.0, "real", "diag", "propulsion"),
            Field("thrust_com", 0.0, "real", "data", "propulsion"),
            Field("thrust", 0.0, "real", "out", "propulsion", ("scrn", "plot", "com")),
            Field("treqd", 0.0, "real", "state", "propulsion"),
            Field("treq", 0.0, "real", "state", "propulsion"),
            Field("fmassed", 0.0, "real", "state", "propulsion"),
            Field("fmasse", 0.0, "real", "state", "propulsion"),
            Field("fuelmass", 0.0, "real", "save", "propulsion"),
            Field("mach_com", 0.0, "real", "data", "propulsion"),
            Field("gfthm", 0.0, "real", "data", "propulsion"),
            Field("tfth", 0.0, "real", "data", "propulsion"),
            Field("mass", 0.0, "real", "out", "propulsion", ("scrn",)),
            Field("tav", 0.0, "real", "diag", "propulsion"),
            Field("mass_init", 0.0, "real", "data", "propulsion"),
            Field("fuel_init", 0.0, "real", "data", "propulsion"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        store.set("mass", store.get("mass_init"))

    def execute(self, vehicle, ctx):
        store = vehicle.store
        dt = ctx.int_step
        mprop = store.get("mprop")
        thrust_com = store.get("thrust_com")
        mach_com = store.get("mach_com")
        gfthm = store.get("gfthm")
        tfth = store.get("tfth")
        mass_init = store.get("mass_init")
        fuel_init = store.get("fuel_init")
        pdynmc = store.get("pdynmc")
        mach = store.get("mach")
        alt = store.get("alt")
        treqd = store.get("treqd")
        treq = store.get("treq")
        fmassed = store.get("fmassed")
        fmasse = store.get("fmasse")
        mass = store.get("mass")
        cd = store.get("cd")
        area = store.get("area")
        alphax = store.get("alphax")

        if mprop < 0:
            raise ValueError(f"mprop {mprop}")

        if mprop == 0:
            thrust = 0.0
            cg = self.deck.look_up("cg_vs_mass", mass)
            store.set("thrust", thrust)
            store.set("cg", cg)
            return

        fidle = self.deck.look_up("fidle_vs_alt_mach", alt, mach)
        tav = self.deck.look_up("tav_vs_alt_mach", alt, mach)
        ff = 0.0
        thrust = 0.0

        if mprop == 1:
            thrust = thrust_com
            ff = self.deck.look_up("ff_vs_thrust_alt_mach", thrust, alt, mach)
            treq = thrust_com
        elif mprop == 2:
            thrust = fidle
            ff = self.deck.look_up("iff_vs_alt", alt)
        elif mprop == 3:
            thrust = tav
            ff = self.deck.look_up("ff_vs_thrust_alt_mach", thrust, alt, mach)
        elif mprop > 3:
            mprop = 4
            treqs = cd * pdynmc * area
            epsmch = mach_com - mach
            tcom = epsmch * gfthm + treqs
            treqd_new = (tcom - 2.0 * treq) / tfth
            treq = integrate(treqd_new, treqd, treq, dt)
            treqd = treqd_new
            treqb = treq / cos(alphax * RAD)
            if treqb < fidle:
                mprop = 5
                treqb = fidle
            if treqb > tav:
                mprop = 6
                treqb = tav
            thrust = treqb
            ff = self.deck.look_up("ff_vs_thrust_alt_mach", thrust, alt, mach)

        fmasse = integrate(ff, fmassed, fmasse, dt)
        fmassed = ff
        mass = mass_init - fmasse
        fuelmass = fuel_init - fmasse
        cg = self.deck.look_up("cg_vs_mass", mass)
        if fuelmass <= 0:
            thrust = 0.0

        store.set("treqd", treqd)
        store.set("treq", treq)
        store.set("fmassed", fmassed)
        store.set("fmasse", fmasse)
        store.set("fuelmass", fuelmass)
        store.set("mprop", mprop)
        store.set("thrust", thrust)
        store.set("mass", mass)
        store.set("cg", cg)
        store.set("fidle", fidle)
        store.set("tav", tav)

    def terminate(self, vehicle, ctx):
        pass
