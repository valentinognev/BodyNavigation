from cadac.constants import AGRAV
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field


class Cruise3Propulsion:
    name = "propulsion"

    def __init__(self, deck):
        self.deck = deck

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("mprop", 0, "int", "data", "propulsion"),
            Field("acowl", 0.0, "real", "data", "propulsion"),
            Field("throttle", 0.0, "real", "data/diag", "propulsion", ("scrn", "plot")),
            Field("thrtl_max", 0.0, "real", "data", "propulsion"),
            Field("qhold", 0.0, "real", "data", "propulsion"),
            Field("mass", 0.0, "real", "out", "propulsion", ("scrn", "plot")),
            Field("mass0", 0.0, "real", "data", "propulsion"),
            Field("tq", 0.0, "real", "data", "propulsion"),
            Field("thrtl_idle", 0.0, "real", "data", "propulsion"),
            Field("fmass0", 0.0, "real", "data", "propulsion"),
            Field("fmasse", 0.0, "real", "state", "propulsion"),
            Field("fmassd", 0.0, "real", "state", "propulsion"),
            Field("ca", 0.0, "real", "diag", "propulsion"),
            Field("spi", 0.0, "real", "diag", "propulsion"),
            Field("thrust", 0.0, "real", "out", "propulsion", ("scrn", "plot")),
            Field("mass_flow", 0.0, "real", "diag", "propulsion"),
            Field("fmassr", 0.0, "real", "diag", "propulsion", ("scrn", "plot")),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        store.set("mass", store.get("mass0"))

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mprop = store.get("mprop")
        acowl = store.get("acowl")
        throttle = store.get("throttle")
        mass0 = store.get("mass0")
        fmass0 = store.get("fmass0")
        mass = store.get("mass")
        fmassr = store.get("fmassr")
        fmasse = store.get("fmasse")
        fmassd = store.get("fmassd")
        rho = store.get("rho")
        mach = store.get("mach")
        dvbe = store.get("dvbe")
        alphax = store.get("alphax")

        spi = 0.0
        ca = 0.0
        thrust = 0.0
        mass_flow = 0.0

        if mprop == 1:
            spi = self.deck.look_up("spi_vs_throttle_mach", throttle, mach)
            ca = self.deck.look_up("ca_vs_alpha_mach", alphax, mach)
            thrust = spi * 0.029 * throttle * AGRAV * rho * dvbe * ca * acowl
            if spi != 0:
                fmassd_next = thrust / (spi * AGRAV)
                fmasse = integrate(fmassd_next, fmassd, fmasse, ctx.int_step)
                fmassd = fmassd_next
                mass_flow = thrust / (AGRAV * spi)
            mass = mass0 - fmasse
            fmassr = fmass0 - fmasse
            if fmassr <= 0:
                mprop = 0

        if mprop == 0:
            fmassd = 0.0
            thrust = 0.0

        store.set("fmasse", fmasse)
        store.set("fmassd", fmassd)
        store.set("mprop", mprop)
        store.set("mass", mass)
        store.set("thrust", thrust)
        store.set("throttle", throttle)
        store.set("ca", ca)
        store.set("spi", spi)
        store.set("mass_flow", mass_flow)
        store.set("fmassr", fmassr)

    def terminate(self, vehicle, ctx):
        pass
