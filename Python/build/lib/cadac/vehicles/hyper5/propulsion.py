from cadac.constants import AGRAV
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field


class Hyper5Propulsion:
    name = "propulsion"

    def __init__(self, deck):
        self.deck = deck

    def define(self, vehicle):
        store = vehicle.store
        plot = ("scrn", "plot")
        for field in (
            Field("phi_const", 0.0, "real", "data", "propulsion"),
            Field("tlag", 0.0, "real", "data", "propulsion"),
            Field("phis", 0.0, "real", "state", "propulsion", ("plot",)),
            Field("phisd", 0.0, "real", "state", "propulsion", ("plot",)),
            Field("mprop", 0, "int", "data", "propulsion", ("plot",)),
            Field("aintake", 0.0, "real", "data", "propulsion"),
            Field("phi", 0.0, "real", "data/diag", "propulsion", plot),
            Field("phi_max", 0.0, "real", "data", "propulsion"),
            Field("qhold", 0.0, "real", "data", "propulsion"),
            Field("mass", 0.0, "real", "out", "propulsion", plot),
            Field("mass0", 0.0, "real", "data", "propulsion"),
            Field("cin", 0.0, "real", "diag", "propulsion", plot),
            Field("tq", 0.0, "real", "data", "propulsion"),
            Field("phi_min", 0.0, "real", "data", "propulsion"),
            Field("fmass0", 0.0, "real", "data", "propulsion"),
            Field("fmasse", 0.0, "real", "state", "propulsion"),
            Field("fmassd", 0.0, "real", "state", "propulsion"),
            Field("thrst_stoch", 0.0, "real", "diag", "propulsion", ("plot",)),
            Field("spi", 0.0, "real", "diag", "propulsion", plot),
            Field("thrust", 0.0, "real", "out", "propulsion", plot),
            Field("mass_flow", 0.0, "real", "diag", "propulsion"),
            Field("fmassr", 0.0, "real", "diag", "propulsion", plot),
            Field("thrst_req", 0.0, "real", "diag", "propulsion", plot),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        store.set("mass", store.get("mass0"))

    def execute(self, vehicle, ctx):
        store = vehicle.store
        dt = ctx.int_step
        mprop = store.get("mprop")
        if mprop not in (0, 1, 2, 3):
            raise ValueError(f"unknown mprop {mprop}")

        tlag = store.get("tlag")
        aintake = store.get("aintake")
        phi_const = store.get("phi_const")
        phi_max = store.get("phi_max")
        qhold = store.get("qhold")
        mass0 = store.get("mass0")
        fmass0 = store.get("fmass0")
        tq = store.get("tq")
        phi_min = store.get("phi_min")
        mass = store.get("mass")
        fmassr = store.get("fmassr")
        phis = store.get("phis")
        phisd = store.get("phisd")
        fmasse = store.get("fmasse")
        fmassd = store.get("fmassd")
        rho = store.get("rho")
        pdynmc = store.get("pdynmc")
        mach = store.get("mach")
        dvbe = store.get("dvbe")
        ca = store.get("ca")
        area = store.get("area")
        alphax = store.get("alphax")

        spi = 0.0
        thrust = 0.0
        mass_flow = 0.0
        cin = 0.0
        phi = 0.0
        thrst_req = 0.0
        thrst_stoch = 0.0

        if mprop > 0:
            if mprop:
                cin = self.deck.look_up("cin_vs_alphax_mach", alphax, mach)
                spi = self.deck.look_up("spi_vs_mach_phi_alphax", mach, phi, alphax)
            if mprop == 1:
                thrust = 0.0676 * phi_const * spi * AGRAV * rho * dvbe * cin * aintake
            if mprop == 2:
                thrst_stoch = 0.0676 * spi * AGRAV * rho * dvbe * cin * aintake
                thrst_req = area * ca * qhold
                phi_req = thrst_req / thrst_stoch
                gainq = 2 * mass / (rho * dvbe * thrst_stoch * tq)
                ephi = gainq * (qhold - pdynmc)
                phi = phi_req + ephi
                phisd_new = (phi - phis) / tlag
                phis = integrate(phisd_new, phisd, phis, dt)
                phisd = phisd_new
                phi = phis
                if phi < phi_min:
                    phi = phi_min
                if phi > phi_max:
                    phi = phi_max
                spi = self.deck.look_up(
                    "spi_vs_mach_phi_alphax", mach, phi / 0.0676, alphax
                )
                thrust = 0.0676 * phi * spi * AGRAV * rho * dvbe * cin * aintake
            if mprop == 3:
                spi = self.deck.look_up(
                    "spi_vs_mach_phi_alphax", mach, phi / 0.0676, alphax
                )
                thrust = 0.0676 * phi * spi * AGRAV * rho * dvbe * cin * aintake
            fmassd_next = thrust / (spi * AGRAV)
            fmasse = integrate(fmassd_next, fmassd, fmasse, dt)
            fmassd = fmassd_next
            mass = mass0 - fmasse
            fmassr = fmass0 - fmasse
            mass_flow = thrust / (AGRAV * spi)
            if fmassr <= 0:
                mprop = 0

        if mprop == 0:
            fmassd = 0.0
            thrust = 0.0

        store.set("phis", phis)
        store.set("phisd", phisd)
        store.set("fmasse", fmasse)
        store.set("fmassd", fmassd)
        store.set("phi", phi)
        store.set("mass", mass)
        store.set("thrust", thrust)
        store.set("mprop", mprop)
        store.set("cin", cin)
        store.set("thrst_stoch", thrst_stoch)
        store.set("spi", spi)
        store.set("mass_flow", mass_flow)
        store.set("fmassr", fmassr)
        store.set("thrst_req", thrst_req)

    def terminate(self, vehicle, ctx):
        pass
