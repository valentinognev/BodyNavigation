from math import acos, atan2, cos, fabs, sin, tan

from cadac.constants import DEG, RAD
from cadac.kernel.state import Field

SMALL = 1e-7
G0 = 9.81
CNALP0 = 7.468


class Sam6RocketAero:
    name = "aerodynamics"

    def __init__(self, deck):
        self.deck = deck

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("area", 0.636, "real", "data", "aerodynamics"),
            Field("alpha_t0x", 0.0, "real", "data", "aerodynamics"),
            Field("beta_t0x", 0.0, "real", "data", "aerodynamics"),
            Field("alpmax", 0.0, "real", "data", "aerodynamics"),
            Field("alppx", 0.0, "real", "diag", "aerodynamics"),
            Field("phipx", 0.0, "real", "diag", "aerodynamics"),
            Field("cnptgt", 0.0, "real", "diag", "aerodynamics"),
            Field("cltgt", 0.0, "real", "out", "aerodynamics"),
            Field("cdtgt", 0.0, "real", "out", "aerodynamics"),
            Field("catgt", 0.0, "real", "out", "aerodynamics"),
            Field("cytgt", 0.0, "real", "out", "aerodynamics"),
            Field("cntgt", 0.0, "real", "out", "aerodynamics"),
            Field("cnalp", 0.0, "real", "out", "aerodynamics"),
            Field("cybet", 0.0, "real", "out", "aerodynamics"),
            Field("gmax", 0.0, "real", "out", "aerodynamics"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        cnalp = CNALP0
        store.set("cnalp", cnalp)
        store.set("cybet", -cnalp)
        names = store.names()
        if "alphax" in names:
            store.set("alphax", store.get("alpha_t0x"))
        if "betax" in names:
            store.set("betax", store.get("beta_t0x"))

    def execute(self, vehicle, ctx):
        store = vehicle.store
        look_up = self.deck.look_up
        area = store.get("area")
        alpmax = store.get("alpmax")
        grav = store.get("grav")
        pdynmc = store.get("pdynmc")
        mach = store.get("mach")
        mprop = store.get("mprop")
        mass = store.get("mass")
        alphax = store.get("alphax")
        betax = store.get("betax")

        alpha = alphax * RAD
        beta = betax * RAD
        alpp = acos(cos(alpha) * cos(beta))
        phip = 0.0
        dum1 = tan(beta)
        dum2 = sin(alpha)
        if dum1 * dum1 > SMALL and dum2 * dum2 > SMALL:
            phip = atan2(dum1, dum2)
        alppx = alpp * DEG
        phipx = phip * DEG

        cltgt = look_up("cltgt_vs_alpha_mach", alppx, mach)
        cdtgt = look_up("cdtgt_vs_alpha_mach", alppx, mach)

        cos_alpha = cos(alpha)
        sin_alpha = sin(alpha)
        catgt = cdtgt * cos_alpha - cltgt * sin_alpha
        if mprop == 0:
            catgt = catgt * 1.1
        cnptgt = cdtgt * sin_alpha + cltgt * cos_alpha
        cntgt = fabs(cnptgt) * cos(phip)
        cytgt = -fabs(cnptgt) * sin(phip)

        cltgt_max = look_up("cltgt_vs_alpha_mach", alpmax, mach)
        cdtgt_max = look_up("cdtgt_vs_alpha_mach", alpmax, mach)
        cnp_max = cdtgt_max * sin(alpmax * RAD) + cltgt_max * cos(alpmax * RAD)
        gmax = cnp_max * pdynmc * area / (mass * grav)

        store.set("alppx", alppx)
        store.set("phipx", phipx)
        store.set("cnptgt", cnptgt)
        store.set("cltgt", cltgt)
        store.set("cdtgt", cdtgt)
        store.set("catgt", catgt)
        store.set("cytgt", cytgt)
        store.set("cntgt", cntgt)
        store.set("gmax", gmax)

    def terminate(self, vehicle, ctx):
        pass


class Sam6RocketPropulsion:
    name = "propulsion"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("mprop", 0, "int", "data/out", "propulsion"),
            Field("pres_sl", 101325.0, "real", "data", "propulsion"),
            Field("aexit", 0.282, "real", "data", "propulsion"),
            Field("mass_launch", 6000.0, "real", "data", "propulsion"),
            Field("mass_fuel", 4000.0, "real", "data", "propulsion"),
            Field("isp", 230.0, "real", "data", "propulsion"),
            Field("thrust_sl", 128600.0, "real", "data", "propulsion"),
            Field("thrust", 0.0, "real", "out", "propulsion", ("com",)),
            Field("mass", 0.0, "real", "out", "propulsion"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mprop = store.get("mprop")
        if mprop not in (0, 1):
            raise ValueError(f"mprop={mprop!r} not supported")
        pres_sl = store.get("pres_sl")
        aexit = store.get("aexit")
        mass_launch = store.get("mass_launch")
        mass_fuel = store.get("mass_fuel")
        isp = store.get("isp")
        thrust_sl = store.get("thrust_sl")
        launch_time = store.get("launch_time")
        press = store.get("press")
        mass = store.get("mass")
        thrust = 0.0
        if mprop == 1:
            mass_flow = thrust_sl / (isp * G0)
            mass = mass_launch - mass_flow * launch_time
            thrust = thrust_sl + (pres_sl - press) * aexit
            if mass <= mass_launch - mass_fuel:
                thrust = 0.0
                mprop = 0
        store.set("mass", mass)
        store.set("mprop", mprop)
        store.set("thrust", thrust)

    def terminate(self, vehicle, ctx):
        pass
