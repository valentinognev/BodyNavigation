from math import acos, atan2, cos, sin, tan

from cadac.constants import DEG, RAD
from cadac.kernel.state import Field
from cadac.math.frames import cadac_sign

SMALL = 1e-7


class Aim5Aero:
    name = "aerodynamics"

    def __init__(self, deck):
        self.deck = deck

    def define(self, vehicle):
        store = vehicle.store
        plot = ("scrn", "plot")
        for field in (
            Field("area", 0.0, "real", "data", "aerodynamics"),
            Field("alpmax", 0.0, "real", "data", "aerodynamics"),
            Field("alppx", 0.0, "real", "diag", "aerodynamics", plot),
            Field("phipx", 0.0, "real", "diag", "aerodynamics", plot),
            Field("cnpaim", 0.0, "real", "diag", "aerodynamics"),
            Field("claim", 0.0, "real", "out", "aerodynamics"),
            Field("cdaim", 0.0, "real", "out", "aerodynamics"),
            Field("caaim", 0.0, "real", "out", "aerodynamics"),
            Field("cyaim", 0.0, "real", "out", "aerodynamics"),
            Field("cnaim", 0.0, "real", "out", "aerodynamics"),
            Field("cnalp", 0.0, "real", "out", "aerodynamics"),
            Field("cybet", 0.0, "real", "out", "aerodynamics"),
            Field("gmax", 0.0, "real", "out", "aerodynamics", plot),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        area = store.get("area")
        alphax = store.get("alphax")
        grav = store.get("grav")
        pdynmc = store.get("pdynmc")
        mach = store.get("mach")
        mprop = store.get("mprop")
        mass = store.get("mass")
        betax = store.get("betax")
        alpmax = store.get("alpmax")

        alpha = alphax * RAD
        beta = betax * RAD
        alpp = acos(cos(alpha) * cos(beta))
        dum1 = tan(beta)
        dum2 = sin(alpha)
        if abs(dum2) < SMALL:
            dum2 = SMALL * cadac_sign(dum2)
        phip = atan2(dum1, dum2)

        alppx = alpp * DEG
        phipx = phip * DEG

        claim = self.deck.look_up("cl_aim_vs_alpha_mach", alppx, mach)
        if mprop:
            cdaim = self.deck.look_up("cd_aim_on_vs_alpha_mach", alppx, mach)
        else:
            cdaim = self.deck.look_up("cd_aim_off_vs_alpha_mach", alppx, mach)

        cos_alpha = cos(alpha)
        sin_alpha = sin(alpha)
        caaim = cdaim * cos_alpha - claim * sin_alpha
        cnpaim = cdaim * sin_alpha + claim * cos_alpha
        cnaim = abs(cnpaim) * cos(phip)
        cyaim = -abs(cnpaim) * sin(phip)

        claim_max = self.deck.look_up("cl_aim_vs_alpha_mach", alpmax, mach)
        if mprop:
            cdaim_max = self.deck.look_up("cd_aim_on_vs_alpha_mach", alpmax, mach)
        else:
            cdaim_max = self.deck.look_up("cd_aim_off_vs_alpha_mach", alpmax, mach)
        cnp_max = cdaim_max * sin(alpmax * RAD) + claim_max * cos(alpmax * RAD)

        falphax = abs(alphax)
        fbetax = abs(betax)
        if falphax < 10:
            cnalp = (0.123 + 0.013 * falphax) * DEG
        else:
            cnalp = 0.06 * falphax**0.625 * DEG
        if fbetax < 10:
            cybet = -(0.123 + 0.013 * fbetax) * DEG
        else:
            cybet = -0.06 * fbetax**0.625 * DEG

        gmax = (cnp_max * pdynmc * area) / (mass * grav)

        store.set("alppx", alppx)
        store.set("phipx", phipx)
        store.set("cnpaim", cnpaim)
        store.set("claim", claim)
        store.set("cdaim", cdaim)
        store.set("caaim", caaim)
        store.set("cyaim", cyaim)
        store.set("cnaim", cnaim)
        store.set("cnalp", cnalp)
        store.set("cybet", cybet)
        store.set("gmax", gmax)

    def terminate(self, vehicle, ctx):
        pass
