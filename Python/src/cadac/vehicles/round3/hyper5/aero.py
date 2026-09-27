from math import cos, sin

from cadac.constants import RAD
from cadac.kernel.state import Field


class Hyper5Aero:
    name = "aerodynamics"

    def __init__(self, deck):
        self.deck = deck

    def define(self, vehicle):
        store = vehicle.store
        plot = ("scrn", "plot")
        for field in (
            Field("cl", 0.0, "real", "out", "aerodynamics", plot),
            Field("cd", 0.0, "real", "out", "aerodynamics", plot),
            Field("cl_ov_cd", 0.0, "real", "diag", "aerodynamics", plot),
            Field("area", 0.0, "real", "data", "aerodynamics"),
            Field("cla", 0.0, "real", "out", "aerodynamics", plot),
            Field("cn", 0.0, "real", "diag", "aerodynamics", plot),
            Field("ca", 0.0, "real", "diag", "aerodynamics", plot),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mach = store.get("mach")
        alphax = store.get("alphax")
        cn = self.deck.look_up("cn_rr3x_vs_alphax_mach", alphax, mach)
        ca = self.deck.look_up("ca_rr3x_vs_alphax_mach", alphax, mach)
        alpha = alphax * RAD
        cd = cn * sin(alpha) + ca * cos(alpha)
        cl = cn * cos(alpha) - ca * sin(alpha)
        cl_ov_cd = cl / cd
        cnp = self.deck.look_up("cn_rr3x_vs_alphax_mach", alphax + 2, mach)
        cnn = self.deck.look_up("cn_rr3x_vs_alphax_mach", alphax - 2, mach)
        cap = self.deck.look_up("ca_rr3x_vs_alphax_mach", alphax + 2, mach)
        can = self.deck.look_up("ca_rr3x_vs_alphax_mach", alphax - 2, mach)
        cna = (cnp - cnn) / 4
        caa = (cap - can) / 4
        cla = cna * cos(alpha) - caa * sin(alpha)
        store.set("cl", cl)
        store.set("cd", cd)
        store.set("cla", cla)
        store.set("cl_ov_cd", cl_ov_cd)
        store.set("cn", cn)
        store.set("ca", ca)

    def terminate(self, vehicle, ctx):
        pass
