from cadac.kernel.state import Field


class Cruise3Aero:
    name = "aerodynamics"

    def __init__(self, deck):
        self.deck = deck

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("cl", 0.0, "real", "out", "aerodynamics"),
            Field("cd", 0.0, "real", "out", "aerodynamics"),
            Field("cl_ov_cd", 0.0, "real", "diag", "aerodynamics", ("scrn", "plot")),
            Field("area", 0.0, "real", "data", "aerodynamics"),
            Field("cla", 0.0, "real", "out", "aerodynamics"),
            Field("alphax", 0.0, "real", "data", "aerodynamics"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        alphax = store.get("alphax")
        mach = store.get("mach")
        cd0 = self.deck.look_up("cd0_vs_mach", mach)
        cl0 = self.deck.look_up("cl0_vs_mach", mach)
        cla = self.deck.look_up("cla_vs_mach", mach)
        ckk = self.deck.look_up("ckk_vs_mach", mach)
        cla0 = self.deck.look_up("cla0_vs_mach", mach)
        cl = cla0 + cla * alphax
        cd = cd0 + ckk * (cl - cl0) ** 2
        cl_ov_cd = cl / cd
        store.set("cl", cl)
        store.set("cd", cd)
        store.set("cla", cla)
        store.set("cl_ov_cd", cl_ov_cd)

    def terminate(self, vehicle, ctx):
        pass
