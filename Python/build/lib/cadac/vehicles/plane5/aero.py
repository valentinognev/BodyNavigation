from cadac.kernel.state import Field


class Plane5Aero:
    name = "aerodynamics"

    def __init__(self, deck):
        self.deck = deck

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("cl", 0.0, "real", "out", "aerodynamics"),
            Field("cd", 0.0, "real", "out", "aerodynamics"),
            Field("cl_ov_cd", 0.0, "real", "diag", "aerodynamics", ("scrn", "plot")),
            Field("area", 27.87, "real", "data", "aerodynamics"),
            Field("mac", 0, "int", "data", "aerodynamics"),
            Field("cla", 0.0, "real", "out", "aerodynamics", ("scrn", "plot")),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mac = store.get("mac")
        mach = store.get("mach")
        alphax = store.get("alphax")
        cl = 0.0
        cd = 0.0
        cla = 0.0
        if mac == 30:
            cl = self.deck.look_up("cl_30MAC_vs_mach_alphax", mach, alphax)
            cd = self.deck.look_up("cd_30MAC_vs_mach_alphax", mach, alphax)
            clp = self.deck.look_up("cl_30MAC_vs_mach_alphax", mach, alphax + 2)
            cln = self.deck.look_up("cl_30MAC_vs_mach_alphax", mach, alphax - 2)
            cla = (clp - cln) / 4
        elif mac == 35:
            cl = self.deck.look_up("cl_35MAC_vs_mach_alphax", mach, alphax)
            cd = self.deck.look_up("cd_35MAC_vs_mach_alphax", mach, alphax)
            clp = self.deck.look_up("cl_35MAC_vs_mach_alphax", mach, alphax + 2)
            cln = self.deck.look_up("cl_35MAC_vs_mach_alphax", mach, alphax - 2)
            cla = (clp - cln) / 4
        elif mac == 40:
            cl = self.deck.look_up("cl_40MAC_vs_mach_alphax", mach, alphax)
            cd = self.deck.look_up("cd_40MAC_vs_mach_alphax", mach, alphax)
            clp = self.deck.look_up("cl_40MAC_vs_mach_alphax", mach, alphax + 2)
            cln = self.deck.look_up("cl_40MAC_vs_mach_alphax", mach, alphax - 2)
            cla = (clp - cln) / 4
        cl_ov_cd = cl / cd
        store.set("cl", cl)
        store.set("cd", cd)
        store.set("cla", cla)
        store.set("cl_ov_cd", cl_ov_cd)

    def terminate(self, vehicle, ctx):
        pass
