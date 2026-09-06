from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field

_ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


class Cruise5Control:
    name = "control"

    def define(self, vehicle):
        store = vehicle.store
        plot = ("scrn", "plot")
        for field in (
            Field("mcontrol", 0, "int", "data", "control", ("scrn",)),
            Field("psivgcx", 0.0, "real", "data", "control", ("plot",)),
            Field("thtvgcx", 0.0, "real", "data", "control", ("plot",)),
            Field("alphacx", 0.0, "real", "data", "control"),
            Field("phimvcx", 0.0, "real", "data", "control"),
            Field("TBV", _ZEROS33, "mat", "out", "control"),
            Field("TBG", _ZEROS33, "mat", "out", "control"),
            Field("gain_thtvg", 0.0, "real", "data", "control"),
            Field("gain_psivg", 0.0, "real", "data", "control"),
            Field("anx", 0.0, "real", "diag", "control", plot),
            Field("avx", 0.0, "real", "diag", "control", plot),
            Field("alphax", 0.0, "real", "out", "control", plot),
            Field("phimvx", 0.0, "real", "out", "control", plot),
            Field("phicx", 0.0, "real", "data", "control", plot),
            Field("phix", 0.0, "real", "state", "control", ("plot",)),
            Field("phixd", 0.0, "real", "state", "control"),
            Field("philimx", 0.0, "real", "data", "control"),
            Field("tphi", 0.0, "real", "data", "control"),
            Field("anposlimx", 0.0, "real", "data", "control"),
            Field("anneglimx", 0.0, "real", "data", "control"),
            Field("gacp", 0.0, "real", "data", "control"),
            Field("ta", 0.0, "real", "data", "control"),
            Field("xi", 0.0, "real", "state", "control"),
            Field("xid", 0.0, "real", "state", "control"),
            Field("qq", 0.0, "real", "diag", "control", ("plot",)),
            Field("tip", 0.0, "real", "diag", "control", ("plot",)),
            Field("alp", 0.0, "real", "state", "control", ("plot",)),
            Field("alpd", 0.0, "real", "state", "control"),
            Field("alpposlimx", 0.0, "real", "data", "control"),
            Field("alpneglimx", 0.0, "real", "data", "control"),
            Field("ancomx", 0.0, "real", "data", "control", plot),
            Field("alcomx", 0.0, "real", "data", "control", plot),
            Field("allimx", 0.0, "real", "data", "control"),
            Field("gcp", 0.0, "real", "data", "control"),
            Field("alx", 0.0, "real", "diag", "control", ("plot",)),
            Field("altdlim", 0.0, "real", "data", "control"),
            Field("gh", 0.0, "real", "data", "control"),
            Field("gv", 0.0, "real", "data", "control"),
            Field("altd", 0.0, "real", "diag", "control", ("plot",)),
            Field("altcom", 0.0, "real", "data", "control", ("plot",)),
        ):
            if field.name not in store.names():
                store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        pass

    def terminate(self, vehicle, ctx):
        pass

    def control_bank(self, vehicle, phicx, int_step):
        store = vehicle.store
        philimx = store.get("philimx")
        tphi = store.get("tphi")
        phix = store.get("phix")
        phixd = store.get("phixd")
        if phicx > philimx:
            phicx = philimx
        if phicx < -philimx:
            phicx = -philimx
        phixd_new = (phicx - phix) / tphi
        phix = integrate(phixd_new, phixd, phix, int_step)
        store.set("phix", phix)
        store.set("phixd", phixd_new)
        return phix
