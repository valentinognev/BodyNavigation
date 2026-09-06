from cadac.constants import DEG, RAD
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import cadtbv


class Hyper5Control:
    name = "control"

    def define(self, vehicle):
        store = vehicle.store
        plot = ("scrn", "plot")
        for field in (
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
            Field("alphax", 0.0, "real", "out", "control", plot),
            Field("alpposlimx", 0.0, "real", "data", "control"),
            Field("alpneglimx", 0.0, "real", "data", "control"),
            Field("xi", 0.0, "real", "state", "control"),
            Field("xid", 0.0, "real", "state", "control"),
            Field("alp", 0.0, "real", "state", "control"),
            Field("alpd", 0.0, "real", "state", "control"),
            Field("anx", 0.0, "real", "diag", "control", plot),
            Field("qq", 0.0, "real", "diag", "control", ("plot",)),
            Field("tip", 0.0, "real", "diag", "control", ("plot",)),
            Field("ancomx", 0.0, "real", "data", "control", plot),
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
        phixd = phixd_new
        store.set("phix", phix)
        store.set("phixd", phixd)
        return phix

    def control_load(self, vehicle, ancomx, int_step):
        store = vehicle.store
        anposlimx = store.get("anposlimx")
        anneglimx = store.get("anneglimx")
        phimvx = store.get("phimvx")
        gacp = store.get("gacp")
        ta = store.get("ta")
        alphax = store.get("alphax")
        alpposlimx = store.get("alpposlimx")
        alpneglimx = store.get("alpneglimx")
        fspv = store.get("FSPV")
        grav = store.get("grav")
        xi = store.get("xi")
        xid = store.get("xid")
        alp = store.get("alp")
        alpd = store.get("alpd")
        mass = store.get("mass")
        dvbe = store.get("dvbe")
        pdynmc = store.get("pdynmc")
        thrust = store.get("thrust")
        area = store.get("area")
        cla = store.get("cla")

        alpha = alphax * RAD
        phimv = phimvx * RAD
        tbv = cadtbv(phimv, alpha)
        fspb = tbv @ fspv

        if ancomx > anposlimx:
            ancomx = anposlimx
        if ancomx < anneglimx:
            ancomx = anneglimx

        fspb3 = fspb[2]
        anx = -fspb3 / grav
        eanx = ancomx - anx
        tip = dvbe * mass / (pdynmc * area * cla / RAD + thrust)

        gr = 0.0
        if ta > 0:
            gr = gacp * tip / dvbe
            gi = gr / ta
            xid_new = gi * eanx
            xi = integrate(xid_new, xid, xi, int_step)
            xid = xid_new
        else:
            xi = 0.0

        qq = gr * eanx + xi
        alpd_new = qq - alp / tip
        alp = integrate(alpd_new, alpd, alp, int_step)
        alpd = alpd_new

        alpx = alp * DEG
        if alpx > alpposlimx:
            alpx = alpposlimx
        if alpx < alpneglimx:
            alpx = alpneglimx

        store.set("xi", xi)
        store.set("xid", xid)
        store.set("alp", alp)
        store.set("alpd", alpd)
        store.set("anx", anx)
        store.set("qq", qq)
        store.set("tip", tip)
        return alpx
