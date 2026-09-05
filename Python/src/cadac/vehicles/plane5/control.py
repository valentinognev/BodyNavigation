from math import cos

from cadac.constants import DEG, RAD
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import cadtbv


class Plane5Control:
    name = "control"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("phimvx", 0.0, "real", "out", "control", ("scrn", "plot")),
            Field("phicx", 0.0, "real", "data", "control", ("scrn", "plot")),
            Field("phix", 0.0, "real", "state", "control", ("plot",)),
            Field("phixd", 0.0, "real", "state", "control"),
            Field("philimx", 0.0, "real", "data", "control"),
            Field("tphi", 0.0, "real", "data", "control"),
            Field("anx", 0.0, "real", "diag", "control", ("scrn", "plot")),
            Field("alphax", 0.0, "real", "out", "control", ("scrn", "plot")),
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
            Field("ancomx", 0.0, "real", "data", "control", ("plot",)),
            Field("altdlim", 0.0, "real", "data", "control"),
            Field("gh", 0.0, "real", "data", "control"),
            Field("gv", 0.0, "real", "data", "control"),
            Field("altd", 0.0, "real", "diag", "control", ("plot",)),
            Field("altcom", 0.0, "real", "data", "control", ("plot",)),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        phimvx = self.control_bank(vehicle, store.get("phicx"), ctx.int_step)
        store.set("phimvx", phimvx)

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

    def control_altitude(self, vehicle, altcom, phimvx):
        store = vehicle.store
        anposlimx = store.get("anposlimx")
        anneglimx = store.get("anneglimx")
        altdlim = store.get("altdlim")
        gh = store.get("gh")
        gv = store.get("gv")
        alt = store.get("alt")
        grav = store.get("grav")
        vbel = store.get("VBEL")

        ealt = gh * (altcom - alt)
        if ealt > altdlim:
            ealt = altdlim
        if ealt < -altdlim:
            ealt = -altdlim
        altd = -vbel[2]
        ancomx = (gv * (ealt - altd) / grav + 1) * (1 / cos(phimvx * RAD))
        if ancomx > anposlimx:
            ancomx = anposlimx
        if ancomx < anneglimx:
            ancomx = anneglimx

        store.set("altd", altd)
        return ancomx

    def terminate(self, vehicle, ctx):
        pass
