from math import atan2, cos

from cadac.constants import DEG, RAD
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import cadtbv

_ALLOWED_MCONTROL = (0, 3, 4, 6, 16, 36, 40, 44)
_ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


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
            Field("altdlim", 0.0, "real", "data", "control"),
            Field("gh", 0.0, "real", "data", "control"),
            Field("gv", 0.0, "real", "data", "control"),
            Field("altd", 0.0, "real", "diag", "control", ("plot",)),
            Field("altcom", 0.0, "real", "data", "control", ("plot",)),
            Field("gain_thtvg", 0.0, "real", "data", "control"),
            Field("gain_psivg", 0.0, "real", "data", "control"),
            Field("psivgcx", 0.0, "real", "data", "control", ("plot",)),
            Field("thtvgcx", 0.0, "real", "data", "control", ("plot",)),
            Field("avx", 0.0, "real", "diag", "control", ("scrn", "plot")),
            Field("mcontrol", 0, "int", "data", "control", ("scrn",)),
            Field("TBV", _ZEROS33, "mat", "out", "control"),
            Field("TBG", _ZEROS33, "mat", "out", "control"),
            Field("alcomx", 0.0, "real", "data", "control", ("scrn", "plot")),
            Field("allimx", 0.0, "real", "data", "control"),
            Field("gcp", 0.0, "real", "data", "control"),
            Field("alx", 0.0, "real", "diag", "control", ("plot",)),
            Field("alphacx", 0.0, "real", "data", "control"),
            Field("phimvcx", 0.0, "real", "data", "control"),
        ):
            if field.name not in store.names():
                store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mcontrol = store.get("mcontrol")
        if mcontrol not in _ALLOWED_MCONTROL:
            raise ValueError(f"unknown mcontrol {mcontrol}")
        int_step = ctx.int_step
        psivgcx = store.get("psivgcx")
        alphacx = store.get("alphacx")
        ancomx = store.get("ancomx")
        alcomx = store.get("alcomx")
        altcom = store.get("altcom")
        phicx = store.get("phicx")
        tgv = store.get("TGV")
        phimvx = 0.0
        alphax = 0.0
        if mcontrol == 0:
            phimvx = 0.0
            alphax = 0.0
        if mcontrol == 3:
            phimvx = self.control_bank(vehicle, phicx, int_step)
            alphax = alphacx
        if mcontrol == 4:
            alphax = self.control_load(vehicle, ancomx, int_step)
        if mcontrol == 40:
            phicx = self.control_lateral(vehicle, alcomx)
            phimvx = self.control_bank(vehicle, phicx, int_step)
        if mcontrol == 44:
            phicx = self.control_lateral(vehicle, alcomx)
            phimvx = self.control_bank(vehicle, phicx, int_step)
            alphax = self.control_load(vehicle, ancomx, int_step)
        if mcontrol == 6:
            ancomx = self.control_altitude(vehicle, altcom, phimvx)
            alphax = self.control_load(vehicle, ancomx, int_step)
        if mcontrol == 16:
            phicx = self.control_heading(vehicle, psivgcx)
            phimvx = self.control_bank(vehicle, phicx, int_step)
            ancomx = self.control_altitude(vehicle, altcom, phimvx)
            alphax = self.control_load(vehicle, ancomx, int_step)
        if mcontrol == 36:
            phimvx = self.control_bank(vehicle, phicx, int_step)
            ancomx = self.control_altitude(vehicle, altcom, phimvx)
            alphax = self.control_load(vehicle, ancomx, int_step)
        tbv = cadtbv(phimvx * RAD, alphax * RAD)
        tvg = tgv.T
        tbg = tbv @ tvg
        store.set("phicx", phicx)
        store.set("TBV", tbv)
        store.set("TBG", tbg)
        store.set("alphax", alphax)
        store.set("phimvx", phimvx)
        store.set("ancomx", ancomx)

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

    def control_altitude(self, vehicle, altcom, phimvx):
        store = vehicle.store
        anposlimx = store.get("anposlimx")
        anneglimx = store.get("anneglimx")
        altdlim = store.get("altdlim")
        gh = store.get("gh")
        gv = store.get("gv")
        alt = store.get("alt")
        grav = store.get("grav")
        vbeg = store.get("VBEG")

        ealt = gh * (altcom - alt)
        if ealt > altdlim:
            ealt = altdlim
        if ealt < -altdlim:
            ealt = -altdlim
        altd = -vbeg[2]
        ancomx = (gv * (ealt - altd) / grav + 1) * (1 / cos(phimvx * RAD))
        if ancomx > anposlimx:
            ancomx = anposlimx
        if ancomx < anneglimx:
            ancomx = anneglimx

        store.set("altd", altd)
        return ancomx

    def control_heading(self, vehicle, psivgcx):
        store = vehicle.store
        gain_psivg = store.get("gain_psivg")
        psivgx = store.get("psivgx")
        if abs(psivgcx) <= 135:
            psivgx_comp = psivgx
        else:
            if psivgx * psivgcx >= 0:
                psivgx_comp = psivgx
            else:
                if psivgx >= 0:
                    sign_psivgx = 1
                else:
                    sign_psivgx = -1
                psivgx_comp = 360 - psivgx * sign_psivgx
        return gain_psivg * (psivgcx - psivgx_comp)

    def control_flightpath(self, vehicle, thtvgcx, phimvx):
        store = vehicle.store
        gain_thtvg = store.get("gain_thtvg")
        alpposlimx = store.get("alpposlimx")
        alpneglimx = store.get("alpneglimx")
        pdynmc = store.get("pdynmc")
        thtvg = store.get("thtvg")
        grav = store.get("grav")
        mass = store.get("mass")
        area = store.get("area")
        cla = store.get("cla")
        avx = gain_thtvg * (thtvgcx * RAD - thtvg)
        anx = avx / cos(phimvx * RAD)
        alphax = (anx * mass * grav) / (pdynmc * area * cla)
        if alphax > alpposlimx:
            alphax = alpposlimx
        if alphax < alpneglimx:
            alphax = alpneglimx
        store.set("anx", anx)
        store.set("avx", avx)
        return alphax

    def control_lateral(self, vehicle, alcomx):
        store = vehicle.store
        allimx = store.get("allimx")
        fspv = store.get("FSPV")
        grav = store.get("grav")
        phimvx = store.get("phimvx")
        alphax = store.get("alphax")
        alpha = alphax * RAD
        phimv = phimvx * RAD
        tbv = cadtbv(phimv, alpha)
        fspb = tbv @ fspv
        fspb3 = fspb[2]
        anx = -fspb3 / grav
        if alcomx > allimx:
            alcomx = allimx
        if alcomx < -allimx:
            alcomx = -allimx
        phic = atan2(alcomx, anx)
        phicx = phic * DEG
        fspv2 = fspv[1]
        alx = fspv2 / grav
        store.set("alx", alx)
        return phicx
