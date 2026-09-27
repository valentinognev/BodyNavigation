from cadac.constants import DEG, RAD
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import cadac_sign


class Aim5Control:
    name = "control"

    def define(self, vehicle):
        store = vehicle.store
        plot = ("scrn", "plot")
        for field in (
            Field("ta", 0.0, "real", "data", "control"),
            Field("tr", 0.0, "real", "data", "control"),
            Field("gacp", 0.0, "real", "data", "control"),
            Field("tip", 0.0, "real", "diag", "control", plot),
            Field("xi", 0.0, "real", "state", "control"),
            Field("xid", 0.0, "real", "state", "control"),
            Field("ratep", 0.0, "real", "state", "control"),
            Field("ratepd", 0.0, "real", "state", "control"),
            Field("alp", 0.0, "real", "state", "control"),
            Field("alpd", 0.0, "real", "state", "control"),
            Field("yi", 0.0, "real", "state", "control"),
            Field("yid", 0.0, "real", "state", "control"),
            Field("ratey", 0.0, "real", "state", "control"),
            Field("rateyd", 0.0, "real", "state", "control"),
            Field("bet", 0.0, "real", "state", "control"),
            Field("betd", 0.0, "real", "state", "control"),
            Field("alphax", 0.0, "real", "in/out", "control", plot),
            Field("betax", 0.0, "real", "in/out", "control", plot),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        alphax = store.get("alphax")
        betax = store.get("betax")
        store.set("alp", alphax * RAD)
        store.set("bet", betax * RAD)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        ta = store.get("ta")
        tr = store.get("tr")
        gacp = store.get("gacp")
        grav = store.get("grav")
        pdynmc = store.get("pdynmc")
        dvae = store.get("dvbe")
        area = store.get("area")
        alpmax = store.get("alpmax")
        cyaim = store.get("cyaim")
        cnaim = store.get("cnaim")
        cnalp = store.get("cnalp")
        cybet = store.get("cybet")
        thrust = store.get("thrust")
        mass = store.get("mass")
        ancomx = store.get("ancomx")
        alcomx = store.get("alcomx")
        xi = store.get("xi")
        xid = store.get("xid")
        ratep = store.get("ratep")
        ratepd = store.get("ratepd")
        alp = store.get("alp")
        alpd = store.get("alpd")
        yi = store.get("yi")
        yid = store.get("yid")
        ratey = store.get("ratey")
        rateyd = store.get("rateyd")
        bet = store.get("bet")
        betd = store.get("betd")
        int_step = ctx.int_step

        # Pitch acceleration controller
        tip = dvae * mass / (pdynmc * area * abs(cnalp) + thrust)
        fspz = -pdynmc * area * cnaim / mass
        gr = gacp * tip * tr / dvae
        gi = gr / ta
        abez = -ancomx * grav
        ep = abez - fspz
        xid_new = gi * ep
        xi = integrate(xid_new, xid, xi, int_step)
        xid = xid_new
        ratepc = -(ep * gr + xi)
        ratepd_new = (ratepc - ratep) / tr
        ratep = integrate(ratepd_new, ratepd, ratep, int_step)
        ratepd = ratepd_new
        alpd_new = (tip * ratep - alp) / tip
        alp = integrate(alpd_new, alpd, alp, int_step)
        alpd = alpd_new
        alphax = alp * DEG
        if abs(alphax) > alpmax:
            alphax = alpmax * cadac_sign(alphax)

        # Yaw acceleration controller
        tiy = dvae * mass / (pdynmc * area * abs(cybet) + thrust)
        fspy = pdynmc * area * cyaim / mass
        gr = gacp * tiy * tr / dvae
        gi = gr / ta
        abey = alcomx * grav
        ey = abey - fspy
        yid_new = gi * ey
        yi = integrate(yid_new, yid, yi, int_step)
        yid = yid_new
        rateyc = ey * gr + yi
        rateyd_new = (rateyc - ratey) / tr
        ratey = integrate(rateyd_new, rateyd, ratey, int_step)
        rateyd = rateyd_new
        betd_new = -(tiy * ratey + bet) / tiy
        bet = integrate(betd_new, betd, bet, int_step)
        betd = betd_new
        betax = bet * DEG
        if abs(betax) > alpmax:
            betax = alpmax * cadac_sign(betax)

        store.set("xi", xi)
        store.set("xid", xid)
        store.set("ratep", ratep)
        store.set("ratepd", ratepd)
        store.set("alp", alp)
        store.set("alpd", alpd)
        store.set("yi", yi)
        store.set("yid", yid)
        store.set("ratey", ratey)
        store.set("rateyd", rateyd)
        store.set("bet", bet)
        store.set("betd", betd)
        store.set("alphax", alphax)
        store.set("betax", betax)
        store.set("tip", tip)

    def terminate(self, vehicle, ctx):
        pass
