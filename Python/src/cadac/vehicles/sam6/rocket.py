from math import acos, atan2, cos, exp, fabs, pow, sin, sqrt, tan

import numpy as np

from cadac.constants import DEG, RAD
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field

SMALL = 1e-7
G0 = 9.81
CNALP0 = 7.468


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _skew(vec):
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


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


class Sam6RocketSensor:
    name = "sensor"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("mseek", 0, "int", "data", "sensor"),
            Field("stel1", 0.0, "real", "data", "sensor"),
            Field("stel2", 0.0, "real", "data", "sensor"),
            Field("stel3", 0.0, "real", "data", "sensor"),
            Field("dta", 0.0, "real", "out", "sensor", ("com",)),
            Field("dvta", 0.0, "real", "out", "sensor", ("com",)),
            Field("tgo_tgt", 9999.0, "real", "diag", "sensor"),
            Field("los_azx", 0.0, "real", "diag", "sensor"),
            Field("los_elx", 0.0, "real", "diag", "sensor"),
            Field("sigdy", 0.0, "real", "diag", "sensor"),
            Field("sigdz", 0.0, "real", "diag", "sensor"),
            Field("UTAA", zeros3, "vec", "out", "sensor"),
            Field("WOEA", zeros3, "vec", "out", "sensor"),
            Field("STAL", zeros3, "vec", "out", "sensor"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mseek = store.get("mseek")
        if mseek == 0:
            return
        flag_exo = store.get("flag_exo")
        alt = store.get("alt")
        alt_endo = store.get("alt_endo")
        if not (flag_exo and alt < alt_endo):
            return
        stel = np.array(
            [store.get("stel1"), store.get("stel2"), store.get("stel3")],
            dtype=float,
        )
        sael = np.asarray(store.get("SAEL"), dtype=float)
        vael = np.asarray(store.get("VAEL"), dtype=float)
        tal = np.asarray(store.get("TAL"), dtype=float)
        stal = stel - sael
        dta = float(np.linalg.norm(stal))
        if dta == 0.0:
            utal = np.zeros(3, dtype=float)
        else:
            utal = stal / dta
        utaa = tal @ utal
        vtael = vael * (-1.0)
        dvta = float(utal @ vtael)
        abs_dvta = fabs(dvta)
        if abs_dvta > SMALL:
            tgo_tgt = dta / abs_dvta
        else:
            tgo_tgt = 0.0
        woea = tal @ _skew(utal) @ vtael * (1.0 / dta)
        store.set("dta", dta)
        store.set("dvta", dvta)
        store.set("tgo_tgt", tgo_tgt)
        store.set("UTAA", utaa)
        store.set("WOEA", woea)
        store.set("STAL", stal)
        store.set("sigdy", float(woea[1]))
        store.set("sigdz", float(woea[2]))

    def terminate(self, vehicle, ctx):
        pass


class Sam6RocketGuidance:
    name = "guidance"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("mguide", 0, "int", "data", "guidance"),
            Field("gnav", 0.0, "real", "data", "guidance"),
            Field("tgo_manvr", 0.0, "real", "data", "guidance"),
            Field("amp_manvr", 0.0, "real", "data", "guidance"),
            Field("frq_manvr", 0.0, "real", "data", "guidance"),
            Field("tgo63_manvr", 0.0, "real", "data", "guidance"),
            Field("annx", 0.0, "real", "diag", "guidance"),
            Field("allx", 0.0, "real", "diag", "guidance"),
            Field("an_manvr", 0.0, "real", "diag", "guidance"),
            Field("al_manvr", 0.0, "real", "diag", "guidance"),
            Field("ancomx", 0.0, "real", "out", "guidance"),
            Field("alcomx", 0.0, "real", "out", "guidance"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mguide = store.get("mguide")
        guid_manvr = mguide // 10
        guid_mode = mguide % 10
        if guid_manvr not in (0, 1) or guid_mode not in (0, 1):
            raise ValueError(
                f"guid_manvr={guid_manvr!r} guid_mode={guid_mode!r} not supported"
            )
        gmax = store.get("gmax")
        gnav = store.get("gnav")
        tgo_manvr = store.get("tgo_manvr")
        amp_manvr = store.get("amp_manvr")
        frq_manvr = store.get("frq_manvr")
        tgo63_manvr = store.get("tgo63_manvr")
        annx = 0.0
        allx = 0.0
        an_manvr = 0.0
        al_manvr = 0.0
        if guid_mode == 1:
            grav = store.get("grav")
            dvta = store.get("dvta")
            utaa = np.asarray(store.get("UTAA"), dtype=float)
            woea = np.asarray(store.get("WOEA"), dtype=float)
            apna = _skew(woea) @ utaa * gnav * fabs(dvta)
            annx = -float(apna[2]) / grav
            allx = float(apna[1]) / grav
        if guid_manvr == 1:
            flag_exo = store.get("flag_exo")
            tgo_tgt = store.get("tgo_tgt")
            if flag_exo and tgo_tgt < tgo_manvr:
                amp = amp_manvr * (1.0 - exp(-tgo_tgt / tgo63_manvr))
                an_manvr = amp * sin(frq_manvr * tgo_tgt)
                al_manvr = amp * cos(frq_manvr * tgo_tgt)
                annx += an_manvr
                allx += al_manvr
        aax = sqrt(allx * allx + annx * annx)
        if aax > gmax:
            aax = gmax
        if fabs(annx) < SMALL or fabs(allx) < SMALL:
            phi = 0.0
        else:
            phi = atan2(annx, allx)
        alcomx = aax * cos(phi)
        ancomx = aax * sin(phi)
        store.set("ancomx", ancomx)
        store.set("alcomx", alcomx)
        store.set("annx", annx)
        store.set("allx", allx)
        store.set("an_manvr", an_manvr)
        store.set("al_manvr", al_manvr)

    def terminate(self, vehicle, ctx):
        pass


class Sam6RocketControl:
    name = "control"

    def define(self, vehicle):
        store = vehicle.store
        com = ("com",)
        for field in (
            Field("maut", 0, "int", "data", "control"),
            Field("flag_exo", 0, "int", "data", "control"),
            Field("ancomx_bias", 0.0, "real", "data", "control"),
            Field("alt_endo", 0.0, "real", "data", "control"),
            Field("tip", 0.0, "real", "diag", "control"),
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
            Field("alphax", 0.0, "real", "out", "control", com),
            Field("betax", 0.0, "real", "out", "control", com),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        maut = store.get("maut")
        if maut not in (0, 1):
            raise ValueError(f"maut={maut!r} not supported")
        flag_exo = store.get("flag_exo")
        ancomx_bias = store.get("ancomx_bias")
        alt_endo = store.get("alt_endo")
        grav = store.get("grav")
        pdynmc = store.get("pdynmc")
        dvae = store.get("dvae")
        alt = store.get("alt")
        area = store.get("area")
        alpmax = store.get("alpmax")
        cytgt = store.get("cytgt")
        cntgt = store.get("cntgt")
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
        tip = 0.0
        alphax = 0.0
        betax = 0.0
        if alt > alt_endo:
            flag_exo = 1
            xi = 0.0
            xid = 0.0
            ratep = 0.0
            ratepd = 0.0
            alp = 0.0
            alpd = 0.0
            yi = 0.0
            yid = 0.0
            ratey = 0.0
            rateyd = 0.0
            bet = 0.0
            betd = 0.0
        if not flag_exo:
            ancomx = ancomx_bias
        if maut == 1 and alt < alt_endo:
            tr = ((-2e-7) * pdynmc + 0.22)
            gacp = pow((0.002 * pdynmc), 0.575) * (1 - 0.5)
            ta = 2.2
            tip = dvae * mass / (pdynmc * area * fabs(cnalp) + thrust)
            fspz = -pdynmc * area * cntgt / mass
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
            if fabs(alphax) > alpmax:
                alphax = alpmax * _sign(alphax)
            tiy = dvae * mass / (pdynmc * area * fabs(cybet) + thrust)
            fspy = pdynmc * area * cytgt / mass
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
            if fabs(betax) > alpmax:
                betax = alpmax * _sign(betax)
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
        store.set("flag_exo", flag_exo)
        store.set("alphax", alphax)
        store.set("betax", betax)
        store.set("tip", tip)

    def terminate(self, vehicle, ctx):
        pass


class Sam6RocketForces:
    name = "forces"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("FSPA", (0.0, 0.0, 0.0), "vec", "out", "forces"),
            Field("aax", 0.0, "real", "diag", "forces"),
            Field("alx", 0.0, "real", "diag", "forces", ("com",)),
            Field("anx", 0.0, "real", "diag", "forces", ("com",)),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        thrust = store.get("thrust")
        catgt = store.get("catgt")
        pdynmc = store.get("pdynmc")
        area = store.get("area")
        mass = store.get("mass")
        grav = store.get("grav")
        cytgt = store.get("cytgt")
        cntgt = store.get("cntgt")
        fspa = np.array(
            [
                (thrust - catgt * pdynmc * area) / mass,
                (cytgt * pdynmc * area) / mass,
                (-cntgt * pdynmc * area) / mass,
            ],
            dtype=float,
        )
        store.set("FSPA", fspa)
        store.set("aax", fspa[0] / grav)
        store.set("alx", fspa[1] / grav)
        store.set("anx", -fspa[2] / grav)

    def terminate(self, vehicle, ctx):
        pass


class Sam6RocketIntercept:
    name = "intercept"

    def define(self, vehicle):
        vehicle.store.define(Field("write", 1, "int", "init", "intercept"))

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        write = store.get("write")
        dta = store.get("dta")
        mguide = store.get("mguide")
        dvta = store.get("dvta")
        alt = store.get("alt")
        if (dta < 1000) and (mguide > 0) and write:
            if dvta > 0:
                write = 0
                vehicle.health = 0
                ctx.combus[ctx.vehicle_slot].status = 0
        if (alt < 0) and write:
            write = 0
            vehicle.health = 0
            ctx.combus[ctx.vehicle_slot].status = 0
        store.set("write", write)

    def terminate(self, vehicle, ctx):
        pass
