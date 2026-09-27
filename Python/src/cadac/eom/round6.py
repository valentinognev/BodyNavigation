"""Zipfel 6-DOF round-Earth equations of motion (CADAC Round6)."""

import math

import numpy as np

from cadac.constants import AGRAV, DEG, EPS, PI, RAD, REARTH, WEII3, R
from cadac.env.us76 import atmosphere76
from cadac.kernel.integrate import integrate
from cadac.kernel.module import ModuleBase
from cadac.kernel.state import Field
from cadac.math.earth import cadtei, cadtge
from cadac.math.frames import (
    cadac_inverse,
    cadac_matmul,
    cadac_sign,
    mat2tr,
    mat3tr,
    polar_from_cart,
    skew,
)
from cadac.math.wgs84 import (
    GM,
    cad_geo84_in,
    cad_grav84,
    cad_in_geo84,
    cad_kepler,
    cad_tdi84,
    cad_tgi84,
)
from cadac.stoch import ROCKET6_MARKOV_COUNT, dryden_white, prepare_for_dryden

FOOT = 3.280834
NMILES = 5.399568e-4

# US 1976 Standard Atmosphere, NASA Marshall 2002 extension (CADAC us76_nasa2002).
_US76_ZS = (
    0.0, 11.019, 20.063, 32.162, 47.35,
    51.413, 71.802, 86.0, 91.0, 94.0,
    97.0, 100.0, 103.0, 106.0, 108.0,
    110.0, 112.0, 115.0, 120.0, 125.0,
    130.0, 135.0, 140.0, 145.0, 150.0,
    155.0, 160.0, 165.0, 170.0, 180.0,
    190.0, 210.0, 230.0, 265.0, 300.0,
    350.0, 400.0, 450.0, 500.0, 550.0,
    600.0, 650.0, 700.0, 750.0, 800.0,
    850.0, 900.0, 950.0, 1000.0,
)
_US76_TMS = (
    288.15, 216.65, 216.65, 228.65, 270.65,
    270.65, 214.65, 186.95, 186.87, 187.74,
    190.40, 195.08, 202.23, 212.89, 223.29,
    240.00, 264.00, 300.00, 360.00, 417.23,
    469.27, 516.59, 559.63, 598.78, 634.39,
    666.80, 696.29, 723.13, 747.57, 790.07,
    825.31, 878.84, 915.78, 955.20, 976.01,
    990.06, 995.83, 998.22, 999.24, 999.67,
    999.85, 999.93, 999.97, 999.99, 999.99,
    1000.0, 1000.0, 1000.0, 1000.0,
)
_US76_WMS = (
    28.9644, 28.9644, 28.9644, 28.9644, 28.9644,
    28.9644, 28.9644, 28.9522, 28.8890, 28.7830,
    28.6200, 28.3950, 28.1040, 27.7650, 27.5210,
    27.2680, 27.0200, 26.6800, 26.2050, 25.8030,
    25.4360, 25.0870, 24.7490, 24.4220, 24.1030,
    23.7920, 23.4880, 23.1920, 22.9020, 22.3420,
    21.8090, 20.8250, 19.9520, 18.6880, 17.7260,
    16.7350, 15.9840, 15.2470, 14.3300, 13.0920,
    11.5050, 9.7180, 7.9980, 6.5790, 5.5430,
    4.8490, 4.4040, 4.1220, 3.9400,
)
_US76_PS = (
    1013.25, 226.32, 54.7487, 8.68014,
    1.10905, 0.66938, 0.039564, 3.7338e-03,
    1.5381e-03, 9.0560e-04, 5.3571e-04, 3.2011e-04,
    1.9742e-04, 1.2454e-04, 9.3188e-05, 7.1042e-05,
    5.5547e-05, 4.0096e-05, 2.5382e-05, 1.7354e-05,
    1.2505e-05, 9.3568e-06, 7.2028e-06, 5.6691e-06,
    4.5422e-06, 3.6930e-06, 3.0395e-06, 2.5278e-06,
    2.1210e-06, 1.5271e-06, 1.1266e-06, 6.4756e-07,
    3.9276e-07, 1.7874e-07, 8.7704e-08, 3.4498e-08,
    1.4518e-08, 6.4468e-09, 3.0236e-09, 1.5137e-09,
    8.2130e-10, 4.8865e-10, 3.1908e-10, 2.2599e-10,
    1.7036e-10, 1.3415e-10, 1.0873e-10, 8.9816e-11,
    7.5138e-11,
)
_US76_RO = 6356.766
_US76_GO = 9.80665
_US76_WMO = 28.9644
_US76_RS = 8314.32


def _us76_quad(z, z0, z1, z2, f0, f1, f2):
    return (
        f0 * (z - z1) * (z - z2) / ((z0 - z1) * (z0 - z2))
        + f1 * (z - z0) * (z - z2) / ((z1 - z0) * (z1 - z2))
        + f2 * (z - z0) * (z - z1) / ((z2 - z0) * (z2 - z1))
    )


def _us76_nasa2002(alt_km):
    """CADAC ``us76_nasa2002``. Returns ``(check, rho, press, tempk, vsound)``."""
    z = alt_km
    if z < 0.0 or z > 1000.0:
        return 1, 0.0, 0.0, 0.0, 0.0
    upper = 48
    i = 0
    while upper - i > 1:
        test = (i + upper) >> 1
        if z > _US76_ZS[test]:
            i = test
        else:
            upper = test
    if i < 7:
        zl = _US76_RO * _US76_ZS[i] / (_US76_RO + _US76_ZS[i])
        zu = _US76_RO * _US76_ZS[i + 1] / (_US76_RO + _US76_ZS[i + 1])
        wm = _US76_WMO
        ht = (_US76_RO * z) / (_US76_RO + z)
        g = (_US76_TMS[i + 1] - _US76_TMS[i]) / (zu - zl)
        if g < 0.0 or g > 0.0:
            press = (
                _US76_PS[i]
                * math.pow(
                    _US76_TMS[i] / (_US76_TMS[i] + g * (ht - zl)),
                    (_US76_GO * _US76_WMO) / (_US76_RS * g * 0.001),
                )
                * 100.0
            )
        else:
            press = (
                _US76_PS[i]
                * math.exp(-(_US76_GO * _US76_WMO * (ht * 1000.0 - zl * 1000.0)) / (_US76_RS * _US76_TMS[i]))
                * 100.0
            )
        tempk = _US76_TMS[i] + g * (ht - zl)
    else:
        if i == 7:
            tempk = _US76_TMS[8]
        if i >= 8 and i < 15:
            tempk = 263.1905 - 76.3232 * math.sqrt(1.0 - math.pow((z - 91.0) / 19.9429, 2.0))
        if i >= 15 and i < 18:
            tempk = 240.0 + 12.0 * (z - 110.0)
        if i >= 18:
            xi = (z - 120.0) * (_US76_RO + 120.0) / (_US76_RO + z)
            tempk = 1000.0 - 640.0 * math.exp(-0.01875 * xi)
        j = i
        if i == 47:
            j = i - 1
        z0 = _US76_ZS[j]
        z1 = _US76_ZS[j + 1]
        z2 = _US76_ZS[j + 2]
        wma = _us76_quad(z, z0, z1, z2, _US76_WMS[j], _US76_WMS[j + 1], _US76_WMS[j + 2])
        alpa = _us76_quad(
            z, z0, z1, z2, math.log(_US76_PS[j]), math.log(_US76_PS[j + 1]), math.log(_US76_PS[j + 2])
        )
        alpb = alpa
        wmb = wma
        if i != 7 and i != 47:
            j = j - 1
            z0 = _US76_ZS[j]
            z1 = _US76_ZS[j + 1]
            z2 = _US76_ZS[j + 2]
            alpb = _us76_quad(
                z, z0, z1, z2, math.log(_US76_PS[j]), math.log(_US76_PS[j + 1]), math.log(_US76_PS[j + 2])
            )
            wmb = _us76_quad(z, z0, z1, z2, _US76_WMS[j], _US76_WMS[j + 1], _US76_WMS[j + 2])
        press = 100.0 * math.exp((alpa + alpb) / 2.0)
        wm = (wma + wmb) / 2.0
    rho = (wm * press) / (_US76_RS * tempk)
    vsound = math.sqrt(1.4 * press / rho)
    return 0, rho, press, tempk, vsound


def _cad_tip(incl, lon_anode, arg_peri):
    clon_anode = math.cos(lon_anode)
    slon_anode = math.sin(lon_anode)
    carg_peri = math.cos(arg_peri)
    sarg_peri = math.sin(arg_peri)
    cincl = math.cos(incl)
    sincl = math.sin(incl)
    tip = np.zeros((3, 3))
    tip[0, 0] = clon_anode * carg_peri - slon_anode * sarg_peri * cincl
    tip[0, 1] = -clon_anode * sarg_peri - slon_anode * carg_peri * cincl
    tip[0, 2] = slon_anode * sincl
    tip[1, 0] = slon_anode * carg_peri + clon_anode * sarg_peri * cincl
    tip[1, 1] = -slon_anode * sarg_peri + clon_anode * carg_peri * cincl
    tip[1, 2] = -clon_anode * sincl
    tip[2, 0] = sarg_peri * sincl
    tip[2, 1] = carg_peri * sincl
    tip[2, 2] = cincl
    return tip


def _cad_in_orb(semi, ecc, inclx, lon_anodex, arg_perix, true_anomx):
    """CADAC ``cad_in_orb``. Angles in degrees. Returns ``(SBII, VBII, parabola_flag)``."""
    pp = semi * (1.0 - ecc * ecc)
    c_true_anom = math.cos(true_anomx * RAD)
    s_true_anom = math.sin(true_anomx * RAD)
    dbi = pp / (1.0 + ecc * c_true_anom)
    sbip = np.array([dbi * c_true_anom, dbi * s_true_anom, 0.0])
    vbip = np.zeros(3)
    parabola_flag = 0
    if pp == 0.0:
        parabola_flag = 1
    else:
        dum = math.sqrt(GM / pp)
        vbip = np.array([-dum * s_true_anom, dum * (ecc + c_true_anom), 0.0])
    tip = _cad_tip(inclx * RAD, lon_anodex * RAD, arg_perix * RAD)
    sbii = cadac_matmul(tip, sbip)
    vbii = cadac_matmul(tip, vbip)
    return sbii, vbii, parabola_flag


def _cad_geo84vel_in(sbii, vbii, time):
    """CADAC ``cad_geo84vel_in``. Returns ``(dvbe, psivdx, thtvdx)`` with angles in degrees."""
    lon, lat, alt = cad_geo84_in(sbii, time)
    tdi = cad_tdi84(lon, lat, alt, time)
    weii = np.zeros((3, 3))
    weii[0, 1] = -WEII3
    weii[1, 0] = WEII3
    vbed = cadac_matmul(tdi, vbii - cadac_matmul(weii, sbii))
    polar = polar_from_cart(vbed)
    return float(polar[0]), DEG * float(polar[1]), DEG * float(polar[2])


class Round6Environment(ModuleBase):
    """Zipfel 6-DOF round Earth atmosphere (CADAC ``round6_environment``)."""

    name = "environment"
    fields = (
        Field("mair", 0, "int", "data", "environment"),
        Field("warning_flag", 0, "int", "init", "environment"),
        Field("press", 0.0, "real", "out", "environment"),
        Field("rho", 0.0, "real", "out", "environment"),
        Field("vsound", 0.0, "real", "diag", "environment"),
        Field("vmach", 0.0, "real", "out", "environment", ("scrn", "plot", "com")),
        Field("pdynmc", 0.0, "real", "out", "environment", ("scrn", "plot")),
        Field("tempk", 0.0, "real", "out", "environment"),
        Field("mfreeze_evrn", 0, "int", "save", "environment"),
        Field("pdynmcf", 0.0, "real", "save", "environment"),
        Field("vmachf", 0.0, "real", "save", "environment"),
        Field("GRAVG", (0.0, 0.0, 0.0), "vec", "out", "environment"),
        Field("grav", 0.0, "real", "out", "environment"),
        Field("dvae", 0.0, "real", "data", "environment"),
        Field("dvael", 0.0, "real", "data", "environment"),
        Field("waltl", 0.0, "real", "data", "environment"),
        Field("dvaeh", 0.0, "real", "data", "environment"),
        Field("walth", 0.0, "real", "data", "environment"),
        Field("vaed3", 0.0, "real", "data", "environment"),
        Field("psiwdx", 0.0, "real", "data", "environment"),
        Field("twind", 0.1, "real", "data", "environment"),
        Field("VAEDS", (0.0, 0.0, 0.0), "vec", "state", "environment"),
        Field("VAEDSD", (0.0, 0.0, 0.0), "vec", "state", "environment"),
        Field("VAED", (0.0, 0.0, 0.0), "vec", "out", "environment"),
        Field("dvba", 0.0, "real", "out", "environment"),
        Field("markov_value", 0.0, "real", "save", "environment"),
        Field("turb_length", 0.0, "real", "data", "environment"),
        Field("turb_sigma", 0.0, "real", "data", "environment"),
        Field("taux1", 0.0, "real", "state", "environment"),
        Field("taux1d", 0.0, "real", "state", "environment"),
        Field("taux2", 0.0, "real", "state", "environment"),
        Field("taux2d", 0.0, "real", "state", "environment"),
        Field("tau", 0.0, "real", "diag", "environment"),
        Field("gauss_value", 0.0, "real", "diag", "environment"),
        Field("tempc", 0.0, "real", "diag", "environment"),
    )

    def __init__(self, weather_deck=None) -> None:
        self.weather_deck = weather_deck

    def initialize(self, vehicle, ctx) -> None:
        store = vehicle.store
        store.set("dvba", store.get("dvbe"))

    def _environment_dryden(self, vehicle, dvba, int_step):
        store = vehicle.store
        turb_length = store.get("turb_length")
        turb_sigma = store.get("turb_sigma")
        tbd = store.get("TBD")
        alppx = store.get("alppx")
        phipx = store.get("phipx")
        markov_value = store.get("markov_value")
        taux1 = store.get("taux1")
        taux1d = store.get("taux1d")
        taux2 = store.get("taux2")
        taux2d = store.get("taux2d")
        prepare_for_dryden(markov_count=ROCKET6_MARKOV_COUNT)
        gauss_value = dryden_white(int_step)
        taux1d_new = taux2
        taux1 = integrate(taux1d_new, taux1d, taux1, int_step)
        taux1d = taux1d_new
        vl = dvba / turb_length
        taux2d_new = -vl * vl * taux1 - 2.0 * vl * taux2 + vl * vl * gauss_value
        taux2 = integrate(taux2d_new, taux2d, taux2, int_step)
        taux2d = taux2d_new
        tau = turb_sigma * math.sqrt(1.0 / (vl * PI)) * (
            taux1 + math.sqrt(3.0) * taux2 / vl
        )
        vtab = np.array(
            [
                -tau * math.sin(alppx * RAD),
                tau * math.sin(phipx * RAD) * math.cos(alppx * RAD),
                tau * math.cos(phipx * RAD) * math.cos(alppx * RAD),
            ],
            dtype=float,
        )
        vtad = tbd.T @ vtab
        store.set("markov_value", markov_value)
        store.set("taux1", taux1)
        store.set("taux1d", taux1d)
        store.set("taux2", taux2)
        store.set("taux2d", taux2d)
        store.set("tau", tau)
        store.set("gauss_value", gauss_value)
        return vtad

    def execute(self, vehicle, ctx) -> None:
        store = vehicle.store
        mair = store.get("mair")
        matmo = mair // 100
        mturb = (mair - matmo * 100) // 10
        mwind = (mair - matmo * 100) % 10
        mair0 = matmo == 0 and mturb == 0 and mwind == 0
        mair12 = matmo == 0 and mturb == 1 and mwind == 2
        mair100 = matmo == 1 and mturb == 0 and mwind == 0
        if not mair0 and not mair12 and not mair100:
            raise ValueError(f"unknown mair {mair}")
        if mair12 and self.weather_deck is None:
            raise ValueError("mair 12 requires a weather Datadeck")

        warning_flag = store.get("warning_flag")
        dvba = store.get("dvba")
        vaeds = store.get("VAEDS")
        vaedsd = store.get("VAEDSD")
        time = store.get("time")
        alt = store.get("alt")
        vbed = store.get("VBED")
        sbii = store.get("SBII")

        gravg = cad_grav84(sbii, time)
        grav = float(np.linalg.norm(gravg))

        if matmo == 1:
            check, rho, press, tempk, vsound = _us76_nasa2002(alt / 1000.0)
            tempc = tempk - 273.16
            if check:
                # HYPER6 exits and does not touch warning_flag. ROCKET6 warns once
                # and continues with the zeros us76_nasa2002 returned.
                if getattr(vehicle, "family", None) == "rocket6":
                    if warning_flag == 0:
                        warning_flag = 1
                else:
                    raise ValueError("altitude is outside us76_nasa2002 atmosphere")
        else:
            rho, press, tempk = atmosphere76(alt)
            tempc = tempk - 273.16
            vsound = math.sqrt(1.4 * R * tempk)

        if vsound == 0.0:
            vmach = math.inf if dvba != 0.0 else math.nan
        else:
            vmach = abs(dvba / vsound)
        pdynmc = 0.5 * rho * dvba * dvba

        vaed = np.zeros(3)
        if mwind > 0:
            int_step = ctx.int_step
            twind = store.get("twind")
            vaed3 = store.get("vaed3")
            dvw = self.weather_deck.look_up("speed", alt)
            psiwdx = self.weather_deck.look_up("direction", alt)
            vaed_raw = np.array(
                [
                    -dvw * math.cos(psiwdx * RAD),
                    -dvw * math.sin(psiwdx * RAD),
                    vaed3,
                ],
                dtype=float,
            )
            vaedsd_new = (vaed_raw - vaeds) * (1.0 / twind)
            vaeds = integrate(vaedsd_new, vaedsd, vaeds, int_step)
            vaedsd = vaedsd_new
            vaed = vaeds
        if mturb == 1:
            int_step = ctx.int_step
            vtad = self._environment_dryden(vehicle, dvba, int_step)
            vaed = vtad + vaeds

        vbad = vbed - vaed
        dvba = float(np.linalg.norm(vbad))
        if vsound == 0.0:
            vmach = math.inf if dvba != 0.0 else math.nan
        else:
            vmach = abs(dvba / vsound)
        pdynmc = 0.5 * rho * dvba * dvba

        if "trcode" in store and "mguid" in store and store.get("mguid") == 6:
            trcode = store.get("trcode")
            if vmach <= store.get("trmach"):
                trcode = 2.0
            if pdynmc <= store.get("trdynm"):
                trcode = 3.0
            store.set("trcode", trcode)

        if "mfreeze" in store:
            mfreeze = store.get("mfreeze")
            mfreeze_evrn = store.get("mfreeze_evrn")
            pdynmcf = store.get("pdynmcf")
            vmachf = store.get("vmachf")
            if mfreeze == 0:
                mfreeze_evrn = 0
            else:
                if mfreeze != mfreeze_evrn:
                    mfreeze_evrn = mfreeze
                    vmachf = vmach
                    pdynmcf = pdynmc
                vmach = vmachf
                pdynmc = pdynmcf
            store.set("mfreeze_evrn", mfreeze_evrn)
            store.set("pdynmcf", pdynmcf)
            store.set("vmachf", vmachf)

        store.set("warning_flag", warning_flag)
        store.set("VAEDS", vaeds)
        store.set("VAEDSD", vaedsd)
        store.set("press", press)
        store.set("rho", rho)
        store.set("vmach", vmach)
        store.set("pdynmc", pdynmc)
        store.set("GRAVG", gravg)
        store.set("grav", grav)
        store.set("VAED", vaed)
        store.set("dvba", dvba)
        store.set("vsound", vsound)
        store.set("tempk", tempk)
        store.set("tempc", tempc)


class Round6Kinematics(ModuleBase):
    """Zipfel 6-DOF round Earth DCM kinematics (CADAC ``round6_kinematics``)."""

    name = "kinematics"
    fields = (
        Field("time", 0.0, "real", "exec", "kinematics", ("scrn", "plot", "com")),
        Field("event_time", 0.0, "real", "exec", "kinematics"),
        Field("int_step_new", 0.0, "real", "data", "kinematics"),
        Field("out_step_fact", 0.0, "real", "data", "kinematics"),
        Field("TBD", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "out", "kinematics"),
        Field("TBI", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "state", "kinematics"),
        Field("TBID", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "state", "kinematics"),
        Field("ortho_error", 0.0, "real", "diag", "kinematics", ("scrn",)),
        Field("psibd", 0.0, "real", "diag", "kinematics", ("plot",)),
        Field("thtbd", 0.0, "real", "diag", "kinematics"),
        Field("phibd", 0.0, "real", "diag", "kinematics"),
        Field("psibdx", 0.0, "real", "in/di", "kinematics", ("scrn", "plot")),
        Field("thtbdx", 0.0, "real", "in/di", "kinematics", ("scrn", "plot")),
        Field("phibdx", 0.0, "real", "in/di", "kinematics", ("scrn", "plot")),
        Field("alppx", 0.0, "real", "out", "kinematics", ("plot",)),
        Field("phipx", 0.0, "real", "out", "kinematics"),
        Field("alphax", 0.0, "real", "init/diag", "kinematics", ("scrn", "plot")),
        Field("betax", 0.0, "real", "diag", "kinematics", ("scrn", "plot")),
        Field("alphaix", 0.0, "real", "diag", "kinematics", ("plot",)),
        Field("betaix", 0.0, "real", "diag", "kinematics", ("plot",)),
    )

    def initialize(self, vehicle, ctx) -> None:
        store = vehicle.store
        time = ctx.sim_time
        int_step_new = ctx.int_step
        psibdx = store.get("psibdx")
        thtbdx = store.get("thtbdx")
        phibdx = store.get("phibdx")
        lonx = store.get("lonx")
        latx = store.get("latx")
        alt = store.get("alt")
        tbd = mat3tr(psibdx * RAD, thtbdx * RAD, phibdx * RAD)
        tdi = cad_tdi84(lonx * RAD, latx * RAD, alt, time)
        tbi = cadac_matmul(tbd, tdi)
        store.set("time", time)
        store.set("int_step_new", int_step_new)
        store.set("TBD", tbd)
        store.set("TBI", tbi)

    def execute(self, vehicle, ctx) -> None:
        store = vehicle.store
        int_step_new = store.get("int_step_new")
        out_step_fact = store.get("out_step_fact")
        dvba = store.get("dvba")
        wbib = store.get("WBIB")
        lonx = store.get("lonx")
        latx = store.get("latx")
        alt = store.get("alt")
        vbed = store.get("VBED")
        vaed = store.get("VAED")
        vbii = store.get("VBII")
        tbi = store.get("TBI")
        tbid = store.get("TBID")

        time = ctx.sim_time
        ctx.int_step = int_step_new
        ctx.out_fact = out_step_fact
        int_step = ctx.int_step

        tbid_new = cadac_matmul(-skew(wbib), tbi)
        tbi = integrate(tbid_new, tbid, tbi, int_step)
        tbid = tbid_new

        unit = np.eye(3)
        ee = unit - cadac_matmul(tbi, tbi.T.copy())
        tbi = tbi + cadac_matmul(ee, tbi) * 0.5

        e1 = ee[0, 0]
        e2 = ee[1, 1]
        e3 = ee[2, 2]
        ortho_error = math.sqrt(e1 * e1 + e2 * e2 + e3 * e3)

        tdi = cad_tdi84(lonx * RAD, latx * RAD, alt, time)
        tbd = cadac_matmul(tbi, tdi.T.copy())
        tbd13 = tbd[0, 2]
        tbd11 = tbd[0, 0]
        tbd33 = tbd[2, 2]
        tbd12 = tbd[0, 1]
        tbd23 = tbd[1, 2]

        # C++ `if(fabs(tbd13)<1)`. Numpy TBI·TDIᵀ at vertical launch often
        # yields |tbd13| a few ulps below 1; that asin path divides by ~1e-8
        # cosine and corrupts roll/yaw (RCS Schmitt). Treat 1-1e-14 as |tbd13|>=1.
        if math.fabs(tbd13) < 1.0 - 1e-14:
            thtbd = math.asin(-tbd13)
            cthtbd = math.cos(thtbd)
        else:
            thtbd = PI / 2.0 * cadac_sign(-tbd13)
            cthtbd = EPS
        cpsi = tbd11 / cthtbd
        if math.fabs(cpsi) > 1.0:
            cpsi = 1.0 * cadac_sign(cpsi)
        cphi = tbd33 / cthtbd
        if math.fabs(cphi) > 1.0:
            cphi = 1.0 * cadac_sign(cphi)
        psibd = math.acos(cpsi) * cadac_sign(tbd12)
        phibd = math.acos(cphi) * cadac_sign(tbd23)
        psibdx = DEG * psibd
        thtbdx = DEG * thtbd
        phibdx = DEG * phibd

        vbab = cadac_matmul(tbd, vbed - vaed)
        vbab1 = vbab[0]
        vbab2 = vbab[1]
        vbab3 = vbab[2]
        alpha = math.atan2(vbab3, vbab1)
        beta = math.asin(vbab2 / dvba)
        alphax = alpha * DEG
        betax = beta * DEG

        dum = vbab1 / dvba
        if math.fabs(dum) > 1.0:
            dum = 1.0 * cadac_sign(dum)
        alpp = math.acos(dum)
        if vbab2 == 0.0 and vbab3 == 0.0:
            phip = 0.0
        elif math.fabs(vbab2) < EPS:
            phip = 0.0
            if vbab3 > 0.0:
                phip = 0.0
            if vbab3 < 0.0:
                phip = PI
        else:
            phip = math.atan2(vbab2, vbab3)
        alppx = alpp * DEG
        phipx = phip * DEG

        vbib = cadac_matmul(tbi, vbii)
        vbib1 = vbib[0]
        vbib2 = vbib[1]
        vbib3 = vbib[2]
        alphai = math.atan2(vbib3, vbib1)
        dvbi = float(np.linalg.norm(vbib))
        betai = math.asin(vbib2 / dvbi)
        alphaix = alphai * DEG
        betaix = betai * DEG

        store.set("TBI", tbi)
        store.set("TBID", tbid)
        store.set("time", time)
        store.set("event_time", ctx.event_time)
        store.set("int_step_new", int_step_new)
        store.set("TBD", tbd)
        store.set("psibdx", psibdx)
        store.set("thtbdx", thtbdx)
        store.set("phibdx", phibdx)
        store.set("alppx", alppx)
        store.set("phipx", phipx)
        store.set("alphax", alphax)
        store.set("betax", betax)
        store.set("ortho_error", ortho_error)
        store.set("psibd", psibd)
        store.set("thtbd", thtbd)
        store.set("phibd", phibd)
        store.set("alphaix", alphaix)
        store.set("betaix", betaix)


class Round6Euler(ModuleBase):
    """Zipfel 6-DOF round Earth rigid-body Euler (CADAC ``round6_euler``)."""

    name = "euler"
    fields = (
        Field("ppx", 0.0, "real", "out", "euler", ("plot",)),
        Field("qqx", 0.0, "real", "out", "euler", ("plot",)),
        Field("rrx", 0.0, "real", "out", "euler", ("plot",)),
        Field("WBEB", (0.0, 0.0, 0.0), "vec", "diag", "euler"),
        Field("WBIB", (0.0, 0.0, 0.0), "vec", "state", "euler"),
        Field("WBIBD", (0.0, 0.0, 0.0), "vec", "state", "euler"),
        Field("WBII", (0.0, 0.0, 0.0), "vec", "out", "euler"),
    )

    def initialize(self, vehicle, ctx) -> None:
        store = vehicle.store
        ppx = store.get("ppx")
        qqx = store.get("qqx")
        rrx = store.get("rrx")
        tbi = store.get("TBI")
        wbeb = np.array([ppx * RAD, qqx * RAD, rrx * RAD], dtype=float)
        weii = np.array([0.0, 0.0, WEII3], dtype=float)
        wbib = wbeb + cadac_matmul(tbi, weii)
        store.set("WBIB", wbib)

    def execute(self, vehicle, ctx) -> None:
        store = vehicle.store
        fmb = store.get("FMB")
        tbi = store.get("TBI")
        ibbb = store.get("IBBB")
        wbib = store.get("WBIB")
        wbibd = store.get("WBIBD")
        int_step = ctx.int_step
        # cadac_inverse (adjoint/det) matches C++ Matrix::inverse(); np.linalg.inv
        # is LAPACK and 1 ulp off (UPDATES 0.168.12). Do not replace with numpy.
        wacc_next = cadac_matmul(
            cadac_inverse(ibbb),
            fmb - cadac_matmul(cadac_matmul(skew(wbib), ibbb), wbib),
        )
        wbib = integrate(wacc_next, wbibd, wbib, int_step)
        wbibd = wacc_next
        wbii = cadac_matmul(tbi.T.copy(), wbib)
        weii = np.array([0.0, 0.0, WEII3], dtype=float)
        wbeb = wbib - cadac_matmul(tbi, weii)
        store.set("WBIB", wbib)
        store.set("WBIBD", wbibd)
        store.set("ppx", wbeb[0] * DEG)
        store.set("qqx", wbeb[1] * DEG)
        store.set("rrx", wbeb[2] * DEG)
        store.set("WBEB", wbeb)
        store.set("WBII", wbii)


class Round6Newton(ModuleBase):
    """Zipfel 6-DOF round Earth translational Newton (CADAC ``round6_newton``)."""

    name = "newton"
    fields = (
        Field("minit", 0, "int", "data", "newton"),
        Field("alpha0x", 0.0, "real", "data", "newton"),
        Field("beta0x", 0.0, "real", "data", "newton"),
        Field("lonx", 0.0, "real", "init/diag", "newton", ("scrn", "plot", "com")),
        Field("latx", 0.0, "real", "init/diag", "newton", ("scrn", "plot", "com")),
        Field("alt", 0.0, "real", "init/out", "newton", ("scrn", "plot", "com")),
        Field("TVD", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "out", "newton"),
        Field("TDI", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "init", "newton"),
        Field("dvbe", 0.0, "real", "init/out", "newton", ("scrn", "plot", "com")),
        Field("dvbi", 0.0, "real", "out", "newton", ("scrn", "plot", "com")),
        Field("WEII", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "init", "newton"),
        Field("psivdx", 0.0, "real", "init/out", "newton", ("scrn", "plot", "com")),
        Field("thtvdx", 0.0, "real", "init/out", "newton", ("scrn", "plot", "com")),
        Field("dbi", 0.0, "real", "out", "newton", ("scrn", "plot")),
        Field("TGI", ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "mat", "init", "newton"),
        Field("VBED", (0.0, 0.0, 0.0), "vec", "out", "newton"),
        Field("altx", 0.0, "real", "diag", "newton"),
        Field("SBII", (0.0, 0.0, 0.0), "vec", "state", "newton", ("com",)),
        Field("VBII", (0.0, 0.0, 0.0), "vec", "state", "newton", ("com",)),
        Field("ABII", (0.0, 0.0, 0.0), "vec", "save", "newton"),
        Field("grndtrck", 0.0, "real", "diag", "newton"),
        Field("FSPB", (0.0, 0.0, 0.0), "vec", "out", "newton"),
        Field("ayx", 0.0, "real", "diag", "newton", ("plot",)),
        Field("anx", 0.0, "real", "diag", "newton", ("plot",)),
        Field("gndtrkmx", 0.0, "real", "diag", "newton"),
        Field("gndtrnmx", 0.0, "real", "diag", "newton"),
        Field("latx_bias", 0.0, "real", "data", "newton"),
        Field("dvbi_bias", 0.0, "real", "data", "newton"),
        Field("dbi_bias", 0.0, "real", "data", "newton"),
        Field("mfreeze_newt", 0, "int", "save", "newton"),
        Field("dvbef", 0.0, "real", "save", "newton"),
        Field("thtvdx_bias", 0.0, "real", "data", "newton"),
        Field("sat_semi", 0.0, "real", "data", "newton"),
        Field("sat_ecc", 0.0, "real", "data", "newton"),
        Field("sat_inclx", 0.0, "real", "data", "newton"),
        Field("sat_lon_anodex", 0.0, "real", "data", "newton"),
        Field("sat_arg_perix", 0.0, "real", "data", "newton"),
        Field("sat_true_anomx", 0.0, "real", "data", "newton"),
        Field("ranglex_l_t", 0.0, "real", "data", "newton"),
        Field("headon_flag", 0, "int", "data", "newton"),
        Field("tgo_insertion", 0.0, "real", "data", "newton"),
    )

    def initialize(self, vehicle, ctx) -> None:
        store = vehicle.store
        minit = store.get("minit")
        if minit not in (0, 1):
            raise ValueError(f"unknown minit {minit}")
        dvbe = store.get("dvbe")
        lonx = store.get("lonx")
        latx = store.get("latx")
        alt = store.get("alt")
        time = store.get("time")
        psibdx = store.get("psibdx")
        thtbdx = store.get("thtbdx")
        phibdx = store.get("phibdx")
        alpha0x = store.get("alpha0x")
        beta0x = store.get("beta0x")

        weii = np.zeros((3, 3))
        weii[0, 1] = -WEII3
        weii[1, 0] = WEII3

        if minit == 1:
            sat_semi = store.get("sat_semi")
            sat_ecc = store.get("sat_ecc")
            sat_inclx = store.get("sat_inclx")
            sat_lon_anodex = store.get("sat_lon_anodex")
            sat_arg_perix = store.get("sat_arg_perix")
            sat_true_anomx = store.get("sat_true_anomx")
            true_anomx = sat_true_anomx + store.get("ranglex_l_t")
            soii, voii, _parabola = _cad_in_orb(
                sat_semi, sat_ecc, sat_inclx, sat_lon_anodex, sat_arg_perix, true_anomx
            )
            lon, lat, _alt_unused = cad_geo84_in(soii, time)
            lonx = lon * DEG
            latx = lat * DEG
            tei = cadtei(time)
            tge = cadtge(lon, lat)
            tig = cadac_matmul(tei.T.copy(), tge.T.copy())
            voeg = cadac_matmul(tig.T.copy(), voii - cadac_matmul(weii, soii))
            polar_ovh = polar_from_cart(voeg)
            psibdx = float(polar_ovh[1]) * DEG
            headon_flag = store.get("headon_flag")
            if headon_flag:
                psibdx = psibdx - 180.0
            stii, vtii, _parabola = _cad_in_orb(
                sat_semi,
                sat_ecc,
                sat_inclx,
                sat_lon_anodex,
                sat_arg_perix,
                sat_true_anomx,
            )
            tgo_insertion = store.get("tgo_insertion")
            spii, vpii, _kepler = cad_kepler(stii, vtii, tgo_insertion)
            dvbi_pdct = float(np.linalg.norm(vpii))
            dbi_desired = float(np.linalg.norm(spii)) + store.get("dbi_bias")
            dvbi_desired = dvbi_pdct + store.get("dvbi_bias")
            _dvbe_pdct, _psivdx_pdct, thtvdx_pdct = _cad_geo84vel_in(spii, vpii, time)
            if headon_flag:
                thtvdx_desired = 0.0
            else:
                thtvdx_desired = thtvdx_pdct + store.get("thtvdx_bias")
            lonp, latp, _altp = cad_geo84_in(spii, tgo_insertion)
            wp_lonx = lonp * DEG
            wp_latx = latp * DEG + store.get("latx_bias")
            store.set("dbi_desired", dbi_desired)
            store.set("dvbi_desired", dvbi_desired)
            store.set("thtvdx_desired", thtvdx_desired)
            if "wp_lonx" in store:
                store.set("wp_lonx", wp_lonx)
                store.set("wp_latx", wp_latx)

        sbii = cad_in_geo84(lonx * RAD, latx * RAD, alt, time)
        dbi = float(np.linalg.norm(sbii))

        salp = math.sin(alpha0x * RAD)
        calp = math.cos(alpha0x * RAD)
        sbet = math.sin(beta0x * RAD)
        cbet = math.cos(beta0x * RAD)
        vbeb = np.array(
            [calp * cbet * dvbe, sbet * dvbe, salp * cbet * dvbe], dtype=float
        )
        tbd = mat3tr(psibdx * RAD, thtbdx * RAD, phibdx * RAD)
        vbed = cadac_matmul(tbd.T.copy(), vbeb)

        tdi = cad_tdi84(lonx * RAD, latx * RAD, alt, time)
        tgi = cad_tgi84(lonx * RAD, latx * RAD, alt, time)
        vbii = cadac_matmul(tdi.T.copy(), vbed) + cadac_matmul(weii, sbii)
        dvbi = float(np.linalg.norm(vbii))

        polar = polar_from_cart(vbed)
        psivdx = DEG * float(polar[1])
        thtvdx = DEG * float(polar[2])

        store.set("lonx", lonx)
        store.set("latx", latx)
        store.set("TDI", tdi)
        store.set("dvbi", dvbi)
        store.set("WEII", weii)
        store.set("psivdx", psivdx)
        store.set("thtvdx", thtvdx)
        store.set("dbi", dbi)
        store.set("TGI", tgi)
        store.set("VBED", vbed)
        store.set("SBII", sbii)
        store.set("VBII", vbii)
        store.set("psibdx", psibdx)

    def execute(self, vehicle, ctx) -> None:
        store = vehicle.store
        tdi = store.get("TDI")
        tgi = store.get("TGI")
        weii = store.get("WEII")
        grndtrck = store.get("grndtrck")
        mfreeze_newt = store.get("mfreeze_newt")
        dvbef = store.get("dvbef")
        sbii = store.get("SBII")
        vbii = store.get("VBII")
        abii = store.get("ABII")
        time = store.get("time")
        gravg = store.get("GRAVG")
        tbi = store.get("TBI")
        fapb = store.get("FAPB")
        vmass = store.get("vmass")
        int_step = ctx.int_step

        fspb = fapb * (1.0 / vmass)
        next_acc = cadac_matmul(tbi.T.copy(), fspb) + cadac_matmul(tgi.T.copy(), gravg)
        next_vel = integrate(next_acc, abii, vbii, int_step)
        sbii = integrate(next_vel, vbii, sbii, int_step)
        abii = next_acc
        vbii = next_vel
        dvbi = float(np.linalg.norm(vbii))
        dbi = float(np.linalg.norm(sbii))

        lon, lat, alt = cad_geo84_in(sbii, time)
        tdi = cad_tdi84(lon, lat, alt, time)
        tgi = cad_tgi84(lon, lat, alt, time)
        lonx = lon * DEG
        latx = lat * DEG
        altx = 0.001 * alt * FOOT

        vbed = cadac_matmul(tdi, vbii - cadac_matmul(weii, sbii))
        polar = polar_from_cart(vbed)
        dvbe = float(polar[0])
        psivdx = DEG * float(polar[1])
        thtvdx = DEG * float(polar[2])
        tvd = mat2tr(psivdx * RAD, thtvdx * RAD)

        ayx = fspb[1] / AGRAV
        anx = -fspb[2] / AGRAV
        grndtrck = (
            grndtrck
            + math.sqrt(vbed[0] * vbed[0] + vbed[1] * vbed[1]) * int_step * REARTH / dbi
        )
        gndtrkmx = 0.001 * grndtrck
        gndtrnmx = NMILES * grndtrck

        if "mfreeze" in store:
            mfreeze = store.get("mfreeze")
            if mfreeze == 0:
                mfreeze_newt = 0
            else:
                if mfreeze != mfreeze_newt:
                    mfreeze_newt = mfreeze
                    dvbef = dvbe
                dvbe = dvbef

        store.set("SBII", sbii)
        store.set("VBII", vbii)
        store.set("ABII", abii)
        store.set("grndtrck", grndtrck)
        store.set("mfreeze_newt", mfreeze_newt)
        store.set("dvbef", dvbef)
        store.set("lonx", lonx)
        store.set("latx", latx)
        store.set("alt", alt)
        store.set("TVD", tvd)
        store.set("TDI", tdi)
        store.set("dvbe", dvbe)
        store.set("dvbi", dvbi)
        store.set("TGI", tgi)
        store.set("VBED", vbed)
        store.set("FSPB", fspb)
        store.set("psivdx", psivdx)
        store.set("thtvdx", thtvdx)
        store.set("dbi", dbi)
        store.set("altx", altx)
        store.set("ayx", ayx)
        store.set("anx", anx)
        store.set("gndtrkmx", gndtrkmx)
        store.set("gndtrnmx", gndtrnmx)
