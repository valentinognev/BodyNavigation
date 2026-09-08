import math

import numpy as np

from cadac.constants import PI, R, RAD
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.stoch import dryden_white, prepare_for_dryden


class Agm6Environment:
    name = "environment"

    def __init__(self, weather_deck=None):
        self.weather_deck = weather_deck

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("plot",)
        scrn_plot = ("scrn", "plot")
        scrn_plot_com = ("scrn", "plot", "com")
        for field in (
            Field("mair", 0, "int", "data", "environment"),
            Field("warning_flag", 0, "int", "init", "environment"),
            Field("press", 0.0, "real", "out", "environment"),
            Field("rho", 0.0, "real", "out", "environment"),
            Field("vsound", 0.0, "real", "diag", "environment"),
            Field("grav", 0.0, "real", "out", "environment"),
            Field("vmach", 0.0, "real", "out", "environment", scrn_plot_com),
            Field("pdynmc", 0.0, "real", "out", "environment", scrn_plot),
            Field("tempk", 0.0, "real", "out", "environment"),
            Field("mfreeze_evrn", 0, "int", "save", "environment"),
            Field("pdynmcf", 0.0, "real", "save", "environment"),
            Field("vmachf", 0.0, "real", "save", "environment"),
            Field("GRAVL", zeros3, "vec", "out", "environment"),
            Field("dvae", 0.0, "real", "data", "environment"),
            Field("dvael", 0.0, "real", "data", "environment"),
            Field("waltl", 0.0, "real", "data", "environment"),
            Field("dvaeh", 0.0, "real", "data", "environment"),
            Field("walth", 0.0, "real", "data", "environment"),
            Field("vaed3", 0.0, "real", "data", "environment"),
            Field("psiwdx", 0.0, "real", "data", "environment"),
            Field("twind", 0.1, "real", "data", "environment"),
            Field("VAELS", zeros3, "vec", "state", "environment"),
            Field("VAELSD", zeros3, "vec", "state", "environment"),
            Field("VAEL", zeros3, "vec", "out", "environment", plot),
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
            Field("VBAL", zeros3, "vec", "out", "environment"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        store.set("dvba", store.get("dvbe"))

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mair = store.get("mair")
        matmo = mair // 100
        mturb = (mair - matmo * 100) // 10
        mwind = (mair - matmo * 100) % 10
        if matmo not in (0, 2) or mturb not in (0, 1) or mwind not in (0, 1, 2):
            raise ValueError(f"unknown mair {mair}")

        hbe = store.get("hbe")
        vbel = store.get("VBEL")
        int_step = ctx.int_step

        if matmo == 0:
            rho, press, tempk = atmosphere76(hbe)
            tempc = tempk - 273.16
            vsound = math.sqrt(1.4 * R * tempk)
        else:
            deck = self._weather()
            rho = deck.look_up("density", hbe)
            press = deck.look_up("pressure", hbe)
            tempc = deck.look_up("temperature", hbe)
            tempk = tempc + 273.16
            vsound = math.sqrt(1.4 * R * tempk)

        vaels = np.array(store.get("VAELS"), dtype=float, copy=True)
        vaelsd = np.array(store.get("VAELSD"), dtype=float, copy=True)
        vael = np.zeros(3)

        if mwind > 0:
            if mwind == 1:
                dvw = store.get("dvae")
                psiwdx = store.get("psiwdx")
            else:
                deck = self._weather()
                dvw = deck.look_up("speed", hbe)
                psiwdx = deck.look_up("direction", hbe)
            vaed_raw = np.array(
                [
                    -dvw * math.cos(psiwdx * RAD),
                    -dvw * math.sin(psiwdx * RAD),
                    store.get("vaed3"),
                ],
                dtype=float,
            )
            twind = store.get("twind")
            vaedsd_new = (vaed_raw - vaels) * (1.0 / twind)
            vaels = integrate(vaedsd_new, vaelsd, vaels, int_step)
            vaelsd = vaedsd_new
            vael = np.array(vaels, dtype=float, copy=True)
            store.set("VAELS", vaels)
            store.set("VAELSD", vaelsd)

        if mturb == 1:
            vtal = self._environment_dryden(store, store.get("dvba"), int_step)
            vael = vtal + vaels

        vbal = vbel - vael
        dvba = float(np.linalg.norm(vbal))
        vmach = abs(dvba / vsound)
        pdynmc = 0.5 * rho * dvba * dvba

        store.set("grav", gravity(hbe))
        store.set("rho", rho)
        store.set("press", press)
        store.set("tempk", tempk)
        store.set("tempc", tempc)
        store.set("vsound", vsound)
        store.set("VAEL", vael)
        store.set("VBAL", vbal)
        store.set("dvba", dvba)
        store.set("vmach", vmach)
        store.set("pdynmc", pdynmc)

    def terminate(self, vehicle, ctx):
        pass

    def _weather(self):
        if self.weather_deck is None:
            raise ValueError("weather_deck required")
        return self.weather_deck

    def _environment_dryden(self, store, dvba, int_step):
        turb_length = store.get("turb_length")
        turb_sigma = store.get("turb_sigma")
        tbd = store.get("TBD") if "TBD" in store.names() else store.get("TBL")
        alppx = store.get("alppx")
        phipx = store.get("phipx")
        prepare_for_dryden(store)
        gauss_value = dryden_white(int_step)
        taux1 = store.get("taux1")
        taux1d = store.get("taux1d")
        taux2 = store.get("taux2")
        taux2d = store.get("taux2d")

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
        vtal = tbd.T @ vtab

        store.set("taux1", taux1)
        store.set("taux1d", taux1d)
        store.set("taux2", taux2)
        store.set("taux2d", taux2d)
        store.set("tau", tau)
        store.set("gauss_value", gauss_value)
        return vtal
