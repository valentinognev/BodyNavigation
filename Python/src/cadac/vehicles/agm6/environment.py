import math

import numpy as np

from cadac.constants import R
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.kernel.state import Field


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
        if matmo != 0 or mturb != 0 or mwind != 0:
            raise ValueError(f"unknown mair {mair}")

        hbe = store.get("hbe")
        vbel = store.get("VBEL")
        rho, press, tempk = atmosphere76(hbe)
        tempc = tempk - 273.16
        vsound = math.sqrt(1.4 * R * tempk)
        vael = np.zeros(3)
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
