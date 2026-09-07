import numpy as np

from cadac.eom.flat3 import Flat3Environment, Flat3Newton
from cadac.kernel.state import Field


def copy_in(store):
    names = store.names()
    _alias(store, names, "sael1", "sbel1")
    _alias(store, names, "sael2", "sbel2")
    _alias(store, names, "sael3", "sbel3")
    _alias(store, names, "dvae", "dvbe")
    _alias(store, names, "FSPA", "FSPV")
    if "SAEL" not in names:
        return
    sael = np.asarray(store.get("SAEL"), dtype=float)
    if np.all(sael == 0.0) and all(n in names for n in ("sael1", "sael2", "sael3")):
        sael = np.array(
            [store.get("sael1"), store.get("sael2"), store.get("sael3")],
            dtype=float,
        )
        store.set("SAEL", sael)
    if "SBEL" in names:
        store.set("SBEL", sael)


def copy_out(store):
    names = store.names()
    if "SBEL" in names:
        sbel = np.asarray(store.get("SBEL"), dtype=float)
        if "SAEL" in names:
            store.set("SAEL", sbel)
        if "sbel1" in names:
            store.set("sbel1", float(sbel[0]))
        if "sbel2" in names:
            store.set("sbel2", float(sbel[1]))
        if "sbel3" in names:
            store.set("sbel3", float(sbel[2]))
    _alias(store, names, "VBEL", "VAEL")
    _alias(store, names, "dvbe", "dvae")
    _alias(store, names, "TBL", "TAL")
    _alias(store, names, "TBV", "TAV")
    _alias(store, names, "ABEL", "AAEL")


def _alias(store, names, src, dst):
    if src in names and dst in names:
        store.set(dst, store.get(src))


class Agm6Flat3Environment:
    name = "environment"

    def __init__(self):
        self._inner = Flat3Environment()

    def define(self, vehicle):
        store = vehicle.store
        scrn_plot = ("scrn", "plot")
        for field in (
            Field("grav", 0.0, "real", "out", "environment"),
            Field("rho", 0.0, "real", "out", "environment"),
            Field("pdynmc", 0.0, "real", "out", "environment", scrn_plot),
            Field("mach", 0.0, "real", "out", "environment", ("scrn", "plot", "com")),
            Field("vsound", 0.0, "real", "diag", "environment"),
            Field("press", 0.0, "real", "out", "environment"),
        ):
            if field.name not in store.names():
                store.define(field)

    def initialize(self, vehicle, ctx):
        copy_in(vehicle.store)
        self._inner.initialize(vehicle, ctx)
        copy_out(vehicle.store)

    def execute(self, vehicle, ctx):
        copy_in(vehicle.store)
        self._inner.execute(vehicle, ctx)
        copy_out(vehicle.store)

    def terminate(self, vehicle, ctx):
        self._inner.terminate(vehicle, ctx)


class Agm6Flat3Newton:
    name = "newton"

    def __init__(self):
        self._inner = Flat3Newton()

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        com = ("com",)
        for field in (
            Field("TAL", zeros33, "mat", "out", "newton"),
            Field("TAV", zeros33, "mat", "diag", "newton"),
            Field("TVL", zeros33, "mat", "diag", "newton"),
            Field("dvae", 0.0, "real", "init/out", "newton"),
            Field("SAEL", zeros3, "vec", "state", "newton", com),
            Field("VAEL", zeros3, "vec", "state", "newton", com),
            Field("AAEL", zeros3, "vec", "state", "newton"),
            Field("psivlx", 0.0, "real", "init/diag", "newton", com),
            Field("thtvlx", 0.0, "real", "init/diag", "newton", com),
            Field("sael1", 0.0, "real", "init", "newton"),
            Field("sael2", 0.0, "real", "init", "newton"),
            Field("sael3", 0.0, "real", "init", "newton"),
            Field("psivl", 0.0, "real", "out", "newton"),
            Field("thtvl", 0.0, "real", "out", "newton"),
            Field("alt", 0.0, "real", "out", "newton", com),
        ):
            if field.name not in store.names():
                store.define(field)
        self._inner.define(vehicle)

    def initialize(self, vehicle, ctx):
        copy_in(vehicle.store)
        self._inner.initialize(vehicle, ctx)
        copy_out(vehicle.store)

    def execute(self, vehicle, ctx):
        copy_in(vehicle.store)
        self._inner.execute(vehicle, ctx)
        copy_out(vehicle.store)

    def terminate(self, vehicle, ctx):
        self._inner.terminate(vehicle, ctx)
