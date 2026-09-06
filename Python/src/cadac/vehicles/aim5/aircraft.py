from math import atan2, sqrt

import numpy as np

from cadac.constants import DEG, EPS, RAD
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field


def _cadac_sign(variable):
    if variable < 0:
        return -1
    return 1


class Aim5AircraftForces:
    name = "forces"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("FSPV", (0.0, 0.0, 0.0), "vec", "out", "forces"),
            Field("acc_longx", 0.0, "real", "data", "forces"),
        ):
            if field.name not in store.names():
                store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        acc_longx = store.get("acc_longx")
        grav = store.get("grav")
        anx = store.get("anx")
        store.set("FSPV", np.array([acc_longx * grav, 0.0, -anx * grav]))

    def terminate(self, vehicle, ctx):
        pass


class Aim5AircraftControl:
    name = "control"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("phiav", 0.0, "real", "state", "control"),
            Field("phiavd", 0.0, "real", "state", "control"),
            Field("tphi", 0.0, "real", "data", "control"),
            Field("philimx", 0.0, "real", "data", "control"),
            Field("phiavx", 0.0, "real", "out", "control"),
            Field("phiavcx", 0.0, "real", "diag", "control"),
            Field("anx", 0.0, "real", "state", "control"),
            Field("anxd", 0.0, "real", "state", "control"),
            Field("tanx", 0.0, "real", "data", "control"),
            Field("alplimx", 0.0, "real", "data", "control"),
            Field("ancomx", 0.0, "real", "diag", "control"),
            Field("clalpha", 0.0, "real", "data", "control"),
            Field("wingloading", 0.0, "real", "data", "control"),
            Field("phiavout", 0.0, "real", "out", "control"),
        ):
            if field.name == "phiavout" and field.name in store.names():
                continue
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        tphi = store.get("tphi")
        philimx = store.get("philimx")
        tanx = store.get("tanx")
        alplimx = store.get("alplimx")
        clalpha = store.get("clalpha")
        wingloading = store.get("wingloading")
        grav = store.get("grav")
        pdynmc = store.get("pdynmc")
        tvl = store.get("TVL")
        acft_option = store.get("acft_option")
        acoml = store.get("ACOML")
        phiav = store.get("phiav")
        phiavd = store.get("phiavd")
        anx = store.get("anx")
        anxd = store.get("anxd")
        int_step = ctx.int_step

        acomv = tvl @ acoml
        acoma2 = float(acomv[1])
        acoma3 = float(acomv[2])
        if abs(acoma2) < EPS and abs(acoma3) < EPS:
            phiavc = 0.0
        else:
            phiavc = atan2(acoma2, -acoma3)
        phiavcx = phiavc * DEG

        if tphi:
            phiavd_new = (phiavc - phiav) / tphi
            phiav = integrate(phiavd_new, phiavd, phiav, int_step)
            phiavd = phiavd_new
        else:
            phiav = phiavc

        phiavx = phiav * DEG
        if abs(phiavx) >= philimx:
            phiavx = philimx * _cadac_sign(phiavx)
        phiavout = phiavx * RAD

        ancomx = sqrt(acoma2 * acoma2 + acoma3 * acoma3) / grav
        if tanx:
            anxd_new = (ancomx - anx) / tanx
            anx = integrate(anxd_new, anxd, anx, int_step)
            anxd = anxd_new
        else:
            anx = ancomx
        if acft_option > 0:
            anlimx = pdynmc * clalpha * alplimx / wingloading
            if abs(anx) >= anlimx:
                anx = anlimx * _cadac_sign(anx)

        store.set("phiav", phiav)
        store.set("phiavd", phiavd)
        store.set("anx", anx)
        store.set("anxd", anxd)
        store.set("phiavout", phiavout)
        store.set("phiavx", phiavx)
        store.set("phiavcx", phiavcx)
        store.set("ancomx", ancomx)

    def terminate(self, vehicle, ctx):
        pass
