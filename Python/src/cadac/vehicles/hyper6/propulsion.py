from math import cos

import numpy as np

from cadac.constants import AGRAV, RAD
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field


class Hyper6Propulsion:
    name = "propulsion"

    def __init__(self, deck):
        self.deck = deck

    def define(self, vehicle):
        store = vehicle.store
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        plot = ("scrn", "plot")
        for field in (
            Field("mprop", 0, "int", "data", "propulsion"),
            Field("acowl", 0.0, "real", "data", "propulsion"),
            Field("throttle", 0.05, "real", "data/diag", "propulsion", plot),
            Field("thrtl_max", 0.0, "real", "data", "propulsion"),
            Field("qhold", 0.0, "real", "data", "propulsion"),
            Field("vmass", 0.0, "real", "out", "propulsion", plot),
            Field("vmass0", 0.0, "real", "data", "propulsion"),
            Field("IBBB", zeros33, "mat", "out", "propulsion"),
            Field("IBBB0", zeros33, "mat", "init", "propulsion"),
            Field("IBBB1", zeros33, "mat", "init", "propulsion"),
            Field("fmass0", 0.0, "real", "data", "propulsion"),
            Field("fmasse", 0.0, "real", "state", "propulsion"),
            Field("fmassd", 0.0, "real", "state", "propulsion"),
            Field("ca", 0.0, "real", "diag", "propulsion"),
            Field("spi", 0.0, "real", "diag", "propulsion"),
            Field("thrust", 0.0, "real", "out", "propulsion"),
            Field("mass_flow", 0.0, "real", "diag", "propulsion"),
            Field("fmassr", 0.0, "real", "diag", "propulsion", plot),
            Field("thrustx", 0.0, "real", "diag", "propulsion", plot),
            Field("tq", 0.0, "real", "data", "propulsion"),
            Field("thrtl_idle", 0.0, "real", "data", "propulsion"),
            Field("fuel_flow_rate", 0.0, "real", "data", "propulsion"),
            Field("vmass0_st", 0.0, "real", "data", "propulsion"),
            Field("fmass0_st", 0.0, "real", "data", "propulsion"),
            Field("moi_roll_exo_0", 0.0, "real", "data", "propulsion"),
            Field("moi_roll_exo_1", 0.0, "real", "data", "propulsion"),
            Field("moi_trans_exo_0", 0.0, "real", "data", "propulsion"),
            Field("moi_trans_exo_1", 0.0, "real", "data", "propulsion"),
            Field("mfreeze_prop", 0, "int", "save", "propulsion"),
            Field("thrustf", 0.0, "real", "save", "propulsion"),
            Field("vmassf", 0.0, "real", "save", "propulsion"),
            Field("IBBBF", zeros33, "mat", "save", "propulsion"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        vmass0 = store.get("vmass0")
        ibbb0 = np.array(
            [
                [1.573e6, 0.0, 0.38e6],
                [0.0, 31.6e6, 0.0],
                [0.38e6, 0.0, 32.54e6],
            ],
            dtype=float,
        )
        ibbb1 = np.array(
            [
                [1.18e6, 0.0, 0.24e6],
                [0.0, 19.25e6, 0.0],
                [0.24e6, 0.0, 20.2e6],
            ],
            dtype=float,
        )
        store.set("vmass", vmass0)
        store.set("IBBB0", ibbb0)
        store.set("IBBB1", ibbb1)
        store.set("IBBB", ibbb0)
        store.set("vmass0_st", 0.0)
        store.set("fmass0_st", 0.0)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        dt = ctx.int_step
        mprop = store.get("mprop")
        if mprop not in (0, 1, 2):
            raise ValueError(f"unknown mprop {mprop}")

        throttle = store.get("throttle")
        qhold = store.get("qhold")
        vmass0 = store.get("vmass0")
        fmass0 = store.get("fmass0")
        tq = store.get("tq")
        thrtl_idle = store.get("thrtl_idle")
        acowl = store.get("acowl")
        thrtl_max = store.get("thrtl_max")
        ibbb = np.array(store.get("IBBB"), dtype=float)
        ibbb0 = np.array(store.get("IBBB0"), dtype=float)
        ibbb1 = np.array(store.get("IBBB1"), dtype=float)
        vmass = store.get("vmass")
        fmassr = store.get("fmassr")
        mfreeze_prop = store.get("mfreeze_prop")
        thrustf = store.get("thrustf")
        vmassf = store.get("vmassf")
        ibbbf = np.array(store.get("IBBBF"), dtype=float)
        fmasse = store.get("fmasse")
        fmassd = store.get("fmassd")
        rho = store.get("rho")
        vmach = store.get("vmach")
        pdynmc = store.get("pdynmc")
        dvba = store.get("dvba")
        refa = store.get("refa")
        cd = store.get("cd")
        alphax = store.get("alphax")

        spi = 0.0
        ca = 0.0
        thrust = 0.0
        mass_flow = 0.0

        if mprop > 0:
            if mprop == 1 or mprop == 2:
                spi = self.deck.look_up("spi_vs_throttle_mach", throttle, vmach)
                ca = self.deck.look_up("ca_vs_alpha_mach", alphax, vmach)
            if mprop == 1:
                thrust = spi * 0.029 * throttle * AGRAV * rho * dvba * ca * acowl
            if mprop == 2:
                denom = 0.029 * spi * AGRAV * rho * dvba * ca * acowl
                if denom != 0:
                    thrst_req = refa * cd * qhold / cos(alphax * RAD)
                    throtl_req = thrst_req / denom
                    gainq = 2 * vmass / (rho * dvba * denom * tq)
                    ethrotl = gainq * (qhold - pdynmc)
                    throttle = ethrotl + throtl_req
                if throttle < 0:
                    throttle = thrtl_idle
                if throttle > thrtl_max:
                    throttle = thrtl_max
                spi = self.deck.look_up("spi_vs_throttle_mach", throttle, vmach)
                thrust = spi * 0.029 * throttle * AGRAV * rho * dvba * ca * acowl
            if spi != 0:
                fmassd_next = thrust / (spi * AGRAV)
                fmasse = integrate(fmassd_next, fmassd, fmasse, dt)
                fmassd = fmassd_next
            vmass = vmass0 - fmasse
            fmassr = fmass0 - fmasse
            mass_ratio = fmasse / fmass0
            ibbb = ibbb0 + (ibbb1 - ibbb0) * mass_ratio
            mass_flow = thrust / (AGRAV * spi)
            if fmassr <= 0:
                mprop = 0
        if mprop == 0:
            fmassd = 0.0
            thrust = 0.0

        names = store.names()
        if "mfreeze" in names:
            mfreeze = store.get("mfreeze")
            if mfreeze == 0:
                mfreeze_prop = 0
            else:
                if mfreeze != mfreeze_prop:
                    mfreeze_prop = mfreeze
                    thrustf = thrust
                    vmassf = vmass
                    ibbbf = ibbb
                thrust = thrustf
                vmass = vmassf
                ibbb = ibbbf

        thrustx = thrust / 1000.0

        store.set("fmasse", fmasse)
        store.set("fmassd", fmassd)
        store.set("mprop", mprop)
        store.set("mfreeze_prop", mfreeze_prop)
        store.set("thrustf", thrustf)
        store.set("vmassf", vmassf)
        store.set("IBBBF", ibbbf)
        store.set("vmass", vmass)
        store.set("IBBB", ibbb)
        store.set("thrust", thrust)
        store.set("throttle", throttle)
        store.set("ca", ca)
        store.set("spi", spi)
        store.set("mass_flow", mass_flow)
        store.set("fmassr", fmassr)
        store.set("thrustx", thrustx)

    def terminate(self, vehicle, ctx):
        pass
