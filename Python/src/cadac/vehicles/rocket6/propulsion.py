import numpy as np

from cadac.constants import AGRAV
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field


class Rocket6Propulsion:
    name = "propulsion"

    def __init__(self):
        pass

    def define(self, vehicle):
        store = vehicle.store
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        plot = ("scrn", "plot")
        for field in (
            Field("mprop", 0, "int", "data", "propulsion"),
            Field("acowl", 0.0, "real", "data", "propulsion"),
            Field("vmass", 0.0, "real", "out", "propulsion", plot),
            Field("vmass0", 0.0, "real", "data", "propulsion"),
            Field("xcg", 0.0, "real", "out", "propulsion", ("plot",)),
            Field("IBBB", zeros33, "mat", "out", "propulsion"),
            Field("fmass0", 0.0, "real", "data", "propulsion"),
            Field("fmasse", 0.0, "real", "state", "propulsion", plot),
            Field("fmassd", 0.0, "real", "state", "propulsion"),
            Field("spi", 0.0, "real", "data", "propulsion"),
            Field("thrust", 0.0, "real", "out", "propulsion", plot),
            Field("fmassr", 0.0, "real", "save", "propulsion", plot),
            Field("xcg_0", 0.0, "real", "data", "propulsion"),
            Field("xcg_1", 0.0, "real", "data", "propulsion"),
            Field("fuel_flow_rate", 0.0, "real", "data", "propulsion"),
            Field("vmass0_st", 0.0, "real", "data", "propulsion"),
            Field("fmass0_st", 0.0, "real", "data", "propulsion"),
            Field("moi_roll_0", 0.0, "real", "data", "propulsion"),
            Field("moi_roll_1", 0.0, "real", "data", "propulsion"),
            Field("moi_trans_0", 0.0, "real", "data", "propulsion"),
            Field("moi_trans_1", 0.0, "real", "data", "propulsion"),
            Field("mfreeze_prop", 0, "int", "save", "propulsion"),
            Field("thrustf", 0.0, "real", "save", "propulsion"),
            Field("vmassf", 0.0, "real", "save", "propulsion"),
            Field("IBBBF", zeros33, "mat", "save", "propulsion"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        dt = ctx.int_step
        mprop = store.get("mprop")
        if mprop not in (0, 3, 4):
            raise ValueError(f"unknown mprop {mprop}")

        vmass0 = store.get("vmass0")
        fmass0 = store.get("fmass0")
        spi = store.get("spi")
        xcg_0 = store.get("xcg_0")
        xcg_1 = store.get("xcg_1")
        fuel_flow_rate = store.get("fuel_flow_rate")
        moi_roll_0 = store.get("moi_roll_0")
        moi_roll_1 = store.get("moi_roll_1")
        moi_trans_0 = store.get("moi_trans_0")
        moi_trans_1 = store.get("moi_trans_1")
        vmass = store.get("vmass")
        xcg = store.get("xcg")
        ibbb = np.array(store.get("IBBB"), dtype=float)
        fmassr = store.get("fmassr")
        mfreeze_prop = store.get("mfreeze_prop")
        thrustf = store.get("thrustf")
        vmassf = store.get("vmassf")
        ibbbf = np.array(store.get("IBBBF"), dtype=float)
        fmasse = store.get("fmasse")
        fmassd = store.get("fmassd")

        thrust = 0.0
        if mprop == 0:
            fmassd = 0.0
            thrust = 0.0
            fmasse = 0.0
            fmassr = 0.0
        if mprop > 0:
            if mprop == 3 or mprop == 4:
                thrust = spi * fuel_flow_rate * AGRAV
                ibbb0 = np.zeros((3, 3))
                ibbb0[0, 0] = moi_roll_0
                ibbb0[1, 1] = moi_trans_0
                ibbb0[2, 2] = moi_trans_0
                ibbb1 = np.zeros((3, 3))
                ibbb1[0, 0] = moi_roll_1
                ibbb1[1, 1] = moi_trans_1
                ibbb1[2, 2] = moi_trans_1
            if spi != 0:
                fmassd_next = thrust / (spi * AGRAV)
                fmasse = integrate(fmassd_next, fmassd, fmasse, dt)
                fmassd = fmassd_next
            vmass = vmass0 - fmasse
            fmassr = fmass0 - fmasse
            mass_ratio = fmasse / fmass0
            ibbb = ibbb0 + (ibbb1 - ibbb0) * mass_ratio
            xcg = xcg_0 + (xcg_1 - xcg_0) * mass_ratio
            if fmassr <= 0:
                mprop = 0
                thrust = 0.0

        if "mfreeze" in store.names():
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

        store.set("fmasse", fmasse)
        store.set("fmassd", fmassd)
        store.set("mprop", mprop)
        store.set("fmassr", fmassr)
        store.set("mfreeze_prop", mfreeze_prop)
        store.set("thrustf", thrustf)
        store.set("vmassf", vmassf)
        store.set("IBBBF", ibbbf)
        store.set("vmass", vmass)
        store.set("xcg", xcg)
        store.set("IBBB", ibbb)
        store.set("thrust", thrust)

    def terminate(self, vehicle, ctx):
        pass
