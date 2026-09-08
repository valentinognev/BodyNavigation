import numpy as np

from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field


class Agm6Propulsion:
    name = "propulsion"

    def __init__(self):
        pass

    def define(self, vehicle):
        store = vehicle.store
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        plot = ("scrn", "plot")
        for field in (
            Field("mprop", 0, "int", "data", "propulsion"),
            Field("aexit", 0.0, "real", "data", "propulsion"),
            Field("vmass", 0.0, "real", "out", "propulsion", plot),
            Field("thrust", 0.0, "real", "out", "propulsion", plot),
            Field("vmass0", 0.0, "real", "data", "propulsion"),
            Field("ai11", 0.0, "real", "out", "propulsion"),
            Field("ai33", 0.0, "real", "out", "propulsion"),
            Field("mfreeze_prop", 0, "int", "save", "propulsion"),
            Field("thrustf", 0.0, "real", "save", "propulsion"),
            Field("vmassf", 0.0, "real", "save", "propulsion"),
            Field("spi", 0.0, "real", "data", "propulsion"),
            Field("throtl", 0.0, "real", "data", "propulsion"),
            Field("thrsl", 0.0, "real", "data", "propulsion"),
            Field("fmass0", 0.0, "real", "data", "propulsion"),
            Field("fmasse", 0.0, "real", "state", "propulsion", ("plot",)),
            Field("fmassed", 0.0, "real", "state", "propulsion"),
            Field("IBBB", zeros33, "mat", "out", "propulsion"),
            Field("eng_ang_mom", 0.0, "real", "out", "propulsion"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        store.set("vmass", store.get("vmass0"))

    def execute(self, vehicle, ctx):
        store = vehicle.store
        dt = ctx.int_step
        mprop = store.get("mprop")
        if mprop not in (0, 1):
            raise ValueError(f"unknown mprop {mprop}")

        aexit = store.get("aexit")
        vmass = store.get("vmass")
        vmass0 = store.get("vmass0")
        ai11 = store.get("ai11")
        ai33 = store.get("ai33")
        spi = store.get("spi")
        throtl = store.get("throtl")
        thrsl = store.get("thrsl")
        fmass0 = store.get("fmass0")
        fmasse = store.get("fmasse")
        fmassed = store.get("fmassed")
        mfreeze_prop = store.get("mfreeze_prop")
        thrustf = store.get("thrustf")
        vmassf = store.get("vmassf")

        if mprop == 1:
            press = store.get("press")
            fmassed_new = thrsl * throtl / (spi * 9.81)
            fmasse = integrate(fmassed_new, fmassed, fmasse, dt)
            fmassed = fmassed_new
            vmass = vmass0 - fmasse
            thrust = thrsl * throtl + (101325 - press) * aexit
        else:
            thrust = 0.0

        if fmasse >= fmass0:
            mprop = 0

        ibbb = np.diag([ai11, ai33, ai33])
        eng_ang_mom = 0.0

        if "mfreeze" in store:
            mfreeze = store.get("mfreeze")
            if mfreeze == 0:
                mfreeze_prop = 0
            else:
                if mfreeze != mfreeze_prop:
                    mfreeze_prop = mfreeze
                    thrustf = thrust
                    vmassf = vmass
                thrust = thrustf
                vmass = vmassf

        store.set("fmasse", fmasse)
        store.set("fmassed", fmassed)
        store.set("mprop", mprop)
        store.set("thrust", thrust)
        store.set("vmass", vmass)
        store.set("mfreeze_prop", mfreeze_prop)
        store.set("thrustf", thrustf)
        store.set("vmassf", vmassf)
        store.set("IBBB", ibbb)
        store.set("eng_ang_mom", eng_ang_mom)

    def terminate(self, vehicle, ctx):
        pass
