from cadac.kernel.state import Field


class Sraam6Propulsion:
    name = "propulsion"

    def __init__(self, deck):
        self.deck = deck

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("mprop", 0, "int", "data", "propulsion"),
            Field("aexit", 0.0, "real", "data", "propulsion"),
            Field("vmass", 92.0, "real", "out", "propulsion"),
            Field("thrust", 0.0, "real", "out", "propulsion", ("scrn", "plot")),
            Field("xcgref", 1.536, "real", "init", "propulsion"),
            Field("xcg", 1.536, "real", "diag", "propulsion", ("scrn",)),
            Field("ai11", 0.308, "real", "out", "propulsion"),
            Field("ai33", 59.80, "real", "out", "propulsion"),
            Field("mfreeze_prop", 0, "int", "save", "propulsion"),
            Field("thrustf", 0.0, "real", "save", "propulsion"),
            Field("vmassf", 0.0, "real", "save", "propulsion"),
            Field("xcgf", 0.0, "real", "save", "propulsion"),
            Field("ai11f", 0.0, "real", "save", "propulsion"),
            Field("ai33f", 0.0, "real", "save", "propulsion"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        psl = 101325.0
        mprop = store.get("mprop")
        aexit = store.get("aexit")
        vmass = store.get("vmass")
        xcg = store.get("xcg")
        ai11 = store.get("ai11")
        ai33 = store.get("ai33")
        mfreeze_prop = store.get("mfreeze_prop")
        thrustf = store.get("thrustf")
        vmassf = store.get("vmassf")
        xcgf = store.get("xcgf")
        ai11f = store.get("ai11f")
        ai33f = store.get("ai33f")
        time = store.get("time")
        press = store.get("press")

        if mprop == 1:
            tsl = self.deck.look_up("thrust_vs_time", time)
            thrust = tsl + (psl - press) * aexit
            vmass = self.deck.look_up("mass_vs_time", time)
            xcg = self.deck.look_up("cg_vs_time", time)
            ai33 = self.deck.look_up("moipitch_vs_time", time)
            ai11 = self.deck.look_up("moiroll_vs_time", time)
            if time > 2.69:
                mprop = 0
                thrust = 0.0
        elif mprop == 0:
            thrust = 0.0
        else:
            raise ValueError(f"unknown mprop {mprop}")

        if "mfreeze" in store.names():
            mfreeze = store.get("mfreeze")
            if mfreeze == 0:
                mfreeze_prop = 0
            else:
                if mfreeze != mfreeze_prop:
                    mfreeze_prop = mfreeze
                    thrustf = thrust
                    vmassf = vmass
                    xcgf = xcg
                    ai11f = ai11
                    ai33f = ai33
                thrust = thrustf
                vmass = vmassf
                xcg = xcgf
                ai11 = ai11f
                ai33 = ai33f
            store.set("mfreeze_prop", mfreeze_prop)
            store.set("thrustf", thrustf)
            store.set("vmassf", vmassf)
            store.set("xcgf", xcgf)
            store.set("ai11f", ai11f)
            store.set("ai33f", ai33f)

        store.set("mprop", mprop)
        store.set("thrust", thrust)
        store.set("vmass", vmass)
        store.set("xcg", xcg)
        store.set("ai11", ai11)
        store.set("ai33", ai33)

    def terminate(self, vehicle, ctx):
        pass
