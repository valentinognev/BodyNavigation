from cadac.kernel.state import Field

PSL = 101325.0


class Sam6Propulsion:
    name = "propulsion"

    def __init__(self, deck):
        self.deck = deck

    def define(self, vehicle):
        store = vehicle.store
        plot_scrn = ("plot", "scrn")
        scrn_plot = ("scrn", "plot")
        plot = ("plot",)
        for field in (
            Field("mprop", 0, "int", "out", "propulsion"),
            Field("aexit", 0.0314, "real", "data", "propulsion"),
            Field("mass", 300.0, "real", "out", "propulsion", plot_scrn),
            Field("thrust", 0.0, "real", "out", "propulsion", scrn_plot),
            Field("xcgref", 0.0, "real", "data", "propulsion"),
            Field("xcg", 2.9, "real", "diag", "propulsion", scrn_plot),
            Field("ai11", 2.9, "real", "out", "propulsion", plot),
            Field("ai33", 440.0, "real", "out", "propulsion", plot),
            Field("mfreeze_prop", 0, "int", "save", "propulsion"),
            Field("thrustf", 0.0, "real", "save", "propulsion"),
            Field("massf", 0.0, "real", "save", "propulsion"),
            Field("xcgf", 0.0, "real", "save", "propulsion"),
            Field("ai11f", 0.0, "real", "save", "propulsion"),
            Field("ai33f", 0.0, "real", "save", "propulsion"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        aexit = store.get("aexit")
        mass = store.get("mass")
        xcg = store.get("xcg")
        ai11 = store.get("ai11")
        ai33 = store.get("ai33")
        mfreeze_prop = store.get("mfreeze_prop")
        thrustf = store.get("thrustf")
        massf = store.get("massf")
        xcgf = store.get("xcgf")
        ai11f = store.get("ai11f")
        ai33f = store.get("ai33f")
        msl_time = store.get("msl_time")
        press = store.get("press")

        tsl = self.deck.look_up("thrust_vs_time", msl_time)
        thrust = tsl + (PSL - press) * aexit
        mass = self.deck.look_up("mass_vs_time", msl_time)
        xcg = self.deck.look_up("cg_vs_time", msl_time)
        ai33 = self.deck.look_up("moipitch_vs_time", msl_time)
        ai11 = self.deck.look_up("moiroll_vs_time", msl_time)
        if msl_time <= 60:
            mprop = 1
        else:
            mprop = 0

        if "mfreeze" in store:
            mfreeze = store.get("mfreeze")
            if mfreeze == 0:
                mfreeze_prop = 0
            else:
                if mfreeze != mfreeze_prop:
                    mfreeze_prop = mfreeze
                    thrustf = thrust
                    massf = mass
                    xcgf = xcg
                    ai11f = ai11
                    ai33f = ai33
                thrust = thrustf
                mass = massf
                xcg = xcgf
                ai11 = ai11f
                ai33 = ai33f

        store.set("mprop", mprop)
        store.set("thrust", thrust)
        store.set("mass", mass)
        store.set("xcg", xcg)
        store.set("ai11", ai11)
        store.set("ai33", ai33)
        store.set("mfreeze_prop", mfreeze_prop)
        store.set("thrustf", thrustf)
        store.set("massf", massf)
        store.set("xcgf", xcgf)
        store.set("ai11f", ai11f)
        store.set("ai33f", ai33f)

    def terminate(self, vehicle, ctx):
        pass
