from cadac.kernel.state import Field


class Aim5Propulsion:
    name = "propulsion"

    def __init__(self, deck):
        self.deck = deck

    def define(self, vehicle):
        store = vehicle.store
        plot = ("scrn", "plot")
        for field in (
            Field("mprop", 0, "int", "diag", "propulsion"),
            Field("pres_sl", 101325, "real", "data", "propulsion"),
            Field("aexit", 0.0, "real", "data", "propulsion"),
            Field("thrust", 0.0, "real", "out", "propulsion", plot),
            Field("mass", 0.0, "real", "out", "propulsion", plot),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mprop = store.get("mprop")
        if mprop not in (0, 1):
            raise ValueError(f"unknown mprop {mprop}")

        pres_sl = store.get("pres_sl")
        aexit = store.get("aexit")
        time = store.get("time")
        press = store.get("press")
        mass = store.get("mass")

        thrust_sl = 0.0
        thrust = 0.0

        if mprop == 1:
            thrust_sl = self.deck.look_up("thrust_vs_time", time)
            thrust = thrust_sl + (pres_sl - press) * aexit
            mass = self.deck.look_up("mass_vs_time", time)

        if time > 0.0 and thrust_sl == 0:
            mprop = 0
            thrust = 0.0

        store.set("mprop", mprop)
        store.set("thrust", thrust)
        store.set("mass", mass)

    def terminate(self, vehicle, ctx):
        pass
