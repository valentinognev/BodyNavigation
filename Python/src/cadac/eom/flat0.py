from cadac.kernel.state import Field


class Flat0Kinematics:
    name = "kinematics"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("time", 0.0, "real", "out", "kinematics", ("com",)),
            Field("launch_delay", 0.0, "real", "data", "kinematics"),
            Field("launch_epoch", 0.0, "real", "init", "kinematics"),
            Field("launch_time", 0.0, "real", "out", "kinematics"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        store.set("time", ctx.sim_time)
        store.set("launch_epoch", store.get("launch_delay"))

    def execute(self, vehicle, ctx):
        store = vehicle.store
        store.set("launch_time", ctx.sim_time - store.get("launch_epoch"))
        store.set("time", ctx.sim_time)

    def terminate(self, vehicle, ctx):
        pass


class Flat0Newton:
    name = "newton"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("srel1", 0.0, "real", "data", "newton"),
            Field("srel2", 0.0, "real", "data", "newton"),
            Field("srel3", 0.0, "real", "data", "newton"),
            Field("SREL", zeros3, "vec", "out", "newton"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        store.set(
            "SREL",
            (store.get("srel1"), store.get("srel2"), store.get("srel3")),
        )

    def execute(self, vehicle, ctx):
        pass

    def terminate(self, vehicle, ctx):
        pass
