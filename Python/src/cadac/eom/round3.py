from cadac.env.gravity import gravity
from cadac.env.iso62 import iso62
from cadac.kernel.state import Field


class Round3Environment:
    name = "environment"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("time", 0.0, "real", "exec", "environment", ("scrn", "plot", "com")),
            Field("event_time", 0.0, "real", "exec", "environment", ("scrn",)),
            Field("int_step_new", 0.0, "real", "data", "environment"),
            Field("out_step_fact", 0.0, "real", "data", "environment"),
            Field("grav", 0.0, "real", "out", "environment"),
            Field("rho", 0.0, "real", "out", "environment"),
            Field("pdynmc", 0.0, "real", "out", "environment", ("scrn", "plot")),
            Field("mach", 0.0, "real", "out", "environment", ("scrn", "plot", "com")),
            Field("vsound", 0.0, "real", "diag", "environment"),
            Field("press", 0.0, "real", "diag", "environment"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        vehicle.store.set("time", ctx.sim_time)
        vehicle.store.set("int_step_new", ctx.int_step)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        ctx.int_step = store.get("int_step_new")
        ctx.out_fact = store.get("out_step_fact")
        alt = store.get("alt")
        dvbe = store.get("dvbe")
        atm = iso62(alt, dvbe)
        store.set("time", ctx.sim_time)
        store.set("event_time", ctx.event_time)
        store.set("grav", gravity(alt))
        store.set("rho", atm["rho"])
        store.set("pdynmc", atm["pdynmc"])
        store.set("mach", atm["mach"])
        store.set("vsound", atm["vsound"])
        store.set("press", atm["press"])

    def terminate(self, vehicle, ctx):
        pass
