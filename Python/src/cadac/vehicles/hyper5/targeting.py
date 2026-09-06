from cadac.kernel.state import Field


class Hyper5Targeting:
    name = "targeting"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("mtargeting", 0, "int", "data", "targeting", ("scrn", "plot")),
            Field("del_radius", 0.0, "real", "data", "targeting"),
            Field("clost_tgt_slot", 0, "int", "out", "targeting"),
            Field("tgtng_sat_slot", 0, "int", "out", "targeting"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        mtargeting = vehicle.store.get("mtargeting")
        if mtargeting == 0:
            return
        raise ValueError(f"unknown mtargeting {mtargeting}")

    def terminate(self, vehicle, ctx):
        pass
