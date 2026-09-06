from cadac.kernel.state import Field


class Plane6Guidance:
    name = "guidance"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("mguid", 0, "int", "data", "guidance", ("scrn",)),
            Field("line_gain", 0.0, "real", "data", "guidance"),
            Field("nl_gain_fact", 0.0, "real", "data", "guidance"),
            Field("decrement", 0.0, "real", "data", "guidance"),
            Field("swel1", 0.0, "real", "data", "guidance"),
            Field("swel2", 0.0, "real", "data", "guidance"),
            Field("swel3", 0.0, "real", "data", "guidance"),
            Field("psiflx", 0.0, "real", "data", "guidance"),
            Field("thtflx", 0.0, "real", "data", "guidance"),
            Field("dwb", 0.0, "real", "diag", "guidance"),
            Field("nl_gain", 0.0, "real", "diag", "guidance", ("scrn", "plot")),
            Field("VBEO", zeros3, "vec", "diag", "guidance"),
            Field("VBEF", zeros3, "vec", "diag", "guidance"),
            Field("dwbh", 0.0, "real", "diag", "guidance", ("scrn", "plot")),
            Field("SWBL", zeros3, "vec", "diag", "guidance"),
            Field("turn_min", 0.0, "real", "diag", "guidance"),
            Field("wp_flag", 0, "int", "diag", "guidance", ("plot",)),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        mguid = vehicle.store.get("mguid")
        if mguid == 0:
            return
        raise ValueError(f"unknown mguid {mguid}")

    def terminate(self, vehicle, ctx):
        pass
