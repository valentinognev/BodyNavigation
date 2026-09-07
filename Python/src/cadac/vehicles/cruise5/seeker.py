from cadac.kernel.state import Field


class Cruise5Seeker:
    name = "seeker"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("mseeker", 0, "int", "data/save", "seeker", ("scrn",)),
            Field("acq_range", 0.0, "real", "data", "seeker"),
            Field("range_go", 0.0, "real", "out", "seeker", ("plot", "scrn")),
            Field("STBG", zeros3, "vec", "out", "seeker", ("plot",)),
            Field("WOEB", zeros3, "vec", "out", "seeker"),
            Field("closing_speed", 0.0, "real", "out", "seeker"),
            Field("time_go", 0.0, "real", "out", "seeker", ("plot", "scrn")),
            Field("psisbx", 0.0, "real", "out", "seeker", ("plot", "scrn")),
            Field("thtsbx", 0.0, "real", "out", "seeker", ("plot", "scrn")),
            Field("targ_com_slot", 0, "int", "save", "seeker"),
            Field("UTBB", zeros3, "vec", "out", "seeker"),
            Field("acquisition", 0, "int", "init/save", "seeker", ("scrn",)),
        ):
            if field.name not in store.names():
                store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        mseeker = vehicle.store.get("mseeker")
        if mseeker == 0:
            return
        raise ValueError(f"unknown mseeker {mseeker}")

    def terminate(self, vehicle, ctx):
        pass
