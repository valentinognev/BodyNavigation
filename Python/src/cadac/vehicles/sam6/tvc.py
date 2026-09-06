import numpy as np

from cadac.kernel.state import Field


class Sam6Tvc:
    name = "tvc"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("plot",)
        for field in (
            Field("mtvc", 0, "int", "data", "tvc"),
            Field("tvclimx", 0.0, "real", "data", "tvc"),
            Field("dtvclimx", 0.0, "real", "data", "tvc"),
            Field("wntvc", 0.0, "real", "data", "tvc"),
            Field("zettvc", 0.0, "real", "data", "tvc"),
            Field("pdynmc_gtvc36", 0.0, "real", "data", "tvc"),
            Field("gtvc0", 0.0, "real", "data", "tvc"),
            Field("parm", 0.0, "real", "data", "tvc"),
            Field("FPB", zeros3, "vec", "out", "tvc"),
            Field("FMPB", zeros3, "vec", "out", "tvc"),
            Field("etax", 0.0, "real", "diag", "tvc", plot),
            Field("zetx", 0.0, "real", "diag", "tvc", plot),
            Field("etacx", 0.0, "real", "diag", "tvc"),
            Field("zetcx", 0.0, "real", "diag", "tvc"),
            Field("gtvc", 0.0, "real", "out", "tvc"),
            Field("etasd", 0.0, "real", "state", "tvc"),
            Field("zetad", 0.0, "real", "state", "tvc"),
            Field("etas", 0.0, "real", "state", "tvc"),
            Field("zeta", 0.0, "real", "state", "tvc"),
            Field("detasd", 0.0, "real", "state", "tvc"),
            Field("dzetad", 0.0, "real", "state", "tvc"),
            Field("detas", 0.0, "real", "state", "tvc"),
            Field("dzeta", 0.0, "real", "state", "tvc"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mtvc = store.get("mtvc")
        if mtvc != 0:
            raise ValueError(f"mtvc={mtvc!r} not supported in this slice")
        zeros = np.zeros(3)
        store.set("FPB", zeros)
        store.set("FMPB", zeros)

    def terminate(self, vehicle, ctx):
        pass
