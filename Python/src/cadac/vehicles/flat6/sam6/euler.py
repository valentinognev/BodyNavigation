import numpy as np

from cadac.constants import DEG
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field


class Sam6Euler:
    name = "euler"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("plot",)
        for field in (
            Field("ppd", 0.0, "real", "state", "euler"),
            Field("pp", 0.0, "real", "state", "euler"),
            Field("qqd", 0.0, "real", "state", "euler"),
            Field("qq", 0.0, "real", "state", "euler"),
            Field("rrd", 0.0, "real", "state", "euler"),
            Field("rr", 0.0, "real", "state", "euler"),
            Field("ppx", 0.0, "real", "out", "euler", plot),
            Field("qqx", 0.0, "real", "out", "euler", plot),
            Field("rrx", 0.0, "real", "out", "euler", plot),
            Field("WBEB", zeros3, "vec", "diag", "euler"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        fmb = store.get("FMB")
        ai11 = store.get("ai11")
        ai33 = store.get("ai33")
        ppd = store.get("ppd")
        pp = store.get("pp")
        qqd = store.get("qqd")
        qq = store.get("qq")
        rrd = store.get("rrd")
        rr = store.get("rr")
        int_step = ctx.int_step

        fmb1 = fmb[0]
        fmb2 = fmb[1]
        fmb3 = fmb[2]

        ppd_new = fmb1 / ai11
        pp = integrate(ppd_new, ppd, pp, int_step)
        ppd = ppd_new

        qqd_new = ((ai33 - ai11) * pp * rr + fmb2) / ai33
        qq = integrate(qqd_new, qqd, qq, int_step)
        qqd = qqd_new

        rrd_new = (-(ai33 - ai11) * pp * qq + fmb3) / ai33
        rr = integrate(rrd_new, rrd, rr, int_step)
        rrd = rrd_new

        wbeb = np.array([pp, qq, rr], dtype=float)
        ppx = pp * DEG
        qqx = qq * DEG
        rrx = rr * DEG

        store.set("ppd", ppd)
        store.set("pp", pp)
        store.set("qqd", qqd)
        store.set("qq", qq)
        store.set("rrd", rrd)
        store.set("rr", rr)
        store.set("ppx", ppx)
        store.set("qqx", qqx)
        store.set("rrx", rrx)
        store.set("WBEB", wbeb)

    def terminate(self, vehicle, ctx):
        pass
