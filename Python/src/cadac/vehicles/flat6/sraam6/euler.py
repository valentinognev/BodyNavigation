from cadac.constants import DEG
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field


class Sraam6Euler:
    name = "euler"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("ppd", 0.0, "real", "state", "euler"),
            Field("pp", 0.0, "real", "state", "euler"),
            Field("qqd", 0.0, "real", "state", "euler"),
            Field("qq", 0.0, "real", "state", "euler"),
            Field("rrd", 0.0, "real", "state", "euler"),
            Field("rr", 0.0, "real", "state", "euler"),
            Field("ppx", 0.0, "real", "out", "euler", ("plot",)),
            Field("qqx", 0.0, "real", "out", "euler", ("plot",)),
            Field("rrx", 0.0, "real", "out", "euler", ("plot",)),
            Field("WBEB", (0.0, 0.0, 0.0), "vec", "diag", "euler"),
        ):
            if field.name not in store:
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
        dt = ctx.int_step

        ppd_new = fmb[0] / ai11
        pp = integrate(ppd_new, ppd, pp, dt)
        ppd = ppd_new

        qqd_new = ((ai33 - ai11) * pp * rr + fmb[1]) / ai33
        qq = integrate(qqd_new, qqd, qq, dt)
        qqd = qqd_new

        rrd_new = (-(ai33 - ai11) * pp * qq + fmb[2]) / ai33
        rr = integrate(rrd_new, rrd, rr, dt)
        rrd = rrd_new

        store.set("ppd", ppd)
        store.set("pp", pp)
        store.set("qqd", qqd)
        store.set("qq", qq)
        store.set("rrd", rrd)
        store.set("rr", rr)
        store.set("WBEB", (pp, qq, rr))
        store.set("ppx", pp * DEG)
        store.set("qqx", qq * DEG)
        store.set("rrx", rr * DEG)

    def terminate(self, vehicle, ctx):
        pass
