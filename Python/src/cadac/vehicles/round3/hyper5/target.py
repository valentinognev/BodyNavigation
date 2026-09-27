import numpy as np

from cadac.eom.round3 import Round3Environment, Round3Newton
from cadac.kernel.events import EventEngine
from cadac.kernel.state import Field, StateStore


class Target3Forces:
    name = "forces"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        if "FSPV" not in store:
            store.define(Field("FSPV", zeros3, "vec", "out", "forces"))
        for field in (
            Field("fwd_accel", 0.0, "real", "data", "forces"),
            Field("side_accel", 0.0, "real", "data", "forces"),
            Field("CORIO_V", zeros3, "vec", "diag", "forces"),
            Field("CENTR_V", zeros3, "vec", "diag", "forces"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        fwd_accel = store.get("fwd_accel")
        side_accel = store.get("side_accel")
        grav = store.get("grav")
        tgv = store.get("tgv")
        tig = store.get("tig")
        tge = store.get("tge")
        weii = store.get("weii")
        vbeg = store.get("vbeg")
        sbii = store.get("sbii")
        tvg = tgv.T
        tgi = tig.T
        teg = tge.T
        weig = tge @ weii @ teg
        corio_v = tvg @ weig @ vbeg * 2
        centr_v = tvg @ weig @ weig @ tgi @ sbii
        grav_g = np.zeros(3)
        grav_g[2] = grav
        grav_v = tvg @ grav_g
        acc_v = corio_v + centr_v - grav_v
        fspv = np.array(
            [acc_v[0] + fwd_accel, acc_v[1] + side_accel, acc_v[2]]
        )
        store.set("FSPV", fspv)
        store.set("CORIO_V", corio_v)
        store.set("CENTR_V", centr_v)

    def terminate(self, vehicle, ctx):
        pass


class Target3Intercept:
    name = "intercept"

    def define(self, vehicle):
        vehicle.store.define(
            Field("targ_health", 0, "int", "diag", "intercept")
        )

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        targ_health = ctx.combus[ctx.vehicle_slot].status
        vehicle.store.set("targ_health", targ_health)

    def terminate(self, vehicle, ctx):
        pass


class Target3:
    type = "TARGET3"

    def __init__(self, name, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Round3Environment(),
            Target3Forces(),
            Round3Newton(),
            Target3Intercept(),
        ]

    def define(self):
        for module in self.modules:
            module.define(self)
        self.com_names = [
            name
            for name in self.store.names()
            if "com" in self.store.field(name).outputs
        ]
