import numpy as np

from cadac.eom.flat3 import Flat3Kinematics
from cadac.kernel.events import EventEngine
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.agm6.flat3io import Agm6Flat3Environment, Agm6Flat3Newton


class Agm6TargetForces:
    name = "forces"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        com = ("com",)
        store.define(Field("FSPA", zeros3, "vec", "out", "forces"))
        if "FSPV" not in store.names():
            store.define(Field("FSPV", zeros3, "vec", "out", "forces"))
        for field in (
            Field("aax", 0.0, "real", "diag", "forces", com),
            Field("alx", 0.0, "real", "diag", "forces", com),
            Field("anx", 0.0, "real", "diag", "forces", com),
            Field("acc_longx", 0.0, "real", "data", "forces"),
            Field("acc_latx", 0.0, "real", "data", "forces"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        acc_longx = store.get("acc_longx")
        acc_latx = store.get("acc_latx")
        grav = store.get("grav")
        fspa = np.array(
            [acc_longx * grav, acc_latx * grav, -grav],
            dtype=float,
        )
        store.set("FSPA", fspa)
        store.set("FSPV", fspa)
        store.set("aax", fspa[0] / grav)
        store.set("alx", fspa[1] / grav)
        store.set("anx", -fspa[2] / grav)

    def terminate(self, vehicle, ctx):
        pass


class Agm6Target:
    type = "TARGET3"

    def __init__(self, name, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Agm6Flat3Environment(),
            Flat3Kinematics(),
            Agm6TargetForces(),
            Agm6Flat3Newton(),
        ]

    def define(self):
        for module in self.modules:
            module.define(self)
        self.com_names = [
            name
            for name in self.store.names()
            if "com" in self.store.field(name).outputs
        ]
