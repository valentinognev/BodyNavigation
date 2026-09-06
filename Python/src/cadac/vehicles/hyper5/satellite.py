import numpy as np

from cadac.eom.round3 import Round3Environment, Round3Newton
from cadac.kernel.events import EventEngine
from cadac.kernel.state import Field, StateStore


class Satellite3Forces:
    name = "forces"

    def define(self, vehicle):
        store = vehicle.store
        if "FSPV" not in store.names():
            store.define(Field("FSPV", (0.0, 0.0, 0.0), "vec", "out", "forces"))
        store.define(Field("sat_thrust", 0.0, "real", "data", "forces"))
        store.define(Field("sat_mass", 100.0, "real", "data", "forces"))

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        sat_thrust = store.get("sat_thrust")
        sat_mass = store.get("sat_mass")
        store.set("FSPV", np.array([sat_thrust / sat_mass, 0.0, 0.0]))

    def terminate(self, vehicle, ctx):
        pass


class Satellite3:
    type = "SATELLITE3"

    def __init__(self, name, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Round3Environment(),
            Satellite3Forces(),
            Round3Newton(),
        ]

    def define(self):
        for module in self.modules:
            module.define(self)
        self.com_names = [
            name
            for name in self.store.names()
            if "com" in self.store.field(name).outputs
        ]
