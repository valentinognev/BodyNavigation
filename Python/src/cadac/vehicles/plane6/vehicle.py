from cadac.eom.flat6 import (
    Flat6Environment,
    Flat6Euler,
    Flat6Kinematics,
    Flat6Newton,
)
from cadac.kernel.events import EventEngine
from cadac.kernel.state import StateStore
from cadac.vehicles.plane6.actuator import Plane6Actuator
from cadac.vehicles.plane6.aero import Plane6Aero
from cadac.vehicles.plane6.control import Plane6Control
from cadac.vehicles.plane6.forces import Plane6Forces
from cadac.vehicles.plane6.guidance import Plane6Guidance
from cadac.vehicles.plane6.propulsion import Plane6Propulsion


class Plane6:
    type = "PLANE6"

    def __init__(self, name, aero_deck, prop_deck, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Flat6Environment(),
            Flat6Kinematics(),
            Plane6Aero(aero_deck),
            Plane6Propulsion(prop_deck),
            Plane6Guidance(),
            Plane6Forces(),
            Plane6Control(),
            Plane6Actuator(),
            Flat6Euler(),
            Flat6Newton(),
        ]

    def define(self):
        store = self.store
        orig_define = store.define

        def define_skip_if_exists(field):
            if field.name not in store.names():
                orig_define(field)

        store.define = define_skip_if_exists
        try:
            for module in self.modules:
                module.define(self)
        finally:
            store.define = orig_define
        self.com_names = [
            name
            for name in self.store.names()
            if "com" in self.store.field(name).outputs
        ]
