from cadac.eom.flat3 import Flat3Environment, Flat3Kinematics, Flat3Newton
from cadac.kernel.events import EventEngine
from cadac.kernel.state import StateStore
from cadac.vehicles.flat3.falcon5.aero import Plane5Aero
from cadac.vehicles.flat3.falcon5.control import Plane5Control
from cadac.vehicles.flat3.falcon5.forces import Plane5Forces
from cadac.vehicles.flat3.falcon5.guidance import Plane5Guidance
from cadac.vehicles.flat3.falcon5.intercept import Plane5Intercept
from cadac.vehicles.flat3.falcon5.propulsion import Plane5Propulsion


class Plane5:
    type = "PLANE"

    def __init__(self, name, aero_deck, prop_deck, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Flat3Environment(),
            Flat3Kinematics(),
            Plane5Aero(aero_deck),
            Plane5Propulsion(prop_deck),
            Plane5Guidance(),
            Plane5Control(),
            Plane5Forces(),
            Flat3Newton(),
            Plane5Intercept(),
        ]

    def define(self):
        for module in self.modules:
            module.define(self)
        self.com_names = [
            name
            for name in self.store.names()
            if "com" in self.store.field(name).outputs
        ]
