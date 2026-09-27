from cadac.eom.round3 import Round3Environment, Round3Newton
from cadac.kernel.events import EventEngine
from cadac.kernel.state import StateStore
from cadac.vehicles.cruise3.aero import Cruise3Aero
from cadac.vehicles.cruise3.forces import Cruise3Forces
from cadac.vehicles.cruise3.propulsion import Cruise3Propulsion


class Cruise3:
    type = "CRUISE3"

    def __init__(self, name, aero_deck, prop_deck, events=None):
        self.name = name
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Round3Environment(),
            Cruise3Aero(aero_deck),
            Cruise3Propulsion(prop_deck),
            Cruise3Forces(),
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
