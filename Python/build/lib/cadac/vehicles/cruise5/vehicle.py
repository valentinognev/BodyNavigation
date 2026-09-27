from cadac.eom.round3 import Round3Environment, Round3Newton
from cadac.kernel.events import EventEngine
from cadac.kernel.state import StateStore
from cadac.vehicles.cruise5.aero import Cruise5Aero
from cadac.vehicles.cruise5.control import Cruise5Control
from cadac.vehicles.cruise5.forces import Cruise5Forces
from cadac.vehicles.cruise5.guidance import Cruise5Guidance
from cadac.vehicles.cruise5.intercept import Cruise5Intercept
from cadac.vehicles.cruise5.propulsion import Cruise5Propulsion
from cadac.vehicles.cruise5.seeker import Cruise5Seeker
from cadac.vehicles.cruise5.targeting import Cruise5Targeting


class Cruise5:
    type = "CRUISE3"

    def __init__(self, name, aero_deck, prop_deck, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Round3Environment(),
            Cruise5Aero(aero_deck),
            Cruise5Propulsion(prop_deck),
            Cruise5Forces(),
            Round3Newton(),
            Cruise5Targeting(),
            Cruise5Seeker(),
            Cruise5Guidance(),
            Cruise5Control(),
            Cruise5Intercept(),
        ]

    def define(self):
        for module in self.modules:
            module.define(self)
        self.com_names = [
            name
            for name in self.store.names()
            if "com" in self.store.field(name).outputs
        ]
