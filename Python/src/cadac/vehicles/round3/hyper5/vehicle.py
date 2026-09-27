from cadac.eom.round3 import Round3Environment, Round3Newton
from cadac.kernel.events import EventEngine
from cadac.kernel.state import StateStore
from cadac.vehicles.round3.hyper5.aero import Hyper5Aero
from cadac.vehicles.round3.hyper5.control import Hyper5Control
from cadac.vehicles.round3.hyper5.forces import Hyper5Forces
from cadac.vehicles.round3.hyper5.guidance import Hyper5Guidance
from cadac.vehicles.round3.hyper5.intercept import Hyper5Intercept
from cadac.vehicles.round3.hyper5.propulsion import Hyper5Propulsion
from cadac.vehicles.round3.hyper5.seeker import Hyper5Seeker
from cadac.vehicles.round3.hyper5.targeting import Hyper5Targeting


class Hyper5:
    type = "HYPER5"

    def __init__(self, name, aero_deck, prop_deck, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Round3Environment(),
            Hyper5Aero(aero_deck),
            Hyper5Propulsion(prop_deck),
            Hyper5Forces(),
            Round3Newton(),
            Hyper5Seeker(),
            Hyper5Guidance(),
            Hyper5Control(),
            Hyper5Intercept(),
            Hyper5Targeting(),
        ]

    def define(self):
        for module in self.modules:
            module.define(self)
        self.com_names = [
            name
            for name in self.store.names()
            if "com" in self.store.field(name).outputs
        ]
