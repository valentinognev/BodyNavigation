from cadac.eom.flat6 import Flat6Newton
from cadac.kernel.events import EventEngine
from cadac.kernel.state import StateStore
from cadac.vehicles.sraam6.actuator import Sraam6Actuator
from cadac.vehicles.sraam6.aero import Sraam6Aero
from cadac.vehicles.sraam6.control import Sraam6Control
from cadac.vehicles.sraam6.environment import Sraam6Environment
from cadac.vehicles.sraam6.euler import Sraam6Euler
from cadac.vehicles.sraam6.forces import Sraam6Forces
from cadac.vehicles.sraam6.guidance import Sraam6Guidance
from cadac.vehicles.sraam6.intercept import Sraam6Intercept
from cadac.vehicles.sraam6.kinematics import Sraam6Kinematics
from cadac.vehicles.sraam6.propulsion import Sraam6Propulsion
from cadac.vehicles.sraam6.seeker import Sraam6Seeker
from cadac.vehicles.sraam6.tvc import Sraam6Tvc


class Sraam6Missile:
    type = "MISSILE6"

    def __init__(self, name, aero_deck, prop_deck, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Sraam6Environment(),
            Sraam6Kinematics(),
            Sraam6Aero(aero_deck),
            Sraam6Propulsion(prop_deck),
            Sraam6Seeker(),
            Sraam6Guidance(),
            Sraam6Control(),
            Sraam6Actuator(),
            Sraam6Forces(),
            Sraam6Euler(),
            Flat6Newton(),
            Sraam6Intercept(),
            Sraam6Tvc(),
        ]

    def define(self):
        store = self.store
        orig_define = store.define

        def define_skip_if_exists(field):
            if field.name not in store:
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
