from cadac.kernel.events import EventEngine
from cadac.kernel.state import StateStore
from cadac.vehicles.round3.rocket3.aerodynamics import Rocket3Aerodynamics
from cadac.vehicles.round3.rocket3.environment import Rocket3Environment
from cadac.vehicles.round3.rocket3.forces import Rocket3Forces
from cadac.vehicles.round3.rocket3.newton import Rocket3Newton
from cadac.vehicles.round3.rocket3.propulsion import Rocket3Propulsion


class Rocket3:
    type = "ROCKET3"

    def __init__(self, name, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Rocket3Environment(),
            Rocket3Propulsion(),
            Rocket3Aerodynamics(),
            Rocket3Forces(),
            Rocket3Newton(),
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
