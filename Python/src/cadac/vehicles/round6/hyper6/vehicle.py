from cadac.eom.round6 import (
    Round6Environment,
    Round6Euler,
    Round6Kinematics,
    Round6Newton,
)
from cadac.kernel.events import EventEngine
from cadac.kernel.state import StateStore
from cadac.vehicles.round6.hyper6.actuator import Hyper6Actuator
from cadac.vehicles.round6.hyper6.aero import Hyper6Aero
from cadac.vehicles.round6.hyper6.control import Hyper6Control
from cadac.vehicles.round6.hyper6.datalink import Hyper6Datalink
from cadac.vehicles.round6.hyper6.forces import Hyper6Forces
from cadac.vehicles.round6.hyper6.gps import Hyper6Gps
from cadac.vehicles.round6.hyper6.guidance import Hyper6Guidance
from cadac.vehicles.round6.hyper6.ins import Hyper6Ins
from cadac.vehicles.round6.hyper6.intercept import Hyper6Intercept
from cadac.vehicles.round6.hyper6.propulsion import Hyper6Propulsion
from cadac.vehicles.round6.hyper6.rcs import Hyper6Rcs
from cadac.vehicles.round6.hyper6.seeker import Hyper6Seeker
from cadac.vehicles.round6.hyper6.startrack import Hyper6Startrack


class Hyper6:
    type = "HYPER6"

    def __init__(self, name, aero_deck, prop_deck, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Round6Kinematics(),
            Round6Environment(),
            Hyper6Aero(aero_deck),
            Hyper6Propulsion(prop_deck),
            Hyper6Gps(),
            Hyper6Startrack(),
            Hyper6Ins(),
            Hyper6Datalink(),
            Hyper6Seeker(),
            Hyper6Guidance(),
            Hyper6Control(),
            Hyper6Actuator(),
            Hyper6Rcs(),
            Hyper6Forces(),
            Round6Newton(),
            Round6Euler(),
            Hyper6Intercept(),
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
