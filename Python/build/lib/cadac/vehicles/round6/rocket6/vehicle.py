from cadac.eom.round6 import (
    Round6Environment,
    Round6Euler,
    Round6Kinematics,
    Round6Newton,
)
from cadac.kernel.events import EventEngine
from cadac.kernel.state import StateStore
from cadac.vehicles.round6.rocket6.aero import Rocket6Aero
from cadac.vehicles.round6.rocket6.control import Rocket6Control
from cadac.vehicles.round6.rocket6.forces import Rocket6Forces
from cadac.vehicles.round6.rocket6.gps import Rocket6Gps
from cadac.vehicles.round6.rocket6.guidance import Rocket6Guidance
from cadac.vehicles.round6.rocket6.ins import Rocket6Ins
from cadac.vehicles.round6.rocket6.intercept import Rocket6Intercept
from cadac.vehicles.round6.rocket6.propulsion import Rocket6Propulsion
from cadac.vehicles.round6.rocket6.rcs import Rocket6Rcs
from cadac.vehicles.round6.rocket6.startrack import Rocket6Startrack
from cadac.vehicles.round6.rocket6.tvc import Rocket6Tvc


class Rocket6:
    type = "HYPER6"
    family = "rocket6"

    def __init__(self, name, aero_deck, events=None, weather_deck=None, prop_deck=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Round6Kinematics(),
            Round6Environment(weather_deck),
            Rocket6Propulsion(),
            Rocket6Aero(aero_deck),
            Rocket6Gps(),
            Rocket6Startrack(),
            Rocket6Ins(),
            Rocket6Guidance(),
            Rocket6Control(),
            Rocket6Rcs(),
            Rocket6Tvc(),
            Rocket6Forces(),
            Round6Newton(),
            Round6Euler(),
            Rocket6Intercept(),
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
