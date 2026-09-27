from cadac.eom.flat6 import Flat6Euler, Flat6Newton
from cadac.kernel.events import EventEngine
from cadac.kernel.state import StateStore
from cadac.vehicles.agm6.actuator import Agm6Actuator
from cadac.vehicles.agm6.aero import Agm6Aero
from cadac.vehicles.agm6.control import Agm6Control
from cadac.vehicles.agm6.datalink import Agm6Datalink
from cadac.vehicles.agm6.environment import Agm6Environment
from cadac.vehicles.agm6.forces import Agm6Forces
from cadac.vehicles.agm6.guidance import Agm6Guidance
from cadac.vehicles.agm6.ins import Agm6Ins
from cadac.vehicles.agm6.intercept import Agm6Intercept
from cadac.vehicles.agm6.kinematics import Agm6Kinematics
from cadac.vehicles.agm6.propulsion import Agm6Propulsion
from cadac.vehicles.agm6.sensor import Agm6Sensor


class Agm6Missile:
    type = "MISSILE6"

    def __init__(self, name, aero_deck, events=None, weather_deck=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Agm6Environment(weather_deck),
            Agm6Kinematics(),
            Agm6Aero(aero_deck),
            Agm6Propulsion(),
            Agm6Forces(),
            Flat6Euler(),
            Flat6Newton(),
            Agm6Ins(),
            Agm6Datalink(),
            Agm6Sensor(),
            Agm6Guidance(),
            Agm6Control(),
            Agm6Actuator(),
            Agm6Intercept(),
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
