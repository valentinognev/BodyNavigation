from cadac.kernel.events import EventEngine
from cadac.kernel.state import StateStore
from cadac.vehicles.sam6.actuator import Sam6Actuator
from cadac.vehicles.sam6.aero import Sam6Aero
from cadac.vehicles.sam6.control import Sam6Control
from cadac.vehicles.sam6.environment import Sam6Environment
from cadac.vehicles.sam6.euler import Sam6Euler
from cadac.vehicles.sam6.forces import Sam6Forces
from cadac.vehicles.sam6.guidance import Sam6Guidance
from cadac.vehicles.sam6.ins import Sam6Ins
from cadac.vehicles.sam6.intercept import Sam6Intercept
from cadac.vehicles.sam6.kinematics import Sam6Kinematics
from cadac.vehicles.sam6.newton import Sam6Newton
from cadac.vehicles.sam6.propulsion import Sam6Propulsion
from cadac.vehicles.sam6.rcs import Sam6Rcs
from cadac.vehicles.sam6.sensor import Sam6Sensor
from cadac.vehicles.sam6.tvc import Sam6Tvc


class Sam6Missile:
    type = "MISSILE6"

    def __init__(self, name, aero_deck, prop_deck, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Sam6Environment(),
            Sam6Kinematics(),
            Sam6Propulsion(prop_deck),
            Sam6Aero(aero_deck),
            Sam6Ins(),
            Sam6Sensor(),
            Sam6Guidance(),
            Sam6Control(),
            Sam6Actuator(),
            Sam6Tvc(),
            Sam6Rcs(),
            Sam6Forces(),
            Sam6Euler(),
            Sam6Newton(),
            Sam6Intercept(),
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
