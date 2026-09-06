from cadac.eom.flat3 import Flat3Environment, Flat3Kinematics, Flat3Newton
from cadac.kernel.events import EventEngine
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.aim5.aero import Aim5Aero
from cadac.vehicles.aim5.control import Aim5Control
from cadac.vehicles.aim5.forces import Aim5Forces
from cadac.vehicles.aim5.guidance import Aim5Guidance
from cadac.vehicles.aim5.intercept import Aim5Intercept
from cadac.vehicles.aim5.propulsion import Aim5Propulsion
from cadac.vehicles.aim5.seeker import Aim5Seeker

_COM_EXTRA = ("SBEL", "VBEL", "psivlx", "thtvlx")


def _sael_fields():
    return (
        Field("sael1", 0.0, "real", "init", "newton"),
        Field("sael2", 0.0, "real", "init", "newton"),
        Field("sael3", 0.0, "real", "init", "newton"),
        Field("dvae", 0.0, "real", "init", "newton"),
    )


class Aim5Flat3Newton(Flat3Newton):
    def define(self, vehicle):
        super().define(vehicle)
        store = vehicle.store
        for field in _sael_fields():
            if field.name not in store.names():
                store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        names = store.names()
        if "sael1" in names:
            store.set("sbel1", store.get("sael1"))
        if "sael2" in names:
            store.set("sbel2", store.get("sael2"))
        if "sael3" in names:
            store.set("sbel3", store.get("sael3"))
        if "dvae" in names:
            store.set("dvbe", store.get("dvae"))
        super().initialize(vehicle, ctx)


def _define_aim5_vehicle(vehicle):
    store = vehicle.store
    orig_define = store.define

    def define_skip_if_exists(field):
        if field.name not in store.names():
            orig_define(field)

    store.define = define_skip_if_exists
    try:
        for module in vehicle.modules:
            module.define(vehicle)
        for field in _sael_fields():
            store.define(field)
    finally:
        store.define = orig_define
    flagged = [
        name
        for name in store.names()
        if "com" in store.field(name).outputs
    ]
    extra = [name for name in _COM_EXTRA if name not in flagged]
    vehicle.com_names = flagged + extra


class Aim5:
    type = "AIM5"

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
            Aim5Aero(aero_deck),
            Aim5Propulsion(prop_deck),
            Aim5Seeker(),
            Aim5Guidance(),
            Aim5Control(),
            Aim5Forces(),
            Aim5Flat3Newton(),
            Aim5Intercept(),
        ]

    def define(self):
        _define_aim5_vehicle(self)
